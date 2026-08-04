from dataclasses import dataclass
from pathlib import Path

import faiss
import joblib
import numpy as np
import tyro
from tqdm import tqdm
import os

from src.track import track_resources

@dataclass
class InferLabelsConfig:
    clustering_method: str
    """Clustering method used to generate the reference labels (e.g., 'kmeans')."""

    clustering_dir: Path
    """Directory of clustering output from s02."""

    reference_dir: Path
    """Directory of reference features, labels and/or centroids from s03."""

    features_dir: Path
    """Directory of per-utterance raw .npy feature files to label."""

    segments_dir: Path
    """Directory of per-utterance segment boundary .npy files from s01."""

    models_dir: Path | None = None
    """Directory with joblib-serialized sklearn StandardScaler and PCA models."""

    num_clusters: int | None = None
    """If set, assert the reference labels cover this many clusters."""

    hnsw_m: int = 32
    """Number of HNSW bi-directional links per node (controls graph connectivity)."""

    hnsw_ef_search: int = 64
    """HNSW efSearch parameter: larger values improve recall at the cost of speed."""

    k_neighbors: int = 3
    """Number of nearest reference segments to vote over for each query."""

    show_progress: bool = True
    """Show a progress bar while labeling utterances."""

def _row_mode(row: np.ndarray) -> np.int64:
    values, counts = np.unique(row, return_counts=True)
    return values[np.argmax(counts)]

def majority_vote(neighbor_labels: np.ndarray) -> np.ndarray:
    if neighbor_labels.shape[1] == 1:
        return neighbor_labels[:, 0]

    if neighbor_labels.shape[1] == 3:
        a = neighbor_labels[:, 0]
        b = neighbor_labels[:, 1]
        c = neighbor_labels[:, 2]
        return np.where((a == b) | (a == c), a, np.where(b == c, b, a))

    return np.array([_row_mode(row) for row in neighbor_labels], dtype=np.int64)

def output_is_complete(output_path: Path, expected_len: int | None = None) -> bool:
    if not output_path.exists():
        return False

    try:
        arr = np.load(output_path, mmap_mode="r")
    except Exception:
        return False

    if arr.ndim != 2 or arr.shape[1] != 3:
        return False

    if expected_len is not None and arr.shape[0] != expected_len:
        return False

    return True

def infer_labels(config: InferLabelsConfig) -> None:

    scaler = joblib.load(config.models_dir / "scaler.joblib") if config.models_dir is not None else None
    pca = joblib.load(config.models_dir / "pca.joblib") if config.models_dir is not None else None

    reference_labels = None
    faiss.omp_set_num_threads(os.cpu_count())

    if config.clustering_method == "kmeans++":
        centroids = np.load(
            config.reference_dir / "centroids.npy"
        ).astype(np.float32, copy=False)
        num_clusters = len(centroids)

        centroids = np.ascontiguousarray(centroids, dtype=np.float32)

        if config.num_clusters is not None:
            assert len(centroids) == config.num_clusters, (
                len(centroids),
                config.num_clusters,
            )

        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(centroids)

    elif config.clustering_method == "graph_knn":
        reference_labels = np.load(
            config.reference_dir / "reference_labels.npy"
        ).astype(np.int64, copy=False)

        if config.num_clusters is not None:
            num_clusters = len(np.unique(reference_labels))
            assert num_clusters == config.num_clusters, (
                num_clusters,
                config.num_clusters,
            )

        reference_features = np.load(
            config.reference_dir / "reference_features.npy"
        ).astype(np.float32, copy=False)

        assert len(reference_features) == len(reference_labels), (
            len(reference_features),
            len(reference_labels),
        )

        reference_features = np.ascontiguousarray(
            reference_features,
            dtype=np.float32,
        )
        faiss.normalize_L2(reference_features)

        dim = reference_features.shape[1]

        index = faiss.IndexHNSWFlat(
            dim,
            config.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efSearch = config.hnsw_ef_search
        index.add(reference_features)

    else:
        raise ValueError(
            f"Unknown clustering method: {config.clustering_method}"
        )

    base = config.features_dir.parent
    dataset_dirs = sorted(base.glob(config.features_dir.name))

    feature_entries = []
    for dataset_dir in dataset_dirs:
        segment_dataset_dir = (
            config.segments_dir.parent / dataset_dir.name
        )

        for feature_path in dataset_dir.rglob("*.npy"):
            feature_entries.append(
                (dataset_dir, segment_dataset_dir, feature_path)
            )

    assert feature_entries, f"No .npy files found in {config.features_dir}"

    features_info = config.features_dir.parts[2:]  
    output_segments_dir = Path("output/inferred_segments") / Path(*features_info) / f"knn_{config.k_neighbors}_hnswm_{config.hnsw_m}_search_{config.hnsw_ef_search}" / config.clustering_dir.relative_to(Path("output/clustering"))
    if config.clustering_method == "kmeans++":
        output_segments_dir = Path("output/inferred_segments") / Path(*features_info) / f"kmeans++_{num_clusters}" / config.clustering_dir.relative_to(Path("output/clustering"))
    output_segments_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving inferred segment labels to {output_segments_dir}")

    max_batch_segments = 5_000_000

    batch_features = []
    batch_segments = []
    batch_output_paths = []
    batch_lengths = []
    current_batch_size = 0

    def flush_batch():
        nonlocal batch_features, batch_segments, batch_output_paths, batch_lengths, current_batch_size

        if not batch_features:
            return

        x = np.vstack(batch_features).astype(np.float32, copy=False)
        x = np.ascontiguousarray(x, dtype=np.float32)

        if config.clustering_method == "kmeans++":
            _, inferred_all = index.search(x, 1)
            inferred_all = inferred_all[:, 0].astype(np.int64, copy=False)

        elif config.clustering_method == "graph_knn":
            faiss.normalize_L2(x)
            _, neighbor_ids = index.search(x, config.k_neighbors)

            if np.any(neighbor_ids < 0):
                raise RuntimeError(
                    "FAISS returned invalid neighbour IDs. "
                    "Try increasing hnsw_ef_search or reducing k_neighbors."
                )

            neighbor_labels = reference_labels[neighbor_ids]
            inferred_all = majority_vote(neighbor_labels)

        else:
            raise ValueError(
                f"Unknown clustering method: {config.clustering_method}"
            )

        start = 0
        for n, segments, output_path in zip(
            batch_lengths,
            batch_segments,
            batch_output_paths,
        ):
        
            inferred = inferred_all[start:start + n]
            start += n

            labeled_segments = segments.copy()
            labeled_segments[:, 2] = inferred

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, labeled_segments)

        assert start == len(inferred_all)

        batch_features = []
        batch_segments = []
        batch_output_paths = []
        batch_lengths = []
        current_batch_size = 0

    with track_resources() as usage_tracker:
        for dataset_dir, segment_dataset_dir, feature_path in tqdm(
            feature_entries, 
            desc="Inferring labels", 
            disable=not config.show_progress
        ):
            rel_path = feature_path.relative_to(dataset_dir)

            segments_path = segment_dataset_dir / rel_path
            output_path = output_segments_dir / rel_path

            segments = np.load(segments_path)
            assert segments.ndim == 2 and segments.shape[1] == 3, segments.shape

            if output_is_complete(output_path, expected_len=len(segments)):
                continue

            features = np.load(feature_path)

            if scaler is not None:
                features = scaler.transform(features)

            if pca is not None:
                features = pca.transform(features)

            features = features.astype(np.float32, copy=False)

            assert segments.ndim == 2 and segments.shape[1] == 3, segments.shape
            assert len(features) == len(segments), (
                feature_path,
                len(features),
                len(segments),
            )

            batch_features.append(features)
            batch_segments.append(segments)
            batch_output_paths.append(output_path)
            batch_lengths.append(len(features))
            current_batch_size += len(features)

            if current_batch_size >= max_batch_segments:
                flush_batch()

        flush_batch()

        print(f"Saved inferred segment labels to {output_segments_dir}")

    runtime = usage_tracker.duration_seconds
    peak_memory = usage_tracker.peak_rss_mb
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Peak memory usage: {peak_memory:.2f} MB")

if __name__ == "__main__":
    infer_labels(tyro.cli(InferLabelsConfig))