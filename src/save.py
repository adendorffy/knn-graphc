from pathlib import Path

import numpy as np
from tqdm import tqdm

def set_output_dir_metadata(
    output_dir: Path,
    metadata: dict | None = None,
) -> None:
    """
    Save metadata to output_dir/metadata.json.
    """
    if metadata is not None:
        clustering_kind = metadata.get("clustering", "unknown")
        info_string = clustering_kind
        if "edges" in metadata:
            edges_info = metadata["edges"]
            edges_str = f"knn_{edges_info['num_neighbors']}" if edges_info.get("num_neighbors") else "full"
            min_sim = edges_info.get("min_sim")
            resolution = metadata.get("resolution", None)
            info_string += f"_{edges_str}_tau_{min_sim}_gamma_{resolution:.4f}"

        final_cluster_count = metadata.get("final_cluster_count", None)
        runtime = metadata.get("runtime_seconds", None)
        peak_ram = metadata.get("peak_ram_mb", None)
        if final_cluster_count is not None and runtime is not None and peak_ram is not None:
            info_string += f"_clusters_{final_cluster_count}_runtime_{runtime}_peakram_{peak_ram}"
        output_dir = output_dir / info_string
    
    return output_dir

def save_clustering_artifacts(
    *,
    output_dir: Path,
    features: np.ndarray,
    labels: list[int] | np.ndarray,
    centroids: np.ndarray | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Save clustering artifacts to disk.
    """
    output_dir = set_output_dir_metadata(output_dir, metadata)
    output_dir = Path(str(output_dir).replace("output/clustering", "output/clustering-artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if features is not None:
        features_path = output_dir / "reference_features.npy"
        if features_path.exists():
            return 
        print(f"Saving reference features to {features_path}...")
        np.save(features_path, features)

    if labels is not None   :
        labels_path = output_dir / "reference_labels.npy"
        print(f"Saving reference labels to {labels_path}...")
        np.save(labels_path, labels)
    
    if centroids is not None:
        centroids_path = output_dir / "centroids.npy"
        print(f"Saving cluster centroids to {centroids_path}...")
        np.save(centroids_path, centroids)

def save_labeled_segments_from_membership(
    *,
    features_dir: Path,
    segments_dir: Path,
    output_dir: Path,
    labels: list[int] | np.ndarray,
    row_indices: np.ndarray | None = None,
    show_progress: bool = True,
    metadata: dict | None = None,
) -> None:

    output_dir = set_output_dir_metadata(output_dir, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_paths = sorted(features_dir.rglob("*.npy"))
    assert feature_paths, f"No feature files found under {features_dir}"

    labels = np.asarray(labels, dtype=np.int64)
    total_labels = len(labels)

    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        assert len(row_indices) == total_labels, \
            f"row_indices ({len(row_indices)}) must match labels ({total_labels})"
        assert np.all(np.diff(row_indices) > 0), "row_indices must be sorted/unique"

    current_idx = 0
    global_row = 0

    for feature_path in tqdm(
        feature_paths,
        desc="Saving labelled segment files",
        disable=not show_progress,
    ):
        rel_path = feature_path.relative_to(features_dir) 
        segments_path = segments_dir / rel_path
        output_path = output_dir / rel_path

        if not segments_path.exists():
            raise FileNotFoundError(f"Missing segment file: {segments_path}")

        features = np.load(feature_path, mmap_mode="r")
        segments = np.load(segments_path)

        if features.ndim == 1:
            features = features[None, :]

        if segments.ndim != 2 or segments.shape[1] != 3:
            raise ValueError(
                f"Expected segment file shape (N, 3), got {segments.shape} "
                f"for {segments_path}"
            )

        num_segments = features.shape[0]

        if len(segments) != num_segments:
            raise ValueError(
                f"Feature/segment mismatch for {rel_path}: "
                f"{num_segments} feature rows but {len(segments)} segment rows"
            )
        

        file_global_start = global_row
        file_global_end = global_row + num_segments

        if row_indices is None:
            file_labels = labels[current_idx : current_idx + num_segments]

            if len(file_labels) != num_segments:
                raise ValueError(
                    f"Not enough labels for {rel_path}: "
                    f"needed {num_segments}, got {len(file_labels)}"
                )

            labeled_segments = segments.copy()
            labeled_segments[:, 2] = file_labels

            current_idx += num_segments
        else:
            mask = (row_indices >= file_global_start) & (row_indices < file_global_end)
            n_selected = mask.sum()
            if n_selected == 0:
                global_row = file_global_end
                continue 

            local_idx = row_indices[mask] - file_global_start
            file_labels = labels[mask]

            labeled_segments = segments[local_idx].copy()
            labeled_segments[:, 2] = file_labels
            current_idx += n_selected
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, labeled_segments)
        global_row = file_global_end

    if current_idx != total_labels:
        raise ValueError(
            f"Used {current_idx} labels, but labels has {total_labels} entries"
        )