from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tyro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import faiss

from src.load import load_features, get_subsample_indices
from src.save import save_labeled_segments_from_membership, save_clustering_artifacts
from src.track import track_resources

@dataclass
class TrainKmeansClusteringConfig:
    features_dir: Path
    """Directory of per-utterance .npy feature files."""

    segments_dir: Path
    """Directory of per-utterance .npy segment boundary files."""

    subsample: float | None = None
    """Subsample the features for edge building and clustering."""

    subsample_seed: int = 42
    """Random seed for subsampling."""

    scaler_path: Path | None = None
    """joblib-serialized sklearn StandardScaler."""

    pca_path: Path | None = None
    """joblib-serialized sklearn PCA model."""

    output_dir: Path = Path("output/clustering")
    """Directory for cluster labels and metadata."""

    num_clusters: int = 10_000
    """Number of clusters for KMeans."""

    show_progress: bool = True
    """Show progress bars and status output during pipeline stages."""
    
    
def train_kmeans(config: TrainKmeansClusteringConfig) -> None:
    scaler: StandardScaler = joblib.load(config.scaler_path)
    pca: PCA = joblib.load(config.pca_path)

    features = load_features(config.features_dir, show_progress=config.show_progress)
    features = scaler.transform(features)
    features = pca.transform(features)
    features = np.ascontiguousarray(features, dtype=np.float32)
    
    subset_idx = None
    if config.subsample is not None and config.subsample < features.shape[0]:
        print(f"Subsampling {config.subsample} features from {features.shape[0]} total")
        cache_dir = Path(str(config.output_dir).replace("output/clustering", "output/subsample_indices"))
        subset_idx = get_subsample_indices(features.shape[0], config.subsample, seed=config.subsample_seed, cache_dir=cache_dir)
        features = features[subset_idx]
        config.output_dir = config.output_dir.parent / f"{config.output_dir.name}-subsample-{config.subsample}"
        config.output_dir.mkdir(parents=True, exist_ok=True)

    with track_resources() as usage:
        acoustic_model = faiss.Kmeans(
            d=features.shape[1],
            k=config.num_clusters,
            niter=15,
            init_method=faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS,
            nredo=1,
            verbose=True,
            gpu=True,
        )
        acoustic_model.train(features)

        _, index = acoustic_model.index.search(features, 1)
        labels = index.flatten().astype(np.int64, copy=False)


    save_labeled_segments_from_membership(
        features_dir=config.features_dir,
        segments_dir=config.segments_dir,
        output_dir=config.output_dir,
        labels=labels,
        row_indices=subset_idx if config.subsample is not None else None,
        show_progress=config.show_progress,
        metadata={
            "features_dir": str(config.features_dir),
            "clustering": "kmeans++",
            "final_cluster_count": config.num_clusters,
            "runtime_seconds": round(usage.duration_seconds, 2),
            "peak_ram_mb": round(usage.peak_rss_mb, 1),
        }
    )
    save_clustering_artifacts(
        output_dir=config.output_dir,
        centroids=acoustic_model.centroids,
        features=None, 
        labels=labels,
        metadata={
            "features_dir": str(config.features_dir),
            "clustering": "kmeans++",
            "final_cluster_count": len(set(labels)),
            "runtime_seconds": round(usage.duration_seconds, 2),
            "peak_ram_mb": round(usage.peak_rss_mb, 1), 
        }
    )

if __name__ == "__main__":
    train_kmeans(tyro.cli(TrainKmeansClusteringConfig))