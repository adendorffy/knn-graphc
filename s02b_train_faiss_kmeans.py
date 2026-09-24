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

    subset_idx = None
    if config.subsample is not None and config.subsample < features.shape[0]:
        print(f"Subsampling {config.subsample} features from {features.shape[0]} total")
        cache_dir = Path(str(config.output_dir).replace("output/clustering", "output/subsample_indices"))
        subset_idx = get_subsample_indices(features.shape[0], config.subsample, seed=config.subsample_seed, cache_dir=cache_dir)
        features = features[subset_idx]
        config.output_dir = config.output_dir.parent / f"{config.output_dir.name}-subsample-{config.subsample}"
        config.output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(config.features_dir, show_progress=config.show_progress)
    features = scaler.transform(features)
    features = pca.transform(features)
    features = np.ascontiguousarray(features, dtype=np.float32)

    d = features.shape[1]
    ngpu = faiss.get_num_gpus()

    res = [faiss.StandardGpuResources() for _ in range(max(ngpu, 1))]
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True

    if ngpu > 1:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, cfg) for i in range(ngpu)]
        index = faiss.IndexShards(d)
        for idx in indexes:
            index.add_shard(idx)
    else:
        index = faiss.GpuIndexFlatL2(res[0], d, cfg)


    with track_resources() as usage:
        clustering = faiss.Clustering(d, config.num_clusters)
        clustering.niter = 15
        clustering.nredo = 1
        clustering.verbose = True
        clustering.init_method = faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS

        clustering.train(features, index)

        _, index_out = index.search(features, 1)
        labels = index_out.flatten().astype(np.int64, copy=False)
        centroids = faiss.vector_to_array(clustering.centroids).reshape(config.num_clusters, d)


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
        centroids=centroids,
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