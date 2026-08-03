import gc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import igraph as ig
import joblib
import tyro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.cluster import ClusterConfig, partition_with_target_clusters
from src.edge import BuildEdgesConfig, build_edges_knn, build_edges_full
from src.load import load_features, get_subsample_indices
from src.save import save_labeled_segments_from_membership, save_clustering_artifacts
from src.track import track_resources


@dataclass
class TrainGraphClusteringConfig:
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

    edges: BuildEdgesConfig = field(default_factory=BuildEdgesConfig)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    show_progress: bool = True
    """Show progress bars and status output during pipeline stages."""

def train_graph_clustering(config: TrainGraphClusteringConfig) -> None:
    scaler: StandardScaler = joblib.load(config.scaler_path)
    pca: PCA = joblib.load(config.pca_path)

    features = load_features(config.features_dir, show_progress=config.show_progress)
    features = scaler.transform(features, copy=False)
    features_pca = pca.transform(features)
    features = features_pca
    del features_pca, pca, scaler
    gc.collect()

    if config.subsample is not None and config.subsample < features.shape[0]:
        print(f"Subsampling {config.subsample} features from {features.shape[0]} total")
        cache_dir = Path(str(config.output_dir).replace("output/clustering", "output/subsample_indices"))
        subset_idx = get_subsample_indices(features.shape[0], config.subsample, seed=config.subsample_seed, cache_dir=cache_dir)
        features = features[subset_idx]
        config.output_dir = config.output_dir.parent / f"{config.output_dir.name}-subsample-{config.subsample}"
        config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.edges.num_neighbors is None:
        with track_resources() as edge_usage:
            edges_indices, edges_weights = build_edges_full(
                features, config.edges, show_progress=config.show_progress
            )
    else:
        with track_resources() as edge_usage:
            edges_indices, edges_weights = build_edges_knn(
                features, config.edges, show_progress=config.show_progress
            )

    with track_resources() as cluster_usage:
        print("Building graph from edges...")
        graph = ig.Graph()
        graph.add_vertices(len(features))

        save_clustering_artifacts(
            output_dir=config.output_dir,
            features=features,
            labels=None,
            metadata={
                "features_dir": str(config.features_dir),
                "clustering": "graph",
                "edges": asdict(config.edges),
                "resolution": config.cluster.resolution_specified if config.cluster.resolution_specified is not None else None,
            },
        )
        del features
        gc.collect()

        graph.add_edges(edges_indices)
        del edges_indices
        gc.collect()

        graph.es["weight"] = edges_weights
        del edges_weights
        gc.collect()

        print("------ graph stats ------")
        num_vertices = graph.vcount()
        num_edges = graph.ecount()
        avg_degree = (2 * num_edges) / num_vertices if num_vertices > 0 else 0.0
        print(f"num nodes:      {num_vertices:,}")
        print(f"num edges:      {num_edges:,}")
        print(f"average degree: {avg_degree:.2f}")
        print("-------------------------")

        best_partition, best_res = partition_with_target_clusters(
            graph, config.cluster, show_progress=config.show_progress
        )
    
    runtime = round(edge_usage.duration_seconds + cluster_usage.duration_seconds, 2)
    peak_ram = round(max(edge_usage.peak_rss_mb, cluster_usage.peak_rss_mb), 1)

    save_labeled_segments_from_membership(
        features_dir=config.features_dir,
        segments_dir=config.segments_dir,
        output_dir=config.output_dir,
        labels=best_partition.membership,
        row_indices=subset_idx if config.subsample is not None else None,
        show_progress=config.show_progress,
        metadata={
            "features_dir": str(config.features_dir),
            "clustering": "graph",
            "edges": asdict(config.edges),
            "resolution": best_res,
            "final_cluster_count": len(set(best_partition.membership)),
            "runtime_seconds": runtime,
            "peak_ram_mb": peak_ram,
        }
    )
    save_clustering_artifacts(
        output_dir=config.output_dir,
        features=None,
        labels=best_partition.membership,
        metadata={
            "features_dir": str(config.features_dir),
            "clustering": "graph",
            "edges": asdict(config.edges),
            "resolution": best_res,
            "final_cluster_count": len(set(best_partition.membership)),
            "runtime_seconds": runtime,
            "peak_ram_mb": peak_ram,   
        }
    )

    print("Pipeline complete!")


if __name__ == "__main__":
    train_graph_clustering(tyro.cli(TrainGraphClusteringConfig))
