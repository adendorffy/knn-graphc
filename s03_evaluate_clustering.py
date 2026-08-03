from dataclasses import dataclass, fields
from pathlib import Path

import tyro
import pandas as pd

from src.eval.clustering import ClusteringMetrics, evaluate_clustering_metrics

@dataclass
class EvalClusteringConfig:
    segments_dir: Path
    """Directory of labeled segment .npy files from s03."""

    textgrid_dir: Path
    """Directory of reference TextGrid files."""

    compute_nes: bool = False
    """Whether to compute the normalized edit score (NES) metric."""

    save_to_csv: bool = False
    """Whether to save metrics to CSV file."""

    segments_pattern: str = "**/*.npy"
    """Glob pattern for segment files under segments_dir."""

    textgrid_pattern: str = "**/*.TextGrid"
    """Glob pattern for TextGrid files under textgrid_dir."""

    include_empty_intervals: bool = True
    """Include empty TextGrid intervals when reading references."""

    show_progress: bool = True
    """Show a tqdm progress bar while loading utterances."""


def print_clustering_metrics(metadata: dict, metrics: ClusteringMetrics) -> None:
    print("\nClustering evaluation")
    print("-" * 40)
    for key, value in metadata.items():
        print(f"{key:>22}  {value}")
        
    print("-" * 40)
    for field in fields(metrics):
        value = getattr(metrics, field.name)
        if str(value) == "nan":
            continue
        if isinstance(value, float):
            print(f"{field.name:>22}  {value*100:.2f}")
        else:
            print(f"{field.name:>22}  {value:,}")
    print("-" * 40)

def metadata_from_segments_dir(segments_dir: Path) -> dict:
    """
    Extract metadata from the segments_dir path.
    """
    metadata = {}
    info_string = segments_dir.name
    parts = info_string.split("_")
    if parts[0] == "graph":
        if parts[1] == "knn":
            metadata["clustering"] = "graph_knn"
            num_neighbours = int(parts[2])
            min_sim = float(parts[4])
            resolution = float(parts[6])
            if "inferred_segments" in str(segments_dir):
                metadata["clustering"] = "graph_knn_inferred"
                new_parts = segments_dir.parts[5].split("_")
                k_vote = int(new_parts[2])
                ef_search = int(new_parts[-1])
                metadata["k_vote"] = k_vote
                metadata["ef_search"] = ef_search

        elif parts[1] == "full":
            metadata["clustering"] = "graph_full"
            num_neighbours = None
            min_sim = float(parts[3])
            resolution = float(parts[5])

    elif parts[0] == "kmeans" or parts[0] == "kmeans++":
        metadata["clustering"] = "kmeans"
        num_neighbours = None
        min_sim = None
        resolution = None
        if "inferred_segments" in str(segments_dir):
            metadata["clustering"] = "kmeans_inferred"
    
    metadata["num_neighbours"] = num_neighbours
    metadata["min_sim"] = min_sim
    metadata["resolution"] = resolution
    metadata["num_clusters"] = int(parts[-5])
    metadata["runtime_seconds"] = float(parts[-3])
    metadata["peak_ram_mb"] = float(parts[-1])

    return metadata

def save_to_csv(metrics: ClusteringMetrics, metadata: dict, csv_path: Path) -> None:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        new_row = {field.name: getattr(metrics, field.name) for field in fields(metrics)}
        new_row.update(metadata)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Appending clustering metrics to {csv_path}")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_dict = {field.name: getattr(metrics, field.name) for field in fields(metrics)}
        metrics_dict.update(metadata)
        df = pd.DataFrame([metrics_dict])
    df.to_csv(csv_path, index=False)
    print(f"Saved clustering metrics to {csv_path}")

def eval_clustering(config: EvalClusteringConfig) -> ClusteringMetrics:
    metrics = evaluate_clustering_metrics(
        config.segments_dir,
        config.textgrid_dir,
        compute_nes=config.compute_nes,
        segments_pattern=config.segments_pattern,
        textgrid_pattern=config.textgrid_pattern,
        include_empty_intervals=config.include_empty_intervals,
        show_progress=config.show_progress,
    )

    metadata = metadata_from_segments_dir(config.segments_dir)
    if config.save_to_csv:
        if "inferred_segments" in str(config.segments_dir):
            csv_dir = Path(str(config.segments_dir).replace("output/inferred_segments", "output/evaluation")).parent
        else:
            csv_dir = Path(str(config.segments_dir).replace("output/clustering", "output/evaluation")).parent
        print(f"Saving clustering metrics to CSV in {csv_dir}")
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "evaluation.csv"
        save_to_csv(metrics, metadata, csv_path)
    else:
        print_clustering_metrics(metadata, metrics)
    return metrics


if __name__ == "__main__":
    eval_clustering(tyro.cli(EvalClusteringConfig))
