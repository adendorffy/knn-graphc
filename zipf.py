import matplotlib.pyplot as plt
import textgrids
import numpy as np
from tqdm import tqdm
import tyro

from collections import Counter
from pathlib import Path
from dataclasses import dataclass
import os

from src.eval.clustering import pair_segment_and_textgrid_paths, labels_from_segment_file

@dataclass
class ZipfConfig:
    segments_dir: Path
    """Directory of labeled segment .npy files from s03."""

    textgrid_dir: Path
    """Directory of reference TextGrid files."""

    output_path: Path
    """Path to save the Zipf plot."""

    save_as_png: bool = False
    """Whether to save the plot as png."""

    matching_method: str = "kmeans"
    """Method used for matching clusters to phones."""

    not_clusters: bool = False
    """Whether to plot the Zipf distribution of phones instead of induced clusters."""

WAVLM_FRAME_RATE = 50.0  # Frame rate for WAVLM features

def set_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{times}",
        "font.size": 9,
        "axes.labelsize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 8,
    })
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

def set_output_path(
    output_path: str | Path,
    segments_dir: str | Path,
    save_as_png: bool,
    not_clusters: bool,
) -> Path:
    output_path = Path(output_path)
    suffix = ".png" if save_as_png else ".pdf"

    if not_clusters:
        output_path = output_path / f"zipf_comparison_plot{suffix}"
    elif output_path.is_dir():
        output_path = output_path / f"{segments_dir.relative_to('output')}{suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def get_color(stem):
    if "full" in stem:
        return "tab:blue"
    elif "kmeans" in stem:
        return "tab:gray"
    elif "knn" in stem:
        return "tab:orange"
    else:
        return "tab:green"

def basic_method_label(stem):
    stem = stem.lower()
    if "kmeans" in stem:
        return "K-means"
    elif "full" in stem:
        return "Full Graph"
    elif "knn" in stem:
        return "kNN Graph"
    else:
        return "Induced clusters"

def extract_threshold(path):
    if "tau" in path.stem:
        threshold = path.stem.split("tau")[1].split("_")[0]
        threshold = float(threshold)
    else:
        threshold = np.inf
    return float(threshold)

def extract_resolution(path):
    if "gamma" in path.stem:
        resolution = path.stem.split("gamma")[1].split("_")[0]
        resolution = float(resolution)
    else:
        resolution = np.inf
    return float(resolution)

LINESTYLE = {
    "K-means":          "--",
    "Full Graph":       "-",
    "kNN Graph":        "-.",
    "Induced clusters": "--",
}

def get_phone_seq(segments_dir: str | Path,
    textgrid_dir: str | Path,
    segments_pattern: str = "**/*.npy",
    textgrid_pattern: str = "**/*.TextGrid",
    frame_rate: float = WAVLM_FRAME_RATE,
    include_empty_intervals: bool = True,
    show_progress: bool = True,
):

    pairs = pair_segment_and_textgrid_paths(
        Path(segments_dir),
        Path(textgrid_dir),
        segments_pattern=segments_pattern,
        textgrid_pattern=textgrid_pattern,
    )

    labels_pred: list[int] = []
    phone_sequences: list[tuple[str, ...]] = []

    for segment_path, textgrid_path in tqdm(
        pairs,
        desc="Extracting phone sequences",
        disable=not show_progress
    ):
        pred, _, phones = labels_from_segment_file(
            segment_path,
            textgrid_path,
            frame_rate=frame_rate,
            include_empty_intervals=include_empty_intervals,
        )

        labels_pred.extend(pred)
        phone_sequences.extend(phones)

    return labels_pred, phone_sequences

def rank_freq(counts):
    """Sort descending, normalize to probabilities, return (ranks, probs)."""
    freqs = np.array(sorted(counts, reverse=True))
    probs = freqs / freqs.sum()
    ranks = np.arange(1, len(probs) + 1)
    return ranks, probs

def create_comparison_zipf_plot_phones(
    phone_sequences: list[tuple[str, ...]],
    output_path: str | Path,
    xlabel: str = "Rank",
    ylabel: str = "Normalised Frequency",
):
    phones = []
    biphones = []
    triphones = []
    for seq in phone_sequences:
        phones.extend(seq)
        for i in range(len(seq) - 1):
            biphones.append((seq[i], seq[i + 1]))
        for i in range(len(seq) - 2):
            triphones.append((seq[i], seq[i + 1], seq[i + 2]))

    legend_handles = {}
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
    legend_order = ["Phones", "Biphones", "Triphones"]

    # Discovered phonetic sequence distribution
    for seq_type, seq_data in [("Phones", phones), ("Biphones", biphones), ("Triphones", triphones)]:
        seq_counts = Counter(seq_data)
        seq_ranks, seq_probs = rank_freq(list(seq_counts.values()))

        line, = ax.loglog(
            seq_ranks,
            seq_probs,
            linestyle="-",
            linewidth=1,
            drawstyle="steps-post",
            label=seq_type,
            zorder=3,
        )
        legend_handles.setdefault(seq_type, line)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.tick_params(direction="in", which="both", length=5)
    
    ax.legend(
        [legend_handles[label] for label in legend_order if label in legend_handles],
        [label for label in legend_order if label in legend_handles],
        loc="upper right",
        frameon=False,
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def create_comparison_zipf_plot_induced_clusters(
    phone_sequences: list[tuple[str, ...]],
    induced_clusterings: dict[str, list[int]],
    output_path: str | Path,
    xlabel: str = "Rank",
    ylabel: str = "Normalised Frequency",
):
    legend_handles = {}
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)

    # Discovered phonetic sequence distribution
    seq_counts = Counter(phone_sequences)
    seq_ranks, seq_probs = rank_freq(list(seq_counts.values()))

    line, = ax.loglog(
        seq_ranks,
        seq_probs,
        linestyle="-",
        color="black",
        linewidth=1,
        drawstyle="steps-post",
        label="Phone sequences",
        zorder=3,
    )
    legend_handles.setdefault("Phone sequences", line)
    print(f"Number of unique phone sequences: {len(seq_counts)}")

    # Induced cluster distribution (from predicted labels)
    for method_name, cluster_labels in induced_clusterings.items():
        cluster_counts = Counter(cluster_labels)
        cluster_ranks, cluster_probs = rank_freq(list(cluster_counts.values()))

        method_label = basic_method_label(method_name)
        color = get_color(method_name.lower())
        linestyle = LINESTYLE.get(method_label, "--")
        print(f"Method: {method_label}, Number of clusters: {len(cluster_counts)}")

        line, = ax.loglog(
            cluster_ranks,
            cluster_probs,
            label=method_label,
            color=color,
            linewidth=1,
            drawstyle="steps-post",
            linestyle=linestyle,
        )
        legend_handles.setdefault(method_label, line)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.tick_params(direction="in", which="both", length=5)

    legend_order = ["Phone sequences"]
    for method_name in induced_clusterings.keys():
        method_label = basic_method_label(method_name)
        if method_label not in legend_order:
            legend_order.append(method_label)
    
    ax.legend(
        [legend_handles[label] for label in legend_order if label in legend_handles],
        [label for label in legend_order if label in legend_handles],
        loc="upper right",
        frameon=False,
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def main(config: ZipfConfig):

    set_plot_style()
    output_path = set_output_path(config.output_path, config.segments_dir, save_as_png=config.save_as_png, not_clusters=config.not_clusters)
    
    if "*" in str(config.segments_dir):
        parts = config.segments_dir.parts
        wildcard_idx = next(
            i for i, part in enumerate(parts)
            if "*" in part or "?" in part or "[" in part
        )
        base = Path(*parts[:wildcard_idx])
        rest = "/".join(parts[wildcard_idx:])
        segments_dirs = sorted(base.glob(f"{rest}/*"))
    else:
        segments_dirs = [config.segments_dir.glob({"*"})]  


    print(f"How many subdirs: {len(segments_dirs)}")
    induced_clusterings = {}
    for dir in segments_dirs:
        if dir.is_dir() and not any(dir.iterdir()):
            print(f"Warning: Directory {dir} is empty. Skipping.")
        method_name = dir.name.split("_")[0]
        if method_name == "graph":
            method_name = "_".join(dir.name.split("_")[:2])
        
        if method_name != config.matching_method:
            print(f"Skipping directory {dir.name} as it does not match the specified matching method '{config.matching_method}'.")
            continue
        
        labels_pred, phone_sequences = get_phone_seq(
            segments_dir=dir,
            textgrid_dir=config.textgrid_dir,
            segments_pattern="**/*.npy",
            textgrid_pattern="**/*.TextGrid",
            frame_rate=WAVLM_FRAME_RATE,
            include_empty_intervals=True,
            show_progress=True,
        )
        if method_name in induced_clusterings:
            induced_clusterings[method_name].extend(labels_pred)
        else:   
            induced_clusterings[method_name] = labels_pred
        if config.not_clusters:
            break  # Only need to process one directory for phone sequences

    assert phone_sequences, "No phone sequences were extracted. Please check the input directories and patterns."

    if config.not_clusters:
        create_comparison_zipf_plot_phones(
            phone_sequences=phone_sequences,
            output_path=output_path,
        )
    
    else:
        create_comparison_zipf_plot_induced_clusters(
            phone_sequences=phone_sequences,
            induced_clusterings=induced_clusterings,
            output_path=output_path,
        )
    print(f"Zipf comparison plot saved to {output_path}")

if __name__ == "__main__":
    main(tyro.cli(ZipfConfig))