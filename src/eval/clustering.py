"""Evaluate discrete segment clusters using PAcc / iPAcc / F1-PAcc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import re

import numpy as np
import tgt
from tqdm import tqdm


WAVLM_FRAME_RATE = 50
IGNORE_TOKENS = {"sil", "spn", "sp", "<unk>", ""}


@dataclass(frozen=True)
class PAccMetrics:
    pacc: float
    ipacc: float
    f1_pacc: float

@dataclass(frozen=True)
class NESMetrics:
    wnes: float
    iwnes: float
    f1_wnes: float

@dataclass(frozen=True)
class ClusteringMetrics(PAccMetrics, NESMetrics):
    num_segments: int
    num_utterances: int
    num_clusters: int
    num_clusters_all: int
    num_gold_types: int


def edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Simple Levenshtein distance for token tuples."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    return int(dp[n, m])


def normalized_edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return edit_distance(a, b) / denom


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def clean_tokens(tokens: list[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\d", "", t) for t in tokens if t.lower() not in IGNORE_TOKENS)


def check_boundary(
    gold_start: float,
    gold_end: float,
    disc_start: float,
    disc_end: float,
) -> bool:
    if gold_start <= disc_start and gold_end >= disc_end:
        return True
 
    gold_duration = round(gold_end - gold_start, 2)
    if gold_duration <= 0:
        return False
 
    overlap_start = max(gold_start, disc_start)
    overlap_end = min(gold_end, disc_end)
    overlap_duration = round(overlap_end - overlap_start, 2)
 
    if overlap_duration <= 0:
        return False
 
    overlap_percentage = overlap_duration / gold_duration
    duration_condition = gold_duration >= 0.06 and overlap_duration >= 0.03
    percentage_condition = gold_duration < 0.06 and overlap_percentage >= 0.5
    return duration_condition or percentage_condition

def matching_phone_intervals(
    phone_tier: tgt.Tier,
    seg_start: float,
    seg_end: float,
) -> tuple[str, ...]:
    """Cleaned, check_boundary-matched gold phone sequence for a discovered segment."""
    candidates = phone_tier.get_annotations_between_timepoints(
        start=seg_start,
        end=seg_end,
        left_overlap=True,
        right_overlap=True,
    )
    matched_texts = [
        iv.text for iv in candidates
        if check_boundary(iv.start_time, iv.end_time, seg_start, seg_end)
    ]
    return clean_tokens(matched_texts)

def pair_segment_and_textgrid_paths(
    segments_dir: Path,
    textgrid_dir: Path,
    *,
    segments_pattern: str = "**/*.npy",
    textgrid_pattern: str = "**/*.TextGrid",
) -> list[tuple[Path, Path]]:

    if "*" in str(segments_dir) or "*" in str(textgrid_dir):
        subdirs = sorted(textgrid_dir.parent.glob(textgrid_dir.name))
        textgrid_paths = []
        for subdir in subdirs:
            textgrid_paths.extend(sorted(subdir.glob(textgrid_pattern)))
    else:
        textgrid_paths = sorted(textgrid_dir.glob(textgrid_pattern))
    print(f"Searching for segments in {segments_dir}/{segments_pattern}")
    segment_paths = sorted(segments_dir.glob(segments_pattern))

    if not segment_paths:
        raise FileNotFoundError(f"No segment files found under {segments_dir}")
    if not textgrid_paths:
        raise FileNotFoundError(f"No TextGrid files found under {textgrid_dir}")

    textgrid_by_stem = {p.stem: p for p in textgrid_paths}

    pairs: list[tuple[Path, Path]] = []
    missing = []

    for segment_path in segment_paths:
        textgrid_path = textgrid_by_stem.get(segment_path.stem)
        if textgrid_path is None:
            missing.append(segment_path.stem)
            continue
        pairs.append((segment_path, textgrid_path))

    # if missing:
        # raise ValueError(
        #     f"Missing TextGrids for {len(missing)} segment files. "
        #     f"First missing stem: {missing[0]}"
        # )

    return pairs


def labels_from_segment_file(
    segment_path: Path,
    textgrid_path: Path,
    *,
    frame_rate: float = WAVLM_FRAME_RATE,
    include_empty_intervals: bool = True,
) -> tuple[list[int], list[tuple[str, ...]]]:
    """
    For each discovered segment, return:
      - predicted cluster id
      - overlapping gold phone sequence
    """
    segments = np.load(segment_path)

    if segments.ndim != 2 or segments.shape[-1] != 3:
        raise ValueError(
            f"Expected shape (N, 3) in {segment_path}, got {tuple(segments.shape)}"
        )

    textgrid = tgt.read_textgrid(
        str(textgrid_path),
        include_empty_intervals=include_empty_intervals,
    )
    phone_tier = textgrid.get_tier_by_name("phones")

    labels_pred: list[int] = []
    phone_sequences: list[tuple[str, ...]] = []
    labels_pred_with_empty: list[int] = []

    for start_frame, end_frame, cluster in segments.tolist():
        start_sec = start_frame / frame_rate
        end_sec = end_frame / frame_rate

        labels_pred_with_empty.append(int(cluster))

        phone_seq = matching_phone_intervals(phone_tier, start_sec, end_sec)
        if len(phone_seq) == 0:
            continue

        labels_pred.append(int(cluster))
        phone_sequences.append(phone_seq)

    return labels_pred, labels_pred_with_empty, phone_sequences


def distances_to_modal_sequence(
    group: list[tuple[str, ...]],
    *,
    reverse: bool = False,
) -> list[float]:
    if len(group) == 1:
        return [] if reverse else [0.0]

    modes = statistics.multimode(group)

    if len(modes) == 1:
        modal = modes[0]
        length = len(modal)
        assert length > 0
        return [edit_distance(x, modal) / length for x in group]

    max_len = len(max(modes, key=len))
    longest_modes = [m for m in modes if len(m) == max_len]
    assert max_len > 0

    min_total_dist = float("inf")
    best_distances = None

    for candidate in longest_modes:
        current_distances = [
            edit_distance(x, candidate) / max_len
            for x in group
        ]
        current_total = np.sum(current_distances)

        if current_total < min_total_dist:
            min_total_dist = current_total
            best_distances = current_distances

    assert best_distances is not None
    return best_distances

    return [normalized_edit_distance(x, modal) for x in group]


def per_from_clusters(
    labels_pred: list[int],
    phone_sequences: list[tuple[str, ...]],
) -> float:
    """
    Forward PER:
    For each discovered cluster, compare its phone sequences to the modal
    phone sequence of that cluster.
    """
    by_cluster: dict[int, list[tuple[str, ...]]] = defaultdict(list)

    for cluster, phone_seq in zip(labels_pred, phone_sequences):
        by_cluster[cluster].append(phone_seq)

    distances: list[float] = []

    for group in by_cluster.values():
        distances.extend(distances_to_modal_sequence(group, reverse=False))

    return float(np.mean(distances)) if distances else 0.0


def reverse_per_from_gold_types(
    labels_pred: list[int],
    phone_sequences: list[tuple[str, ...]],
) -> float:
    """
    Reverse PER:
    For each gold phone sequence/type, compare the cluster assignments
    associated with that type.

    Here each discovered segment contributes a one-token cluster sequence,
    e.g. cluster 42 becomes ("42",). This is the direct analogue of checking
    whether the same gold phonetic realization maps to consistent cluster ids.
    """
    by_gold_type: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)

    for cluster, phone_seq in zip(labels_pred, phone_sequences):
        by_gold_type[phone_seq].append((str(cluster),))

    distances: list[float] = []

    for group in by_gold_type.values():
        distances.extend(distances_to_modal_sequence(group, reverse=True))

    return float(np.mean(distances)) if distances else 0.0

# ---------------------------------------------------------------------------
# wNES / iwNES / F1-wNES
#
# Unlike PER/reverse-PER above (which compare every item in a group to the
# group's *modal* sequence), NES is based on *pairwise* normalized edit
# distances within each group, then weighted by the number of elements in
# the group ("per-element weighted NED" in Malan et al.). wNES = 1 - weighted
# NED, iwNES = 1 - weighted reverse-NED, and F1-wNES is their harmonic mean.
# ---------------------------------------------------------------------------
 
 
def pairwise_mean_normalized_distance(
    group: list[tuple[str, ...]],
) -> float | None:
    """
    Mean pairwise normalized edit distance over all C(n, 2) pairs in a group.
    Returns None for singleton groups, since pairwise distance is undefined
    (the caller decides whether to credit or pop singletons).
 
    Optimization: identical sequences always have distance 0, so instead of
    computing edit distance for every one of the C(n, 2) pairs, we collapse
    the group into its distinct sequences (via Counter) and only run edit
    distance on distinct-vs-distinct pairs -- O(k^2) edit-distance calls
    where k = number of distinct sequences, instead of O(n^2). Same-sequence
    pairs are added back analytically (they're always 0, so they only affect
    the denominator). This is exact, not an approximation: for repeat-heavy
    groups (e.g. a gold word realized identically hundreds of times, or a
    cluster id repeated across a gold word's occurrences) k is often tiny
    relative to n.
    """
    n = len(group)
    if n < 2:
        return None
 
    total_pairs = n * (n - 1) // 2
    counts = Counter(group)
    items = list(counts.items())
 
    cross_dist_sum = 0.0
    for i in range(len(items)):
        seq_i, count_i = items[i]
        for j in range(i + 1, len(items)):
            seq_j, count_j = items[j]
            cross_dist_sum += normalized_edit_distance(seq_i, seq_j) * count_i * count_j
 
    return cross_dist_sum / total_pairs


def ned_from_clusters(
    labels_pred: list[int],
    phone_sequences: list[tuple[str, ...]],
) -> float:
    """
    Forward NED, weighted per element:
    For each discovered cluster, compute the mean pairwise normalized edit
    distance between its phone sequences. Each cluster's value is weighted
    by its number of segments (larger clusters count more). Singleton
    clusters are credited with distance 0 (they can't be internally
    inconsistent), matching the forward-PER convention above.
    """
    by_cluster: dict[int, list[tuple[str, ...]]] = defaultdict(list)
 
    for cluster, phone_seq in zip(labels_pred, phone_sequences):
        by_cluster[cluster].append(phone_seq)
 
    weights: list[int] = []
    means: list[float] = []
 
    for group in by_cluster.values():
        mean_dist = pairwise_mean_normalized_distance(group)
        weights.append(len(group))
        means.append(0.0 if mean_dist is None else mean_dist)
 
    return float(np.average(means, weights=weights)) if weights else 0.0

def reverse_ned_from_gold_types(
    labels_pred: list[int],
    phone_sequences: list[tuple[str, ...]],
) -> float:
    """
    Reverse NED, weighted per element:
    For each gold phone sequence/type, compute the mean pairwise normalized
    edit distance between the (string-ified) cluster ids assigned to its
    occurrences. Weighted by the number of occurrences of that gold type.
    Singleton gold types are popped (excluded), since a single occurrence
    has no pair to compare against.
    """
    by_gold_type: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
 
    for cluster, phone_seq in zip(labels_pred, phone_sequences):
        by_gold_type[phone_seq].append((str(cluster),))
 
    weights: list[int] = []
    means: list[float] = []
 
    for group in by_gold_type.values():
        mean_dist = pairwise_mean_normalized_distance(group)
        if mean_dist is None:
            continue  # singleton gold type: popped, not credited
        weights.append(len(group))
        means.append(mean_dist)
 
    return float(np.average(means, weights=weights)) if weights else 0.0

def evaluate_clustering_metrics(
    segments_dir: str | Path,
    textgrid_dir: str | Path,
    *,
    segments_pattern: str = "**/*.npy",
    textgrid_pattern: str = "**/*.TextGrid",
    compute_nes: bool = True,
    frame_rate: float = WAVLM_FRAME_RATE,
    include_empty_intervals: bool = True,
    show_progress: bool = True,
) -> ClusteringMetrics:
    """
    Compare predicted cluster ids to gold phone sequences from TextGrids.

    Expected segment file format:
      shape (N, 3), columns: start_frame, end_frame, cluster_id
    """
    pairs = pair_segment_and_textgrid_paths(
        Path(segments_dir),
        Path(textgrid_dir),
        segments_pattern=segments_pattern,
        textgrid_pattern=textgrid_pattern,
    )

    labels_pred: list[int] = []
    labels_pred_with_empty: list[int] = []
    phone_sequences: list[tuple[str, ...]] = []

    for segment_path, textgrid_path in tqdm(
        pairs,
        desc="Evaluating PAcc",
        disable=not show_progress,
    ):
        pred, pred_with_empty, phones = labels_from_segment_file(
            segment_path,
            textgrid_path,
            frame_rate=frame_rate,
            include_empty_intervals=include_empty_intervals,
        )

        labels_pred.extend(pred)
        labels_pred_with_empty.extend(pred_with_empty)
        phone_sequences.extend(phones)

    print(f"Found {len(set(labels_pred))} clusters across {len(labels_pred)} segments in {len(pairs)} utterances.")

    # Modal-sequence-based metrics (PAcc / iPAcc)
    per_value = per_from_clusters(labels_pred, phone_sequences)
    reverse_per_value = reverse_per_from_gold_types(labels_pred, phone_sequences)
 
    pacc = 1.0 - per_value
    ipacc = 1.0 - reverse_per_value
    f1_pacc = f1_score(pacc, ipacc)

    if compute_nes:
        # Compute only when tractable
        # Pairwise, per-element-weighted metrics (wNES / iwNES)
        ned_value = ned_from_clusters(labels_pred, phone_sequences)
        reverse_ned_value = reverse_ned_from_gold_types(labels_pred, phone_sequences)

        wnes = 1.0 - ned_value
        iwnes = 1.0 - reverse_ned_value
        f1_wnes = f1_score(wnes, iwnes)
        
        return ClusteringMetrics(
            pacc=pacc,
            ipacc=ipacc,
            f1_pacc=f1_pacc,
            wnes=wnes,
            iwnes=iwnes,
            f1_wnes=f1_wnes,
            num_segments=len(labels_pred),
            num_utterances=len(pairs),
            num_clusters=len(set(labels_pred)),
            num_clusters_all=len(set(labels_pred_with_empty)),
            num_gold_types=len(set(phone_sequences))
        )

    return ClusteringMetrics(
        pacc=pacc,
        ipacc=ipacc,
        f1_pacc=f1_pacc,
        wnes=float("nan"),
        iwnes=float("nan"),
        f1_wnes=float("nan"),
        num_segments=len(labels_pred),
        num_utterances=len(pairs),
        num_clusters=len(set(labels_pred)),
        num_clusters_all=len(set(labels_pred_with_empty)),
        num_gold_types=len(set(phone_sequences))
    )