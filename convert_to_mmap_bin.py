from pathlib import Path
import numpy as np
from tqdm import tqdm

def build_ulm_mmap(
    inferred_segments_dir: str,
    output_path: str,
):
    inferred_segments_dir = Path(inferred_segments_dir)
    output_path = Path(output_path)

    paths = sorted(inferred_segments_dir.rglob("*.npy"))
    if not paths:
        raise RuntimeError(
            f"No .npy files found in {inferred_segments_dir}"
        )

    # ---------------------------------------------------------
    # First pass: inspect labels and determine total token count
    # ---------------------------------------------------------
    total_tokens = 0
    unique_labels = set()
    min_label = None
    max_label = None

    print(f"Found {len(paths):,} utterance files.")

    for path in tqdm(paths, desc="Inspecting segments", unit="file"):
        segments = np.load(path, mmap_mode="r")

        if segments.ndim != 2 or segments.shape[1] != 3:
            raise ValueError(
                f"{path}: expected shape (N, 3), got {segments.shape}"
            )

        labels = segments[:, 2].astype(np.int64)

        if np.any(labels < 0):
            raise ValueError(f"{path}: negative cluster IDs found")

        total_tokens += len(labels)

        if len(labels):
            file_min = int(labels.min())
            file_max = int(labels.max())

            min_label = (
                file_min if min_label is None
                else min(min_label, file_min)
            )
            max_label = (
                file_max if max_label is None
                else max(max_label, file_max)
            )

            unique_labels.update(np.unique(labels).tolist())

    unique_labels = np.array(sorted(unique_labels), dtype=np.int64)

    expected = np.arange(min_label, max_label + 1)
    contiguous = np.array_equal(unique_labels, expected)

    print()
    print(f"Utterances:       {len(paths):,}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Minimum label:    {min_label}")
    print(f"Maximum label:    {max_label}")
    print(f"Unique labels:    {len(unique_labels):,}")
    print(f"Labels contiguous: {contiguous}")

    if not contiguous:
        missing = np.setdiff1d(expected, unique_labels)

        print(f"Missing labels:     {missing[:50]}")

        if len(missing) > 50:
            print(f"... plus {len(missing) - 50} more")

        print(
            "Note: labels are not contiguous, but this is OK. "
            "The LM vocabulary size will be based on max_label + 1 "
            "so original cluster IDs are preserved."
        )

    vocab_size = max_label + 1

    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise RuntimeError(
            f"Vocabulary size {vocab_size} does not fit in uint16."
        )

    # ---------------------------------------------------------
    # Second pass: write flat mmap stream
    # ---------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mmap = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="w+",
        shape=(total_tokens,),
    )

    offset = 0

    for path in paths:
        segments = np.load(path)
        labels = segments[:, 2].astype(np.uint16)

        mmap[offset : offset + len(labels)] = labels
        offset += len(labels)

    mmap.flush()

    print()
    print(f"Saved:            {output_path}")
    print(f"VOCAB_SIZE:       {vocab_size}")
    print(
        f"Binary size:      "
        f"{output_path.stat().st_size / 1024**2:.1f} MiB"
    )


if __name__ == "__main__":
    build_ulm_mmap(
        inferred_segments_dir=(
            "output/inferred_segments/zerosyl/LibriSpeech/train-960/"
            "knn_1_hnswm_32_search_64/"
            "zerosyl/inference-test/LibriSpeech/train-clean-100/"
            "graph_knn_100_tau_0.45_gamma_0.0500_clusters_9693_"
            "runtime_1360.61_peakram_15978.8"
        ),
        output_path=(
            "output/ulm/"
            "LibriSpeech-train-960-graph_knn100-tau0.45-"
            "gamma0.0500-k9693.bin"
        ),
    )