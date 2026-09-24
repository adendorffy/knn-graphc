from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import gc
import os

import faiss
import joblib
import numpy as np
import torch
import tyro
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import OPTConfig, OPTForCausalLM


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class ScoreSLM21Config:
    # ---------------------------------------------------------
    # Pre-extracted SLM21 ZeroSyl features
    # ---------------------------------------------------------
    features_dir: Path
    """
    Directory containing batched SLM21 feature files, e.g.

        output/slm21/features/lexical/

    Expected files:

        batch_000000.npz
        batch_000001.npz
        ...

    Each .npz must contain:
        features
        offsets
        keys
        lengths
    """

    output_path: Path
    """Output file: one 'key loglikelihood' pair per line."""

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------
    tokenizer_method: str
    """
    Tokenizer type:

        graph_knn
        kmeans
    """

    models_dir: Path | None = None
    """
    Directory containing the scaler/PCA used by this tokenizer:

        scaler.joblib
        pca.joblib

    Set to None if no scaler/PCA should be applied.
    """

    tokenizer_dir: Path = Path(".")
    """
    Tokenizer artefact directory.

    For graph_knn, this directory must contain:

        reference_features.npy
        reference_labels.npy

    For kmeans, this directory must contain:

        centroids.npy
    """

    # ---------------------------------------------------------
    # Graph kNN settings
    # ---------------------------------------------------------
    k_neighbors: int = 1

    hnsw_m: int = 32

    hnsw_ef_search: int = 64

    # ---------------------------------------------------------
    # Language model
    # ---------------------------------------------------------
    checkpoint_path: str = "best.pt"
    """OPT LM checkpoint trained on this tokenizer's units."""

    device: str = "cuda:0"

    # ---------------------------------------------------------
    # LM evaluation
    # ---------------------------------------------------------
    lm_batch_size: int = 8
    """Number of encoded utterances scored together by OPT."""

    normalize: bool = False
    """
    Divide log likelihood by number of predicted units.

    For sWUGGY-style lexical scoring, normally leave False.
    """

    use_fp16: bool = True
    """Use FP16 for OPT inference on CUDA."""

    # ---------------------------------------------------------
    # Input / practical controls
    # ---------------------------------------------------------
    features_pattern: str = "batch_*.npz"

    resume: bool = True
    """Skip keys already present in output_path."""

    show_progress: bool = True


# =====================================================================
# General helpers
# =====================================================================


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def load_completed_keys(
    output_path: Path,
) -> set[str]:

    if not output_path.exists():
        return set()

    keys = set()

    with output_path.open() as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            key = line.split(maxsplit=1)[0]
            keys.add(key)

    return keys


# =====================================================================
# Scaler / PCA
# =====================================================================


def load_feature_transforms(
    models_dir: Path | None,
):
    if models_dir is None:
        print("No scaler/PCA requested.")
        return None, None

    scaler_path = models_dir / "scaler.joblib"

    pca_path = models_dir / "pca.joblib"

    scaler = None
    pca = None

    if scaler_path.exists():
        print(f"Loading scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)

    else:
        print(f"No scaler found at {scaler_path}")

    if pca_path.exists():
        print(f"Loading PCA: {pca_path}")
        pca = joblib.load(pca_path)

    else:
        print(f"No PCA found at {pca_path}")

    return scaler, pca


def transform_features(
    features: np.ndarray,
    scaler,
    pca,
) -> np.ndarray:

    x = features.astype(
        np.float32,
        copy=False,
    )

    if scaler is not None:
        x = scaler.transform(x)

    if pca is not None:
        x = pca.transform(x)

    return np.ascontiguousarray(
        x,
        dtype=np.float32,
    )


# =====================================================================
# Graph tokenizer
# =====================================================================


def _row_mode(
    row: np.ndarray,
) -> np.int64:

    values, counts = np.unique(
        row,
        return_counts=True,
    )

    return values[np.argmax(counts)]


def majority_vote(
    neighbor_labels: np.ndarray,
) -> np.ndarray:

    if neighbor_labels.shape[1] == 1:
        return neighbor_labels[:, 0]

    if neighbor_labels.shape[1] == 3:
        a = neighbor_labels[:, 0]
        b = neighbor_labels[:, 1]
        c = neighbor_labels[:, 2]

        return np.where(
            (a == b) | (a == c),
            a,
            np.where(
                b == c,
                b,
                a,
            ),
        )

    return np.asarray(
        [_row_mode(row) for row in neighbor_labels],
        dtype=np.int64,
    )


def build_graph_tokenizer(
    tokenizer_dir: Path,
    hnsw_m: int,
    hnsw_ef_search: int,
):
    print("Loading graph tokenizer...")

    reference_labels = np.load(tokenizer_dir / "reference_labels.npy").astype(
        np.int64,
        copy=False,
    )

    reference_features = np.load(tokenizer_dir / "reference_features.npy").astype(
        np.float32,
        copy=False,
    )

    if len(reference_features) != len(reference_labels):
        raise ValueError(
            "Reference feature/label "
            "length mismatch: "
            f"{len(reference_features)} vs "
            f"{len(reference_labels)}"
        )

    reference_features = np.ascontiguousarray(
        reference_features,
        dtype=np.float32,
    )

    faiss.normalize_L2(reference_features)

    dim = reference_features.shape[1]

    print(f"Reference segments: {len(reference_features):,}")

    print(f"Reference clusters: {len(np.unique(reference_labels)):,}")

    print(f"Feature dimension: {dim}")

    print(f"Building HNSW: M={hnsw_m}, efSearch={hnsw_ef_search}")

    index = faiss.IndexHNSWFlat(
        dim,
        hnsw_m,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efSearch = hnsw_ef_search

    index.add(reference_features)

    del reference_features
    gc.collect()

    return index, reference_labels


def tokenize_graph(
    features: np.ndarray,
    index,
    reference_labels: np.ndarray,
    k_neighbors: int,
) -> np.ndarray:

    x = np.ascontiguousarray(
        features,
        dtype=np.float32,
    )

    faiss.normalize_L2(x)

    _, neighbor_ids = index.search(
        x,
        k_neighbors,
    )

    if np.any(neighbor_ids < 0):
        raise RuntimeError("FAISS returned invalid neighbour IDs.")

    neighbor_labels = reference_labels[neighbor_ids]

    units = majority_vote(neighbor_labels)

    return units.astype(
        np.int64,
        copy=False,
    )


# =====================================================================
# K-means tokenizer
# =====================================================================


def build_kmeans_tokenizer(
    tokenizer_dir: Path,
):
    print("Loading K-means tokenizer...")

    centroids_path = tokenizer_dir / "centroids.npy"

    centroids = np.load(centroids_path).astype(
        np.float32,
        copy=False,
    )

    centroids = np.ascontiguousarray(
        centroids,
        dtype=np.float32,
    )

    print(f"Centroids: {len(centroids):,}")

    print(f"Feature dimension: {centroids.shape[1]}")

    index = faiss.IndexFlatL2(centroids.shape[1])

    index.add(centroids)

    return index


def tokenize_kmeans(
    features: np.ndarray,
    index,
) -> np.ndarray:

    x = np.ascontiguousarray(
        features,
        dtype=np.float32,
    )

    _, cluster_ids = index.search(
        x,
        1,
    )

    return cluster_ids[:, 0].astype(
        np.int64,
        copy=False,
    )


# =====================================================================
# Language model
# =====================================================================


def load_lm(
    checkpoint_path: str,
    device: torch.device,
    use_fp16: bool,
):
    print(f"Loading LM checkpoint: {checkpoint_path}")

    if is_url(str(checkpoint_path)):
        checkpoint = torch.hub.load_state_dict_from_url(
            str(checkpoint_path),
            map_location="cpu",
        )

    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    cfg = OPTConfig(**checkpoint["cfg"])

    model = OPTForCausalLM(cfg)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    if use_fp16 and device.type == "cuda":
        model = model.half()

    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {num_params:,}")

    print(f"LM vocabulary size: {model.config.vocab_size}")

    print(f"BOS ID: {model.config.bos_token_id}")

    print(f"PAD ID: {model.config.pad_token_id}")

    print(f"Maximum context: {model.config.max_position_embeddings}")

    return model


# =====================================================================
# LM batching / scoring
# =====================================================================


def make_lm_sequence(
    units: np.ndarray,
    bos_id: int,
):
    units_tensor = torch.from_numpy(
        units.astype(
            np.int64,
            copy=False,
        )
    )

    tokens = torch.cat(
        [
            torch.tensor(
                [bos_id],
                dtype=torch.long,
            ),
            units_tensor,
        ]
    )

    src = tokens[:-1]
    tgt = tokens[1:]

    return src, tgt


def score_lm_batch(
    examples: list[
        tuple[
            str,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    model,
    device: torch.device,
    normalize: bool,
) -> list[tuple[str, float]]:
    """
    Score one batch of variable-length
    unit sequences.
    """

    if not examples:
        return []

    keys = [item[0] for item in examples]

    src_list = [item[1] for item in examples]

    tgt_list = [item[2] for item in examples]

    seqlens = [len(src) for src in src_list]

    max_len = max(seqlens)

    if max_len > model.config.max_position_embeddings:
        raise RuntimeError(
            "Evaluation sequence exceeds "
            "OPT context window: "
            f"{max_len} > "
            f"{model.config.max_position_embeddings}. "
            "Add chunked long-sequence scoring "
            "if this occurs."
        )

    src_ids = pad_sequence(
        src_list,
        batch_first=True,
        padding_value=(model.config.pad_token_id),
    )

    tgt_ids = pad_sequence(
        tgt_list,
        batch_first=True,
        padding_value=(model.config.pad_token_id),
    )

    src_ids = src_ids.to(
        device,
        non_blocking=True,
    )

    tgt_ids = tgt_ids.to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():
        logits = model(src_ids).logits

        bsz = src_ids.size(0)

        losses = torch.nn.functional.cross_entropy(
            input=logits.reshape(
                -1,
                logits.size(-1),
            ),
            target=tgt_ids.reshape(-1),
            reduction="none",
        ).view(
            bsz,
            -1,
        )

    results = []

    for (
        b,
        (key, seqlen),
    ) in enumerate(
        zip(
            keys,
            seqlens,
        )
    ):
        ll = (
            -losses[
                b,
                :seqlen,
            ]
            .sum()
            .item()
        )

        if normalize:
            ll /= seqlen

        results.append(
            (
                key,
                ll,
            )
        )

    return results


# =====================================================================
# Read saved ZeroSyl feature batches
# =====================================================================


def iter_feature_batch(
    batch_path: Path,
):
    """
    Yield:

        key, utterance_features

    from one saved SLM21 .npz batch.
    """

    with np.load(
        batch_path,
        allow_pickle=False,
    ) as data:
        features = data["features"]

        offsets = data["offsets"]

        keys = data["keys"]

        if len(offsets) != len(keys) + 1:
            raise ValueError(f"{batch_path}: offset/key length mismatch")

        for i, key in enumerate(keys):
            start = int(offsets[i])

            end = int(offsets[i + 1])

            utterance_features = features[start:end]

            yield (
                str(key),
                utterance_features,
            )


# =====================================================================
# Main evaluator
# =====================================================================


def compute_scores(
    config: ScoreSLM21Config,
):
    device = torch.device(config.device)

    faiss.omp_set_num_threads(os.cpu_count() or 1)

    # ---------------------------------------------------------
    # Locate feature batches
    # ---------------------------------------------------------
    feature_batches = sorted(config.features_dir.glob(config.features_pattern))

    if not feature_batches:
        raise RuntimeError(
            f"No feature batches found "
            f"in {config.features_dir} "
            f"with pattern "
            f"{config.features_pattern}"
        )

    print(f"Found {len(feature_batches):,} feature batches.")

    # ---------------------------------------------------------
    # Load tokenizer-specific transform
    # ---------------------------------------------------------
    scaler, pca = load_feature_transforms(config.models_dir)

    # ---------------------------------------------------------
    # Build tokenizer
    # ---------------------------------------------------------
    method = config.tokenizer_method.lower()

    if method == "graph_knn":
        tokenizer_index, reference_labels = build_graph_tokenizer(
            tokenizer_dir=(config.tokenizer_dir),
            hnsw_m=(config.hnsw_m),
            hnsw_ef_search=(config.hnsw_ef_search),
        )

        max_tokenizer_id = int(reference_labels.max())

    elif method in (
        "kmeans",
        "kmeans++",
    ):
        tokenizer_index = build_kmeans_tokenizer(tokenizer_dir=(config.tokenizer_dir))

        reference_labels = None

        max_tokenizer_id = tokenizer_index.ntotal - 1

    else:
        raise ValueError("tokenizer_method must be 'graph_knn' or 'kmeans'.")

    # ---------------------------------------------------------
    # Load matching OPT language model
    # ---------------------------------------------------------
    model = load_lm(
        checkpoint_path=(config.checkpoint_path),
        device=device,
        use_fp16=(config.use_fp16),
    )

    # ---------------------------------------------------------
    # Critical vocabulary sanity check
    # ---------------------------------------------------------
    bos_id = model.config.bos_token_id

    if max_tokenizer_id >= bos_id:
        raise RuntimeError(
            "Tokenizer IDs overlap "
            "the LM special token space:\n"
            f"  max tokenizer ID = "
            f"{max_tokenizer_id}\n"
            f"  LM BOS ID        = "
            f"{bos_id}"
        )

    # ---------------------------------------------------------
    # Output/resume
    # ---------------------------------------------------------
    config.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed_keys(config.output_path) if config.resume else set()

    if completed:
        print(f"Resuming with {len(completed):,} already-scored items.")

    mode = "a" if config.resume else "w"

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------
    n_scored = 0
    n_skipped = 0
    n_empty = 0

    lm_examples = []

    # ---------------------------------------------------------
    # Helper to flush LM minibatch
    # ---------------------------------------------------------
    def flush_lm_batch(
        fout,
    ):
        nonlocal lm_examples
        nonlocal n_scored

        if not lm_examples:
            return

        scored = score_lm_batch(
            examples=lm_examples,
            model=model,
            device=device,
            normalize=(config.normalize),
        )

        for key, ll in scored:
            fout.write(f"{key} {ll}\n")

        n_scored += len(scored)

        lm_examples = []

    # ---------------------------------------------------------
    # Process feature batches
    # ---------------------------------------------------------
    with config.output_path.open(
        mode,
        buffering=1,
    ) as fout:
        iterator = tqdm(
            feature_batches,
            desc="Tokenizing + scoring",
            disable=(not config.show_progress),
        )

        for batch_path in iterator:
            for (
                key,
                raw_features,
            ) in iter_feature_batch(batch_path):
                # ---------------------------------------------
                # Resume
                # ---------------------------------------------
                if key in completed:
                    n_skipped += 1
                    continue

                if len(raw_features) == 0:
                    n_empty += 1
                    continue

                # ---------------------------------------------
                # Tokenizer-specific preprocessing
                # ---------------------------------------------
                features = transform_features(
                    raw_features,
                    scaler,
                    pca,
                )

                # ---------------------------------------------
                # Assign token IDs
                # ---------------------------------------------
                if method == "graph_knn":
                    units = tokenize_graph(
                        features=features,
                        index=(tokenizer_index),
                        reference_labels=(reference_labels),
                        k_neighbors=(config.k_neighbors),
                    )

                else:
                    units = tokenize_kmeans(
                        features=features,
                        index=(tokenizer_index),
                    )

                # ---------------------------------------------
                # Validate against LM vocabulary
                # ---------------------------------------------
                if len(units):
                    max_unit = int(units.max())

                    if max_unit >= bos_id:
                        raise RuntimeError(
                            f"{key}: token {max_unit} >= BOS ID {bos_id}"
                        )

                # ---------------------------------------------
                # Construct OPT input/target
                # ---------------------------------------------
                src, tgt = make_lm_sequence(
                    units=units,
                    bos_id=bos_id,
                )

                lm_examples.append(
                    (
                        key,
                        src,
                        tgt,
                    )
                )

                # ---------------------------------------------
                # Flush LM batch
                # ---------------------------------------------
                if len(lm_examples) >= config.lm_batch_size:
                    flush_lm_batch(fout)

            # We have finished consuming this .npz;
            # numpy releases its arrays here.

        # Final partial batch.
        flush_lm_batch(fout)

    print()
    print("=" * 70)
    print("SLM21 SCORING COMPLETE")
    print("=" * 70)

    print(f"Tokenizer:        {config.tokenizer_method}")

    print(f"Scored:           {n_scored:,}")

    print(f"Already scored:   {n_skipped:,}")

    print(f"Empty utterances: {n_empty:,}")

    print(f"Output:           {config.output_path}")


if __name__ == "__main__":
    compute_scores(tyro.cli(ScoreSLM21Config))
