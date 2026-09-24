from dataclasses import dataclass, field
from pathlib import Path
import gc
import json
import os
import tarfile

import faiss
import joblib
import numpy as np
import torch
import tyro
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from src.models.poolers.zerosyl import ZeroSylConfig, ZeroSylPooler
from src.models.wavlm.wavlm import load_wavlm_encoder


SAMPLE_RATE = 16_000


@dataclass
class StreamLibriLightConfig:
    # ---------------------------------------------------------
    # Input / output
    # ---------------------------------------------------------
    tar_path: Path
    """Path to the LibriLight TAR archive."""

    output_prefix: Path
    """
    Output prefix. Creates:
      <output_prefix>.bin
      <output_prefix>.json
    """

    extension: str = ".flac"
    """Audio extension inside the TAR archive."""

    member_contains: str | None = None
    """
    Optional substring used to restrict archive members.
    For example: 'small/', 'medium/', etc.
    None processes all matching audio files.
    """

    # ---------------------------------------------------------
    # WavLM / ZeroSyl
    # ---------------------------------------------------------
    wavlm_ckpt_path: Path = Path("WavLM-Large.pt")

    device: str = "cuda:0"

    zerosyl: ZeroSylConfig = field(default_factory=ZeroSylConfig)

    # ---------------------------------------------------------
    # Training scaler / PCA
    # ---------------------------------------------------------
    models_dir: Path | None = None
    """
    Directory containing scaler.joblib and pca.joblib.
    These must be the SAME models used for the reference clustering.
    """

    # ---------------------------------------------------------
    # Inference configuration
    # ---------------------------------------------------------
    clustering_method: str = "graph_knn"

    reference_dir: Path = Path("output/clustering-artifacts")

    num_clusters: int | None = None

    hnsw_m: int = 32

    hnsw_ef_search: int = 64

    k_neighbors: int = 1

    # ---------------------------------------------------------
    # LM stream configuration
    # ---------------------------------------------------------
    boundary_token_id: int | None = None
    """
    Token written between utterances.

    For your 9693-cluster vocabulary, use 9693 if this is also
    the BOS token in your OPT model.

    Set to None if you do not want an explicit utterance boundary.
    """

    output_dtype: str = "uint16"

    # ---------------------------------------------------------
    # Practical settings
    # ---------------------------------------------------------
    show_progress: bool = True

    chunk_seconds: float = 30.0
    """Maximum audio duration sent through WavLM at once."""

    flush_every: int = 1000
    """Flush output file after this many utterances."""


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

        return np.where(
            (a == b) | (a == c),
            a,
            np.where(b == c, b, a),
        )

    return np.array(
        [_row_mode(row) for row in neighbor_labels],
        dtype=np.int64,
    )


def build_inference_index(config: StreamLibriLightConfig):
    """
    Build exactly the same type of FAISS inference index as
    s04_infer_labels.py.
    """

    faiss.omp_set_num_threads(os.cpu_count() or 1)

    reference_labels = None

    if config.clustering_method == "kmeans++":
        print("Loading K-means centroids...")

        centroids = np.load(
            config.reference_dir / "centroids.npy"
        ).astype(np.float32, copy=False)

        centroids = np.ascontiguousarray(
            centroids,
            dtype=np.float32,
        )

        num_clusters = len(centroids)

        if config.num_clusters is not None:
            assert num_clusters == config.num_clusters, (
                num_clusters,
                config.num_clusters,
            )

        index = faiss.IndexFlatL2(
            centroids.shape[1]
        )

        index.add(centroids)

        print(
            f"Loaded {num_clusters:,} K-means centroids."
        )

    elif config.clustering_method == "graph_knn":
        print("Loading graph reference labels...")

        reference_labels = np.load(
            config.reference_dir / "reference_labels.npy"
        ).astype(np.int64, copy=False)

        n_unique = len(np.unique(reference_labels))

        if config.num_clusters is not None:
            assert n_unique == config.num_clusters, (
                n_unique,
                config.num_clusters,
            )

        print(
            f"Reference labels: "
            f"{len(reference_labels):,} segments, "
            f"{n_unique:,} clusters"
        )

        print("Loading graph reference features...")

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

        print(
            f"Building HNSW index: "
            f"N={len(reference_features):,}, "
            f"dim={dim}, "
            f"M={config.hnsw_m}, "
            f"efSearch={config.hnsw_ef_search}"
        )

        index = faiss.IndexHNSWFlat(
            dim,
            config.hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efSearch = config.hnsw_ef_search

        index.add(reference_features)

        # FAISS has copied the vectors into the index.
        del reference_features
        gc.collect()

    else:
        raise ValueError(
            f"Unknown clustering method: "
            f"{config.clustering_method}"
        )

    return index, reference_labels


def infer_units(
    features: np.ndarray,
    config: StreamLibriLightConfig,
    index,
    reference_labels: np.ndarray | None,
) -> np.ndarray:
    """
    Reproduce the actual inference portion of s04_infer_labels.py.
    """

    features = np.ascontiguousarray(
        features,
        dtype=np.float32,
    )

    if config.clustering_method == "kmeans++":
        _, inferred = index.search(
            features,
            1,
        )

        return inferred[:, 0].astype(
            np.int64,
            copy=False,
        )

    if config.clustering_method == "graph_knn":
        faiss.normalize_L2(features)

        _, neighbor_ids = index.search(
            features,
            config.k_neighbors,
        )

        if np.any(neighbor_ids < 0):
            raise RuntimeError(
                "FAISS returned invalid neighbour IDs. "
                "Try increasing hnsw_ef_search or reducing "
                "k_neighbors."
            )

        neighbor_labels = reference_labels[
            neighbor_ids
        ]

        return majority_vote(
            neighbor_labels
        )

    raise ValueError(
        f"Unknown clustering method: "
        f"{config.clustering_method}"
    )


def list_audio_members(
    tf: tarfile.TarFile,
    extension: str,
    member_contains: str | None,
):
    extension = extension.lower()

    members = []

    for info in tf.getmembers():
        if not info.isfile():
            continue

        if not info.name.lower().endswith(extension):
            continue

        if (
            member_contains is not None
            and member_contains not in info.name
        ):
            continue

        members.append(info)

    members.sort(
        key=lambda x: x.name
    )

    return members

def process_audio_member(
    tf: tarfile.TarFile,
    info: tarfile.TarInfo,
    wavlm,
    pooler,
    scaler,
    pca,
    index,
    reference_labels,
    config: StreamLibriLightConfig,
    device: torch.device,
):
    """
    Read one compressed audio member and process it in manageable
    waveform chunks so that long LibriLight recordings do not OOM.
    """

    # ---------------------------------------------------------
    # 1. Read this archive member
    # ---------------------------------------------------------
    member_file = tf.extractfile(info)

    if member_file is None:
        raise RuntimeError(
            f"Could not extract {info.name}"
        )

    audio_bytes = member_file.read()

    # ---------------------------------------------------------
    # 2. Decode
    # ---------------------------------------------------------
    decoder = AudioDecoder(
        audio_bytes,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )

    waveform = (
        decoder
        .get_all_samples()
        .data
        .squeeze(0)
    )

    del audio_bytes

    # ---------------------------------------------------------
    # 3. Process manageable chunks
    # ---------------------------------------------------------
    chunk_samples = int(
        config.chunk_seconds * SAMPLE_RATE
    )
    min_samples = SAMPLE_RATE

    all_units = []
    start = 0

    while start < waveform.numel():
        end = min(
            start + chunk_samples,
            waveform.numel(),
        )

        if 0 < waveform.numel() - end < min_samples:
            end = waveform.numel()

        waveform_chunk = waveform[start:end]

        if waveform_chunk.numel() < min_samples:
            waveform_chunk = torch.nn.functional.pad(
                waveform_chunk,
                (0, min_samples - waveform_chunk.numel()),
            )

        # Ignore pathological empty chunks.
        if waveform_chunk.numel() == 0:
            continue

        length = waveform_chunk.size(-1)

        waveforms = (
            waveform_chunk
            .view(1, 1, -1)
            .to(device)
        )

        lengths = torch.tensor(
            [length],
            dtype=torch.long,
            device=device,
        )

        # -----------------------------------------------------
        # 4. WavLM + ZeroSyl
        # -----------------------------------------------------
        with torch.inference_mode():
            (
                _,
                all_hidden_states,
                key_padding_mask,
            ) = wavlm(
                waveforms,
                lengths=lengths,
                normalize=True,
                center_pad=True,
            )

            (
                segment_features,
                _segment_durations,
                segment_pad_mask,
            ) = pooler.forward(
                all_hidden_states,
                key_padding_mask,
                lengths=lengths,
                center_pad=True,
            )

        valid_mask = ~segment_pad_mask[0]

        valid_features = (
            segment_features[0][valid_mask]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        # -----------------------------------------------------
        # Release GPU tensors immediately
        # -----------------------------------------------------
        del waveform_chunk
        del waveforms
        del lengths
        del all_hidden_states
        del key_padding_mask
        del segment_features
        del segment_pad_mask

        torch.cuda.empty_cache()

        # -----------------------------------------------------
        # 5. scaler + PCA
        # -----------------------------------------------------
        if scaler is not None:
            valid_features = scaler.transform(
                valid_features
            )

        if pca is not None:
            valid_features = pca.transform(
                valid_features
            )

        valid_features = valid_features.astype(
            np.float32,
            copy=False,
        )

        # -----------------------------------------------------
        # 6. Infer discrete units
        # -----------------------------------------------------
        units = infer_units(
            valid_features,
            config,
            index,
            reference_labels,
        )

        all_units.append(units)

        start = end 

        del valid_features

    del waveform

    if not all_units:
        return np.empty(
            0,
            dtype=np.int64,
        )

    return np.concatenate(all_units)


def build_stream(config: StreamLibriLightConfig):
    device = torch.device(config.device)

    output_bin = Path(
        f"{config.output_prefix}.bin"
    )

    output_json = Path(
        f"{config.output_prefix}.json"
    )

    output_bin.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Validate output dtype
    # ---------------------------------------------------------
    dtype = np.dtype(config.output_dtype)

    if dtype != np.uint16:
        raise ValueError(
            "This script currently expects output_dtype=uint16."
        )

    max_uint16 = np.iinfo(np.uint16).max

    if (
        config.boundary_token_id is not None
        and config.boundary_token_id > max_uint16
    ):
        raise ValueError(
            f"boundary_token_id={config.boundary_token_id} "
            "does not fit in uint16."
        )

    # ---------------------------------------------------------
    # Load training scaler/PCA
    # ---------------------------------------------------------
    if config.models_dir is not None:
        scaler_path = (
            config.models_dir / "scaler.joblib"
        )
        pca_path = (
            config.models_dir / "pca.joblib"
        )

        print(
            f"Loading scaler from {scaler_path}"
        )
        scaler = joblib.load(scaler_path)

        print(
            f"Loading PCA from {pca_path}"
        )
        pca = joblib.load(pca_path)

    else:
        scaler = None
        pca = None

    # ---------------------------------------------------------
    # Build the same inference index used by s04
    # ---------------------------------------------------------
    index, reference_labels = (
        build_inference_index(config)
    )

    # ---------------------------------------------------------
    # Load WavLM / ZeroSyl
    # ---------------------------------------------------------
    print("Loading WavLM...")

    wavlm = load_wavlm_encoder(
        checkpoint_path=config.wavlm_ckpt_path,
        num_layers=config.zerosyl.required_wavlm_layers,
        device=device,
    )

    wavlm.eval()

    print("Loading ZeroSyl...")

    pooler = ZeroSylPooler(
        config.zerosyl
    )

    pooler.to(device).eval()

    # ---------------------------------------------------------
    # Open archive
    # ---------------------------------------------------------
    with tarfile.open(
        config.tar_path,
        "r:*",
    ) as tf:

        members = list_audio_members(
            tf,
            extension=config.extension,
            member_contains=config.member_contains,
        )

        if not members:
            raise RuntimeError(
                f"No {config.extension} files found "
                f"in {config.tar_path}"
            )

        print(
            f"Found {len(members):,} audio files."
        )

        # -----------------------------------------------------
        # Stream directly into the binary file.
        #
        # No need to know total length before writing.
        # np.memmap can read this binary file later.
        # -----------------------------------------------------
        total_tokens = 0
        utterances_written = 0
        skipped_empty = 0

        min_label = None
        max_label = None

        unique_labels = set()

        with output_bin.open("wb") as fout:

            iterator = tqdm(
                members,
                desc="LibriLight → units",
                disable=not config.show_progress,
            )

            for i, info in enumerate(iterator):

                units = process_audio_member(
                    tf=tf,
                    info=info,
                    wavlm=wavlm,
                    pooler=pooler,
                    scaler=scaler,
                    pca=pca,
                    index=index,
                    reference_labels=reference_labels,
                    config=config,
                    device=device,
                )

                if len(units) == 0:
                    skipped_empty += 1
                    continue

                if np.any(units < 0):
                    raise RuntimeError(
                        f"Negative unit ID in "
                        f"{info.name}"
                    )

                if units.max() > max_uint16:
                    raise RuntimeError(
                        f"Unit ID {units.max()} does "
                        "not fit in uint16."
                    )

                # ---------------------------------------------
                # Track vocabulary statistics
                # ---------------------------------------------
                this_min = int(units.min())
                this_max = int(units.max())

                if min_label is None:
                    min_label = this_min
                    max_label = this_max
                else:
                    min_label = min(
                        min_label,
                        this_min,
                    )
                    max_label = max(
                        max_label,
                        this_max,
                    )

                unique_labels.update(
                    np.unique(units).tolist()
                )

                # ---------------------------------------------
                # Write utterance boundary
                # ---------------------------------------------
                if config.boundary_token_id is not None:
                    np.asarray(
                        [config.boundary_token_id],
                        dtype=np.uint16,
                    ).tofile(fout)

                    total_tokens += 1

                # ---------------------------------------------
                # Write inferred acoustic units
                # ---------------------------------------------
                units.astype(
                    np.uint16,
                    copy=False,
                ).tofile(fout)

                total_tokens += len(units)
                utterances_written += 1

                if (
                    config.flush_every > 0
                    and utterances_written
                    % config.flush_every
                    == 0
                ):
                    fout.flush()

                del units

        # -----------------------------------------------------
        # Add one final boundary marker, matching the spirit of
        # your old memmap builder.
        # -----------------------------------------------------
        if (
            config.boundary_token_id is not None
            and utterances_written > 0
        ):
            with output_bin.open("ab") as fout:
                np.asarray(
                    [config.boundary_token_id],
                    dtype=np.uint16,
                ).tofile(fout)

            total_tokens += 1

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    unique_labels_arr = np.asarray(
        sorted(unique_labels),
        dtype=np.int64,
    )

    if min_label is not None:
        expected = np.arange(
            min_label,
            max_label + 1,
        )

        missing_labels = np.setdiff1d(
            expected,
            unique_labels_arr,
        )
    else:
        missing_labels = np.array(
            [],
            dtype=np.int64,
        )

    metadata = {
        "tar_path": str(config.tar_path),
        "dtype": str(dtype),
        "total_tokens": int(total_tokens),
        "utterances_written": int(
            utterances_written
        ),
        "skipped_empty": int(
            skipped_empty
        ),
        "minimum_cluster_label": (
            None
            if min_label is None
            else int(min_label)
        ),
        "maximum_cluster_label": (
            None
            if max_label is None
            else int(max_label)
        ),
        "unique_cluster_labels": int(
            len(unique_labels_arr)
        ),
        "missing_cluster_labels": (
            missing_labels.tolist()
        ),
        "boundary_token_id": (
            config.boundary_token_id
        ),
        "clustering_method": (
            config.clustering_method
        ),
        "k_neighbors": (
            config.k_neighbors
        ),
        "hnsw_m": (
            config.hnsw_m
        ),
        "hnsw_ef_search": (
            config.hnsw_ef_search
        ),
        "reference_dir": str(
            config.reference_dir
        ),
        "models_dir": (
            None
            if config.models_dir is None
            else str(config.models_dir)
        ),
    }

    with output_json.open(
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        f"Utterances written: "
        f"{utterances_written:,}"
    )
    print(
        f"Empty utterances:    "
        f"{skipped_empty:,}"
    )
    print(
        f"Total tokens:        "
        f"{total_tokens:,}"
    )
    print(
        f"Min cluster ID:      "
        f"{min_label}"
    )
    print(
        f"Max cluster ID:      "
        f"{max_label}"
    )
    print(
        f"Unique clusters:     "
        f"{len(unique_labels_arr):,}"
    )

    if len(missing_labels):
        print(
            f"Unused cluster IDs:  "
            f"{missing_labels.tolist()}"
        )

    print()
    print(
        f"Data:     {output_bin}"
    )
    print(
        f"Metadata: {output_json}"
    )


if __name__ == "__main__":
    build_stream(
        tyro.cli(StreamLibriLightConfig)
    )