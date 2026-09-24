from dataclasses import dataclass, field
from pathlib import Path
import json
import zipfile

import numpy as np
import torch
import tyro
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from src.models.poolers.zerosyl import ZeroSylConfig, ZeroSylPooler
from src.models.wavlm.wavlm import load_wavlm_encoder


SAMPLE_RATE = 16_000


@dataclass
class ExtractSLM21FeaturesConfig:
    # ---------------------------------------------------------
    # Input / output
    # ---------------------------------------------------------
    zip_path: Path
    """Path to the SLM21 ZIP archive."""

    output_dir: Path
    """
    Root output directory.

    Features are written as batched .npz files under:

        output_dir/<category>/batch_000000.npz
        output_dir/<category>/batch_000001.npz
        ...

    Each batch stores:
        features : concatenated ZeroSyl features, shape (N_segments, D)
        offsets  : utterance boundaries into features, shape (B + 1,)
        keys     : original WAV filename stems
        paths    : original paths inside the ZIP
        lengths  : number of ZeroSyl segments per utterance
    """

    category: str = "all"
    """
    SLM21 category to extract.

    Examples:
        lexical
        syntactic
        semantic
        all

    Use 'all' to extract every category containing audio under <category>/dev/.
    """

    audio_extensions: tuple[str, ...] = (".wav", ".flac")
    """Audio extensions to process."""

    # ---------------------------------------------------------
    # WavLM + ZeroSyl
    # ---------------------------------------------------------
    wavlm_ckpt_path: Path = Path("WavLM-Large.pt")
    """Path to WavLM checkpoint."""

    device: str = "cuda:0"

    zerosyl: ZeroSylConfig = field(default_factory=ZeroSylConfig)

    # ---------------------------------------------------------
    # Batching
    # ---------------------------------------------------------
    batch_size: int = 4
    """Number of utterances processed together by WavLM."""

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------
    feature_dtype: str = "float32"
    """
    Storage dtype for pooled ZeroSyl features.

    Recommended: float32 if these will later be transformed with
    different scaler/PCA/tokenizer pipelines.

    float16 saves disk space but introduces quantisation.
    """

    # ---------------------------------------------------------
    # Practical controls
    # ---------------------------------------------------------
    resume: bool = True
    """Skip batch files that already exist."""

    show_progress: bool = True


def find_category_dev(
    member_name: str,
) -> tuple[str, int] | None:
    """
    Find paths of the form:

        <category>/dev/...

    Examples:

        lexical/dev/foo.wav
        semantic/dev/synthetic/foo.wav
        SLM21/syntactic/dev/foo.wav

    Returns:
        (category, category_path_index)

    or None if no <category>/dev/ structure is found.
    """

    parts = Path(member_name).parts

    for i in range(len(parts) - 1):
        if parts[i + 1].lower() == "dev":
            return parts[i], i

    return None


def discover_categories(
    zf: zipfile.ZipFile,
    extensions: tuple[str, ...],
) -> list[str]:
    """
    Discover all categories that contain audio under:

        <category>/dev/...
    """

    extensions = tuple(
        ext.lower()
        for ext in extensions
    )

    categories = set()

    for info in zf.infolist():

        if info.is_dir():
            continue

        if not info.filename.lower().endswith(
            extensions
        ):
            continue

        result = find_category_dev(
            info.filename
        )

        if result is None:
            continue

        category, _ = result

        categories.add(category)

    return sorted(categories)


def is_under_category_dev(
    member_name: str,
    category: str,
) -> bool:

    result = find_category_dev(
        member_name
    )

    if result is None:
        return False

    found_category, _ = result

    return (
        found_category.lower()
        == category.lower()
    )


def get_category_audio_members(
    zf: zipfile.ZipFile,
    category: str,
    extensions: tuple[str, ...],
) -> list[zipfile.ZipInfo]:
    """
    Return all audio members below:

        <category>/dev/...

    Members are sorted by uncompressed file size so that similarly
    sized recordings tend to land in the same WavLM batch, reducing
    padding waste.
    """

    extensions = tuple(
        ext.lower()
        for ext in extensions
    )

    members = []

    for info in zf.infolist():

        if info.is_dir():
            continue

        if not info.filename.lower().endswith(
            extensions
        ):
            continue

        if not is_under_category_dev(
            info.filename,
            category,
        ):
            continue

        members.append(info)

    # Similar-size recordings together -> less WavLM padding.
    members.sort(
        key=lambda info: info.file_size
    )

    return members


def make_key(
    member_name: str,
) -> str:
    """
    Preserve the SLM21 WAV identifier.

    Example:

        semantic/dev/synthetic/lCWubBGXgR.wav

    ->

        lCWubBGXgR
    """

    return Path(
        member_name
    ).stem


def extract_feature_batch(
    audio_bytes_batch: list[bytes],
    wavlm,
    pooler,
    device: torch.device,
) -> list[np.ndarray]:
    """
    Decode and process one batch:

        ZIP bytes
        -> waveform
        -> WavLM
        -> ZeroSyl
        -> one feature array per utterance

    Returns:
        list of arrays, each shape:

            (num_segments, feature_dim)

    No scaler, PCA, clustering, or token inference occurs here.
    """

    # ---------------------------------------------------------
    # Decode all utterances in this batch
    # ---------------------------------------------------------
    waveforms = []
    lengths = []

    for audio_bytes in audio_bytes_batch:

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

        waveforms.append(
            waveform
        )

        lengths.append(
            waveform.numel()
        )

    # ---------------------------------------------------------
    # Pad batch to longest waveform
    # ---------------------------------------------------------
    max_len = max(
        lengths
    )

    padded = torch.zeros(
        len(waveforms),
        1,
        max_len,
        dtype=waveforms[0].dtype,
    )

    for i, waveform in enumerate(
        waveforms
    ):
        padded[
            i,
            0,
            : waveform.numel()
        ] = waveform

    padded = padded.to(
        device
    )

    lengths_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
        device=device,
    )

    # ---------------------------------------------------------
    # WavLM + ZeroSyl
    # ---------------------------------------------------------
    with torch.inference_mode():

        (
            _,
            all_hidden_states,
            key_padding_mask,
        ) = wavlm(
            padded,
            lengths=lengths_tensor,
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
            lengths=lengths_tensor,
            center_pad=True,
        )

    # ---------------------------------------------------------
    # Pull only valid pooled features back to CPU
    # ---------------------------------------------------------
    feature_batch = []

    for features, pad_mask in zip(
        segment_features,
        segment_pad_mask,
    ):

        valid_features = (
            features[~pad_mask]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        feature_batch.append(
            valid_features
        )

    # Explicitly release the large GPU tensors.
    del padded
    del lengths_tensor
    del all_hidden_states
    del key_padding_mask
    del segment_features
    del segment_pad_mask
    del waveforms

    return feature_batch


def save_feature_batch(
    output_path: Path,
    infos: list[zipfile.ZipInfo],
    feature_batch: list[np.ndarray],
    feature_dtype: np.dtype,
) -> None:
    """
    Store a complete WavLM/ZeroSyl batch in one .npz file.

    Instead of saving many separate feature matrices:

        utterance1.npy
        utterance2.npy
        utterance3.npy

    concatenate them:

        features = [
            utterance1_features
            utterance2_features
            utterance3_features
        ]

    and save offsets:

        [0, len1, len1+len2, len1+len2+len3]

    so each utterance can later be reconstructed exactly.
    """

    if len(infos) != len(feature_batch):
        raise ValueError(
            "Number of archive members and feature arrays differ."
        )

    lengths = np.asarray(
        [
            len(features)
            for features in feature_batch
        ],
        dtype=np.int64,
    )

    offsets = np.zeros(
        len(lengths) + 1,
        dtype=np.int64,
    )

    offsets[1:] = np.cumsum(
        lengths
    )

    nonempty = [
        features
        for features in feature_batch
        if len(features) > 0
    ]

    if nonempty:
        features_concat = np.concatenate(
            nonempty,
            axis=0,
        ).astype(
            feature_dtype,
            copy=False,
        )

    else:
        # If somehow every utterance in this batch is empty,
        # infer feature dimension if possible.
        feature_dim = (
            feature_batch[0].shape[1]
            if feature_batch
            and feature_batch[0].ndim == 2
            else 0
        )

        features_concat = np.empty(
            (0, feature_dim),
            dtype=feature_dtype,
        )

    keys = np.asarray(
        [
            make_key(info.filename)
            for info in infos
        ]
    )

    paths = np.asarray(
        [
            info.filename
            for info in infos
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        features=features_concat,
        offsets=offsets,
        lengths=lengths,
        keys=keys,
        paths=paths,
    )


def write_category_manifest(
    category_output_dir: Path,
    category: str,
    batch_files: list[Path],
    num_utterances: int,
    num_segments: int,
    feature_dim: int | None,
    feature_dtype: str,
):
    """
    Small metadata file describing the extracted category.
    """

    manifest = {
        "category": category,
        "num_batches": len(batch_files),
        "num_utterances": num_utterances,
        "num_segments": num_segments,
        "feature_dim": feature_dim,
        "feature_dtype": feature_dtype,
        "batch_files": [
            path.name
            for path in batch_files
        ],
    }

    manifest_path = (
        category_output_dir
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )


def extract_category(
    zf: zipfile.ZipFile,
    category: str,
    config: ExtractSLM21FeaturesConfig,
    wavlm,
    pooler,
    device: torch.device,
) -> None:

    members = get_category_audio_members(
        zf=zf,
        category=category,
        extensions=config.audio_extensions,
    )

    print()
    print("=" * 70)
    print(
        f"CATEGORY: {category}"
    )
    print("=" * 70)

    print(
        f"Found {len(members):,} audio files "
        f"under {category}/dev/"
    )

    if not members:
        print(
            f"No audio found for category {category}; skipping."
        )
        return

    category_output_dir = (
        config.output_dir
        / category
    )

    category_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_batches = (
        len(members)
        + config.batch_size
        - 1
    ) // config.batch_size

    n_processed = 0
    n_skipped_batches = 0
    total_segments = 0
    feature_dim = None

    batch_files = []

    iterator = tqdm(
        range(num_batches),
        desc=f"Extracting {category}",
        disable=not config.show_progress,
    )

    for batch_idx in iterator:

        start = (
            batch_idx
            * config.batch_size
        )

        end = min(
            start + config.batch_size,
            len(members),
        )

        batch_infos = members[
            start:end
        ]

        output_path = (
            category_output_dir
            / f"batch_{batch_idx:06d}.npz"
        )

        batch_files.append(
            output_path
        )

        # -----------------------------------------------------
        # Resume support
        # -----------------------------------------------------
        if (
            config.resume
            and output_path.exists()
        ):
            try:
                with np.load(
                    output_path,
                    allow_pickle=False,
                ) as data:

                    batch_lengths = data[
                        "lengths"
                    ]

                    total_segments += int(
                        batch_lengths.sum()
                    )

                    n_processed += len(
                        batch_lengths
                    )

                    if (
                        feature_dim is None
                        and data["features"].ndim == 2
                    ):
                        feature_dim = int(
                            data["features"].shape[1]
                        )

                n_skipped_batches += 1
                continue

            except Exception:
                print(
                    f"\nExisting batch appears incomplete/corrupt; "
                    f"recomputing {output_path}"
                )

        # -----------------------------------------------------
        # Read ONLY this batch from ZIP
        # -----------------------------------------------------
        try:
            audio_bytes_batch = [
                zf.read(info)
                for info in batch_infos
            ]

            # -------------------------------------------------
            # WavLM + ZeroSyl
            # -------------------------------------------------
            feature_batch = extract_feature_batch(
                audio_bytes_batch=audio_bytes_batch,
                wavlm=wavlm,
                pooler=pooler,
                device=device,
            )

            # -------------------------------------------------
            # Establish dimensionality
            # -------------------------------------------------
            for features in feature_batch:
                if features.ndim != 2:
                    raise RuntimeError(
                        f"Expected 2-D ZeroSyl features, "
                        f"got shape {features.shape}"
                    )

                if feature_dim is None:
                    feature_dim = int(
                        features.shape[1]
                    )

                elif (
                    features.shape[1]
                    != feature_dim
                ):
                    raise RuntimeError(
                        "Feature dimensionality changed: "
                        f"{features.shape[1]} != {feature_dim}"
                    )

            # -------------------------------------------------
            # Save ONE feature archive for this WavLM batch
            # -------------------------------------------------
            save_feature_batch(
                output_path=output_path,
                infos=batch_infos,
                feature_batch=feature_batch,
                feature_dtype=np.dtype(
                    config.feature_dtype
                ),
            )

            batch_segments = sum(
                len(features)
                for features in feature_batch
            )

            total_segments += (
                batch_segments
            )

            n_processed += len(
                batch_infos
            )

            del audio_bytes_batch
            del feature_batch

        except Exception:
            print()
            print(
                "Failed on batch containing:"
            )

            for info in batch_infos:
                print(
                    f"  {info.filename}"
                )

            raise

    write_category_manifest(
        category_output_dir=category_output_dir,
        category=category,
        batch_files=batch_files,
        num_utterances=n_processed,
        num_segments=total_segments,
        feature_dim=feature_dim,
        feature_dtype=config.feature_dtype,
    )

    print()
    print(
        f"Category complete:      {category}"
    )
    print(
        f"Utterances:             {n_processed:,}"
    )
    print(
        f"ZeroSyl segments:       {total_segments:,}"
    )
    print(
        f"Feature dimension:      {feature_dim}"
    )
    print(
        f"Batches already there:  {n_skipped_batches:,}"
    )
    print(
        f"Output:                 {category_output_dir}"
    )


def extract_slm21_features(
    config: ExtractSLM21FeaturesConfig,
) -> None:

    device = torch.device(
        config.device
    )

    if config.batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1"
        )

    feature_dtype = np.dtype(
        config.feature_dtype
    )

    if feature_dtype not in (
        np.float32,
        np.float16,
    ):
        raise ValueError(
            "feature_dtype must be float32 or float16"
        )

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load WavLM and ZeroSyl ONCE
    # ---------------------------------------------------------
    print(
        "Loading WavLM..."
    )

    wavlm = load_wavlm_encoder(
        checkpoint_path=config.wavlm_ckpt_path,
        num_layers=(
            config.zerosyl
            .required_wavlm_layers
        ),
        device=device,
    )

    wavlm.eval()

    print(
        "Loading ZeroSyl pooler..."
    )

    pooler = ZeroSylPooler(
        config.zerosyl
    )

    pooler.to(
        device
    ).eval()

    # ---------------------------------------------------------
    # Open ZIP ONCE
    # ---------------------------------------------------------
    with zipfile.ZipFile(
        config.zip_path,
        "r",
    ) as zf:

        available_categories = (
            discover_categories(
                zf,
                config.audio_extensions,
            )
        )

        print(
            "Available SLM21 categories:"
        )

        for category in (
            available_categories
        ):
            print(
                f"  {category}"
            )

        # -----------------------------------------------------
        # Select requested categories
        # -----------------------------------------------------
        if config.category.lower() == "all":

            categories = (
                available_categories
            )

        else:
            matches = [
                category
                for category
                in available_categories
                if category.lower()
                == config.category.lower()
            ]

            if not matches:
                raise ValueError(
                    f"Category {config.category!r} "
                    f"not found.\n"
                    f"Available categories: "
                    f"{available_categories}"
                )

            categories = matches

        # -----------------------------------------------------
        # Extract each category
        # -----------------------------------------------------
        for category in categories:

            extract_category(
                zf=zf,
                category=category,
                config=config,
                wavlm=wavlm,
                pooler=pooler,
                device=device,
            )

    print()
    print("=" * 70)
    print("SLM21 FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    extract_slm21_features(
        tyro.cli(
            ExtractSLM21FeaturesConfig
        )
    )