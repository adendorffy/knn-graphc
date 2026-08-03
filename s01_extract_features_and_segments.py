from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm

from src.models.poolers.zerosyl import ZeroSylConfig, ZeroSylPooler
from src.models.wavlm.wavlm import load_wavlm_encoder

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ExtractFeaturesConfig:
    wav_dir: Path
    """Root directory of input audio files (searched recursively)."""

    features_dir: Path
    """Output directory for per-utterance pooled segment feature .npy files."""

    segments_dir: Path
    """Output directory for per-utterance segment boundary .npy files."""

    wavlm_ckpt_path: Path
    """Path to the WavLM checkpoint (.pt)."""

    extension: str = ".flac"
    """File extension of input audio files (e.g. .flac, .wav)."""

    batch_size: int = 4
    """Batch size for WavLM inference."""

    num_workers: int = 0
    """DataLoader worker processes (0 = load audio in the main process)."""

    device: str = "cuda:0"
    """PyTorch device for model inference (e.g. cuda:0)."""

    zerosyl: ZeroSylConfig = field(default_factory=ZeroSylConfig)
    """ZeroSyl syllable-boundary pooler settings."""


class WavDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Path, torch.Tensor, int]:
        path = self.paths[index]
        decoder = AudioDecoder(path, sample_rate=SAMPLE_RATE, num_channels=1)
        waveform = decoder.get_all_samples().data.squeeze(0)
        length = waveform.size(-1)
        return path, waveform, length


def collate_fn(
    batch: list[tuple[Path, torch.Tensor, int]],
) -> tuple[list[Path], torch.Tensor, torch.Tensor]:
    paths, waveforms, lengths = zip(*batch)
    max_len = max(lengths)
    padded = torch.zeros(len(batch), 1, max_len, dtype=waveforms[0].dtype)
    for i, (waveform, length) in enumerate(zip(waveforms, lengths)):
        padded[i, 0, :length] = waveform
    return list(paths), padded, torch.tensor(lengths, dtype=torch.long)


def relative_output_path(audio_path: Path, audio_root: Path, output_root: Path) -> Path:
    rel = audio_path.relative_to(audio_root)
    return output_root / rel.with_suffix(".npy")


def extract_features_and_segments(config: ExtractFeaturesConfig):

    device = torch.device(config.device)

    audio_root = config.wav_dir.resolve()
    audio_paths = sorted(audio_root.rglob(f"*{config.extension}"))
    assert audio_paths, (
        f"No audio files with extension {config.extension!r} found under {audio_root}"
    )

    dataset = WavDataset(audio_paths)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    wavlm = load_wavlm_encoder(
        checkpoint_path=config.wavlm_ckpt_path,
        num_layers=config.zerosyl.required_wavlm_layers,
        device=device,
    )

    pooler = ZeroSylPooler(config.zerosyl)
    pooler.to(device).eval()

    for batch in tqdm(loader):
        paths, waveforms, lengths = batch

        waveforms = waveforms.to(device)
        lengths = lengths.to(device)

        with torch.inference_mode():
            _, all_hidden_states, key_padding_mask = wavlm(
                waveforms, lengths=lengths, normalize=True, center_pad=True
            )
            segment_features, segment_durations, segment_pad_mask = pooler.forward(
                all_hidden_states, key_padding_mask, lengths=lengths, center_pad=True
            )

        for path, features, durations, pad_mask in zip(
            paths, segment_features, segment_durations, segment_pad_mask
        ):  
            features_out_path = relative_output_path(
                path, audio_root, config.features_dir
            )
            if features_out_path.exists():
                continue
                
            features_out_path.parent.mkdir(parents=True, exist_ok=True)

            valid_mask = ~pad_mask
            valid_features = features[valid_mask]
            valid_durations = durations[valid_mask]

            valid_ends = valid_durations.cumsum(0)
            valid_starts = valid_ends - valid_durations
            dummy_units = torch.zeros_like(valid_starts)
            valid_segments = torch.stack([valid_starts, valid_ends, dummy_units], dim=1)
            
            np.save(
                features_out_path,
                valid_features.cpu().numpy().astype(np.float32, copy=False),
            )

            segments_out_path = relative_output_path(
                path, audio_root, config.segments_dir
            )
            segments_out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(segments_out_path, valid_segments.cpu().numpy())


if __name__ == "__main__":
    extract_features_and_segments(tyro.cli(ExtractFeaturesConfig))
