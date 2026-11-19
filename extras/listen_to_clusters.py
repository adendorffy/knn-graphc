from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import tyro
from tqdm import tqdm

WAVLM_FRAME_RATE = 50.0   

def set_output_path(segments_dir: Path, output_root: str) -> Path:
    rel = segments_dir.relative_to(Path("output/"))
    print(f"Output path not specified. Using default: {output_root}/{rel}") 
    return Path(output_root) / rel


@dataclass
class ListenConfig:
    segments_dir: Path
    audio_dir: Path
    output_dir: Path | None = None

    top_k: int = 5
    examples_per_cluster: int = 100


def listen(config: ListenConfig):
    config.output_dir = set_output_path(
        config.segments_dir,
        config.output_dir if config.output_dir is not None else "output/audio-clusters",
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_examples = defaultdict(list)
    cluster_sizes = Counter()

    segment_files = sorted(config.segments_dir.rglob("*.npy"))

    for seg_path in tqdm(segment_files):

        rel = seg_path.relative_to(config.segments_dir)

        audio_path = (
            config.audio_dir /
            rel.with_suffix(".flac")
        )

        segments = np.load(seg_path)

        for start, end, cluster in segments.astype(int):
            cluster_sizes[int(cluster)] += 1
            cluster_examples[int(cluster)].append(
                (audio_path, start, end)
            )

    largest = cluster_sizes.most_common(config.top_k)

    print("\nLargest clusters:")
    for cluster, n in largest:
        print(f"Cluster {cluster}: {n} samples")

    silence_duration = 0.2 

    for cluster, count in largest:

        pieces = []
        for audio_path, start, end in cluster_examples[cluster][:config.examples_per_cluster]:

            audio, sr = sf.read(audio_path)

            start_sample = int(start / WAVLM_FRAME_RATE * sr)
            end_sample = int(end / WAVLM_FRAME_RATE * sr)

            snippet = audio[start_sample:end_sample]
            snippet = audio[start_sample:end_sample]
            
            rms = np.sqrt(np.mean(snippet**2))
            target_rms = 0.1

            if rms > 1e-8:
                snippet *= target_rms / rms

            pieces.append(snippet)
            silence = np.zeros(
                int(silence_duration * sr),
                dtype=audio.dtype,
            )
            pieces.append(silence)

        if not pieces:
            continue

        cluster_audio = np.concatenate(pieces)

        out_path = (
            config.output_dir /
            f"cluster_{cluster}_{count}.flac"
        )

        sf.write(out_path, cluster_audio, sr)
        
    print(f"\nDone! Audio clusters written to {config.output_dir}")

if __name__ == "__main__":
    listen(tyro.cli(ListenConfig))