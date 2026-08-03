from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro


@dataclass
class ReadEvalConfig:
    eval_path: Path
    """Path to evaluation.csv."""

    top_n: int = 10
    """Number of top rows to print."""


def main(config: ReadEvalConfig) -> None:
    df = pd.read_csv(config.eval_path)
    print(f"Loaded evaluation results from {config.eval_path}")

    df = df.sort_values(by="f1_wnes", ascending=False)

    for col in ["wnes", "pacc", "iwnes", "ipacc", "f1_wnes"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)

    for col in ["runtime_seconds", "peak_ram_mb"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    labels = {
        "num_neighbours": "K",
        "min_sim": "τ",
        "resolution": "γ",
    }

    for col, label in labels.items():
        if col in df.columns:
            values = sorted(df[col].dropna().unique())
            if values:
                print(f"{label} = {{{', '.join(str(v) for v in values)}}}")

    cols = [
        "clustering",
        "num_neighbours",
        "min_sim",
        "resolution",
        "num_clusters",
        "wnes",
        "iwnes",
        "f1_wnes",
        "pacc",
        "ipacc",
        "runtime_seconds",
        "peak_ram_mb",
    ]
    cols = [c for c in cols if c in df.columns]

    print(df[cols].head(config.top_n))


if __name__ == "__main__":
    main(tyro.cli(ReadEvalConfig))