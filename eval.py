from pathlib import Path
import pandas as pd

eval_path = Path("output/evaluation/zerosyl/LibriSpeech/train-*/kmeans++_15928/zerosyl/inference-test/LibriSpeech/train-clean-100/evaluation.csv")
df = pd.read_csv(eval_path)
print(f"Loaded evaluation results from {eval_path}")

# order by wnes column
df = df.sort_values(by="f1_wnes", ascending=False)
df["wnes"] = round(df["wnes"] * 100, 2)
df["pacc"] = round(df["pacc"] * 100, 2)
df["iwnes"] = round(df["iwnes"] * 100, 2)
df["ipacc"] = round(df["ipacc"] * 100, 2)
df["f1_wnes"] = round(df["f1_wnes"] * 100, 2)
if "runtime_seconds" in df.columns:
    df["runtime_seconds"] = round(df["runtime_seconds"], 2)
if "peak_ram_mb" in df.columns:
    df["peak_ram_mb"] = round(df["peak_ram_mb"], 2)

labels = {
    "num_neighbours": "K",
    "min_sim": "τ",
    "resolution": "γ",
}

for col, label in labels.items():
    values = sorted(df[col].unique())
    print(f"{label} = {{{', '.join(str(v) for v in values)}}}")

if "runtime_seconds" in df.columns and "peak_ram_mb" in df.columns:
    print(df[["clustering", "num_neighbours", "min_sim", "resolution", "num_clusters", "wnes", "iwnes", "f1_wnes", "pacc", "ipacc", "runtime_seconds", "peak_ram_mb"]].head(10))
else:
    print(df[["clustering", "num_neighbours", "min_sim", "resolution", "num_clusters", "wnes", "iwnes", "pacc", "ipacc"]].head(10))