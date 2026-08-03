from pathlib import Path
import numpy as np

segments_dir = Path("output/inferred_segments/zerosyl/LibriSpeech/train-*/knn_3_hnswm_32_search_64/zerosyl/inference-test/LibriSpeech/dev-clean/graph_knn_100_tau_0.65_gamma_0.1100_clusters_14115_runtime_13.12_peakram_1195.1")

num_segments = 0
for segments_path in segments_dir.rglob("*.npy"):
    segments = np.load(segments_path)
    num_segments += len(segments)

print(f"Number of files in {segments_dir}: {len(list(segments_dir.rglob('*.npy')))}")
print(f"Number of segments in {segments_dir}: {num_segments}")
