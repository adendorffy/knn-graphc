set -euo pipefail
    
uv run python listen_to_clusters.py \
    --segments-dir output/inferred_segments/zerosyl/LibriSpeech/train-other-500/knn_1_hnswm_32_search_64/zerosyl/inference-test/LibriSpeech/train-clean-100/graph_knn_100_tau_0.65_gamma_0.1100_clusters_78978_runtime_1326.82_peakram_13979.0 \
    --audio-dir ../data/audio/LibriSpeech/train-other-500 \
    --output-dir output/audio-clusters \
    --top-k 5 \
    --examples-per-cluster 100