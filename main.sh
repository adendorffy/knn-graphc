#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Experiment configuration
###############################################################################

TRAIN_DATASET="LibriSpeech/train-clean-100"
INFERENCE_DATASET="LibriSpeech/train-*"

CLUSTERING_METHOD="kmeans++" # Options: graph_knn, kmeans++

K=33902
TAU=0.55
GAMMA=0.1100
LEIDEN_ITERS=2

PCA_COMPONENTS=350
KNN_NEIGHBORS=100
HNSW_M=32
HNSW_SEARCH=64

###############################################################################
# Which stages to run
###############################################################################

RUN_EXTRACT=false
RUN_TRAIN_MODELS=false
RUN_GRAPH=false
RUN_KMEANS=false
RUN_INFERENCE=false
RUN_EVALUATE=true
SAVE_TO_CSV=true
RUN_ZIPF=false

###############################################################################
# Paths
###############################################################################

DATA_DIR="../data"

FEATURES_DIR="output/features/zerosyl"
SEGMENTS_DIR="output/segments/zerosyl"
MODELS_DIR="output/models/zerosyl"

CLUSTERING_ROOT="output/clustering/zerosyl/inference-test"
ARTIFACTS_ROOT="output/clustering-artifacts/zerosyl/inference-test"
INFERRED_ROOT="output/inferred_segments/zerosyl"

###############################################################################
# Helper
###############################################################################

banner () {
    echo
    echo "=================================================================="
    echo "$1"
    echo "=================================================================="
}

###############################################################################
# Extract features
###############################################################################

if $RUN_EXTRACT; then
    banner "Extracting features and segments for ${INFERENCE_DATASET}"

    uv run python s01_extract_features_and_segments.py \
        --wav-dir "${DATA_DIR}/audio/${INFERENCE_DATASET}" \
        --features-dir "${FEATURES_DIR}" \
        --segments-dir "${SEGMENTS_DIR}" \
        --wavlm-ckpt-path ../checkpoints/WavLM-Large.pt \
        --batch-size 4 \
        --num-workers 8

fi

###############################################################################
# Train scaler + PCA
###############################################################################

if $RUN_TRAIN_MODELS; then
    banner "Training scaler and PCA"

    uv run python s01b_train_scaler_pca_models.py \
        --config.features-dir "${FEATURES_DIR}/${INFERENCE_DATASET}" \
        --config.output-dir "${MODELS_DIR}/${INFERENCE_DATASET}" \
        --config.number_of_components "${PCA_COMPONENTS}"

fi

###############################################################################
# Train graph clustering
###############################################################################

if $RUN_GRAPH; then
    banner "Training graph clustering"

    uv run python s02_train_graph_clustering.py \
        --features-dir "${FEATURES_DIR}/${TRAIN_DATASET}" \
        --segments-dir "${SEGMENTS_DIR}/${TRAIN_DATASET}" \
        --scaler-path "${MODELS_DIR}/${TRAIN_DATASET}/scaler.joblib" \
        --pca-path "${MODELS_DIR}/${TRAIN_DATASET}/pca.joblib" \
        --edges.min-sim "${TAU}" \
        --edges.num-neighbors "${K}" \
        --cluster.resolution-specified "${GAMMA}" \
        --cluster.leiden-iterations "${LEIDEN_ITERS}" \
        --output-dir "${CLUSTERING_ROOT}/${TRAIN_DATASET}" \
        --show-progress

fi

###############################################################################
# Train KMeans
###############################################################################

if $RUN_KMEANS; then
    banner "Training k-means"

    uv run python s02b_train_faiss_kmeans.py \
        --features-dir "${FEATURES_DIR}/${TRAIN_DATASET}" \
        --segments-dir "${SEGMENTS_DIR}/${TRAIN_DATASET}" \
        --scaler-path "${MODELS_DIR}/${TRAIN_DATASET}/scaler.joblib" \
        --pca-path "${MODELS_DIR}/${TRAIN_DATASET}/pca.joblib" \
        --output-dir "${CLUSTERING_ROOT}/${TRAIN_DATASET}" \
        --num-clusters "${K}" \
        --show-progress

fi

###############################################################################
# Locate graph output automatically
###############################################################################

if [ "${CLUSTERING_METHOD}" = "graph_knn" ]; then
    echo "Using graph clustering method: ${CLUSTERING_METHOD}"
    CLUSTER_OUTPUT_DIR=$(
    find "${CLUSTERING_ROOT}/${TRAIN_DATASET}" \
        -maxdepth 1 \
        -type d \
        -name "${CLUSTERING_METHOD}_${K}_tau_${TAU}_gamma_${GAMMA}_*" \
    | sort \
    | tail -n1
    )

    ARTIFACT_DIR=$(
    find "${ARTIFACTS_ROOT}/${TRAIN_DATASET}" \
        -maxdepth 1 \
        -type d \
        -name "${CLUSTERING_METHOD}_${K}_tau_${TAU}_gamma_${GAMMA}_*" \
    | sort \
    | tail -n1
    )

elif [ "${CLUSTERING_METHOD}" = "kmeans++" ]; then
    echo "Using k-means clustering method: ${CLUSTERING_METHOD}"
    CLUSTER_OUTPUT_DIR=$(
    find "${CLUSTERING_ROOT}/${TRAIN_DATASET}" \
        -maxdepth 1 \
        -type d \
        -name "${CLUSTERING_METHOD}_clusters_${K}_*" \
    | sort \
    | tail -n1
    )
    ARTIFACT_DIR=$(
    find "${ARTIFACTS_ROOT}/${TRAIN_DATASET}" \
        -maxdepth 1 \
        -type d \
        -name "${CLUSTERING_METHOD}_clusters_${K}_*" \
    | sort \
    | tail -n1
    )
else
    echo "Unknown clustering method: ${CLUSTERING_METHOD}"
    exit 1
fi

echo "Located clustering output directory: ${CLUSTER_OUTPUT_DIR}"
echo "Located clustering artifact directory: ${ARTIFACT_DIR}"

###############################################################################
# Inference
###############################################################################

if $RUN_INFERENCE; then
banner "Running inference"

if [ "${CLUSTERING_METHOD}" = "kmeans++" ]; then
    uv run python s04_infer_labels.py \
        --clustering-method "${CLUSTERING_METHOD}" \
        --features-dir "${FEATURES_DIR}/${INFERENCE_DATASET}" \
        --segments-dir "${SEGMENTS_DIR}/${INFERENCE_DATASET}" \
        --models-dir "${MODELS_DIR}/${TRAIN_DATASET}" \
        --reference-dir "${ARTIFACT_DIR}" \
        --clustering-dir "${CLUSTER_OUTPUT_DIR}" 

elif [ "${CLUSTERING_METHOD}" = "graph_knn" ]; then
    uv run python s04_infer_labels.py \
        --clustering-method "${CLUSTERING_METHOD}" \
        --features-dir "${FEATURES_DIR}/${INFERENCE_DATASET}" \
        --segments-dir "${SEGMENTS_DIR}/${INFERENCE_DATASET}" \
        --models-dir "${MODELS_DIR}/${TRAIN_DATASET}" \
        --reference-dir "${ARTIFACT_DIR}" \
        --clustering-dir "${CLUSTER_OUTPUT_DIR}" \
        --k-neighbors "${KNN_NEIGHBORS}" \
        --hnsw-m "${HNSW_M}" \
        --hnsw-ef-search "${HNSW_SEARCH}" 
fi 
fi

###############################################################################
# Inferred output directory
###############################################################################

CLUSTER_INFO=$(basename "${CLUSTER_OUTPUT_DIR}")

if [ "${CLUSTERING_METHOD}" = "graph_knn" ]; then
    INFERRED_DIR="${INFERRED_ROOT}/${INFERENCE_DATASET}/knn_${KNN_NEIGHBORS}_hnswm_${HNSW_M}_search_${HNSW_SEARCH}/zerosyl/inference-test/${TRAIN_DATASET}/${CLSTER_INFO}"
elif [ "${CLUSTERING_METHOD}" = "kmeans++" ]; then
    INFERRED_DIR="${INFERRED_ROOT}/${INFERENCE_DATASET}/kmeans++_${K}/zerosyl/inference-test/${TRAIN_DATASET}/${CLUSTER_INFO}"
else
    echo "Unknown clustering method: ${CLUSTERING_METHOD}"
    exit 1
fi

###############################################################################
# Evaluation
###############################################################################

if $RUN_EVALUATE; then
    banner "Evaluating clustering"

    if [[ "$SAVE_TO_CSV" == true ]]; then
        uv run python s03_evaluate_clustering.py \
            --textgrid-dir "${DATA_DIR}/alignments/${INFERENCE_DATASET}" \
            --segments-dir "${INFERRED_DIR}" \
            --compute-nes \
            --save-to-csv
    else
        uv run python s03_evaluate_clustering.py \
            --textgrid-dir "${DATA_DIR}/alignments/${INFERENCE_DATASET}" \
            --segments-dir "${INFERRED_DIR}" \
            --compute-nes
    fi
fi

###############################################################################
# Zipf plot
###############################################################################

if $RUN_ZIPF; then
    banner "Plotting Zipf"

    uv run python zipf.py \
        --textgrid-dir "${DATA_DIR}/alignments/${INFERENCE_DATASET}" \
        --segments-dir "${INFERRED_DIR}" \
        --output-path figs \
        --save-as-png \
        --matching-method "${CLUSTERING_METHOD}"

fi

banner "Done"