#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Experiment configuration
###############################################################################

TRAIN_DATASET="LibriSpeech/train-clean-100"
INFERENCE_DATASET="LibriSpeech/train-*"

CLUSTERING_METHOD="graph_knn"

K=100
TAU=0.65
GAMMA=0.1100
LEIDEN_ITERS=2

PCA_COMPONENTS=350
KNN_NEIGHBORS=100
HNSW_M=32
HNSW_SEARCH=128

###############################################################################
# Which stages to run
###############################################################################

RUN_EXTRACT=false
RUN_TRAIN_MODELS=false
RUN_GRAPH=false
RUN_KMEANS=false
RUN_INFERENCE=false
RUN_EVALUATE=false
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
# Inference settings to evaluate
###############################################################################

EF_SEARCH_VALUES=(256 512)
RUN_KMEANS_TRAIN=false
RUN_KMEANS_INFERENCE=true
RUN_KMEANS_EVALUATE=true

###############################################################################
 # Locate graph output automatically
###############################################################################

GRAPH_DIR=$(
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
echo GRAPH_DIR="${GRAPH_DIR}"
echo ARTIFACT_DIR="${ARTIFACT_DIR}"

if [[ -z "${GRAPH_DIR}" ]]; then
    echo "ERROR: Could not locate graph directory."
    exit 1
fi

if [[ -z "${ARTIFACT_DIR}" ]]; then
    echo "ERROR: Could not locate artifact directory."
    exit 1
fi

echo
echo "Graph directory:"
echo "  ${GRAPH_DIR}"

echo
echo "Artifact directory:"
echo "  ${ARTIFACT_DIR}"

GRAPH_NAME=$(basename "${GRAPH_DIR}")

###############################################################################
# Run inference and evaluation for each ef_search value
###############################################################################

for EF_SEARCH in "${EF_SEARCH_VALUES[@]}"; do

    banner "Experiment: k=${KNN_NEIGHBORS}, ef_search=${EF_SEARCH}"

    ###########################################################################
    # Inference
    ###########################################################################

    if $RUN_INFERENCE; then
        banner "Running inference with ef_search=${EF_SEARCH}"

        uv run python s04_infer_labels.py \
            --clustering-method "${CLUSTERING_METHOD}" \
            --features-dir "${FEATURES_DIR}/${INFERENCE_DATASET}" \
            --segments-dir "${SEGMENTS_DIR}/${INFERENCE_DATASET}" \
            --models-dir "${MODELS_DIR}/${TRAIN_DATASET}" \
            --reference-dir "${ARTIFACT_DIR}" \
            --clustering-dir "${GRAPH_DIR}" \
            --k-neighbors "${KNN_NEIGHBORS}" \
            --hnsw-m "${HNSW_M}" \
            --hnsw-ef-search "${EF_SEARCH}"
    fi

    ###########################################################################
    # Inferred output directory
    ###########################################################################

    INFERRED_DIR="output/inferred_segments/zerosyl/${INFERENCE_DATASET}/knn_${KNN_NEIGHBORS}_hnswm_${HNSW_M}_search_${EF_SEARCH}/zerosyl/inference-test/${TRAIN_DATASET}/${GRAPH_NAME}"

    ###########################################################################
    # Evaluation
    ###########################################################################

    if $RUN_EVALUATE; then
        banner "Evaluating clustering with ef_search=${EF_SEARCH}"

        uv run python s03_evaluate_clustering.py \
            --textgrid-dir "${DATA_DIR}/alignments/${INFERENCE_DATASET}" \
            --segments-dir "${INFERRED_DIR}" \
            --compute-nes \
            --save-to-csv
    fi

    ###########################################################################
    # Zipf plot
    ###########################################################################

    if $RUN_ZIPF; then
        banner "Plotting Zipf distribution with ef_search=${EF_SEARCH}"

        uv run python zipf.py \
            --textgrid-dir "${DATA_DIR}/alignments/${INFERENCE_DATASET}" \
            --segments-dir "${INFERRED_DIR}" \
            --output-path "figs/ef_search_${EF_SEARCH}" \
            --save-as-png \
            --matching-method "${CLUSTERING_METHOD}"
    fi

done

banner "Done"

###############################################################################
# Train, infer and evaluate K-means models
###############################################################################

KMEANS_K_VALUES=(78705 33902 15928)

for TRAIN_K in "${KMEANS_K_VALUES[@]}"; do

    banner "K-means experiment: K=${TRAIN_K}"

    ###########################################################################
    # Train K-means model
    ###########################################################################

    if $RUN_KMEANS_TRAIN; then
        KMEANS_DIR=$(
            find "${CLUSTERING_ROOT}/${TRAIN_DATASET}" \
                -maxdepth 1 \
                -type d \
                -name "*kmeans*${TRAIN_K}*" \
            | sort \
            | tail -n1
        )   
        if [[ -n "${KMEANS_DIR}" ]]; then
            echo "K-means model already exists for K=${TRAIN_K}:"
            echo "  ${KMEANS_DIR}"
            continue
        fi
        banner "Training K-means with K=${TRAIN_K}"

        uv run python s02b_train_faiss_kmeans.py \
            --features-dir "${FEATURES_DIR}/${TRAIN_DATASET}" \
            --segments-dir "${SEGMENTS_DIR}/${TRAIN_DATASET}" \
            --scaler-path "${MODELS_DIR}/${TRAIN_DATASET}/scaler.joblib" \
            --pca-path "${MODELS_DIR}/${TRAIN_DATASET}/pca.joblib" \
            --num-clusters "${TRAIN_K}" \
            --output-dir "${CLUSTERING_ROOT}/${TRAIN_DATASET}" \
            --show-progress
    fi

    ###########################################################################
    # Locate trained K-means model
    ###########################################################################

    KMEANS_DIR=$(
        find "${CLUSTERING_ROOT}/${TRAIN_DATASET}" \
            -maxdepth 1 \
            -type d \
            -name "*kmeans*${TRAIN_K}*" \
        | sort \
        | tail -n1
    )
    KMEANS_REF_DIR=$(
        find "${ARTIFACTS_ROOT}/${TRAIN_DATASET}" \
            -maxdepth 1 \
            -type d \
            -name "*kmeans*${TRAIN_K}*" \
        | sort \
        | tail -n1
    )

    if [[ -z "${KMEANS_DIR}" ]]; then
        echo "ERROR: Could not locate K-means model for K=${TRAIN_K}"
        exit 1
    fi

    echo
    echo "K-means directory:"
    echo "  ${KMEANS_DIR}"

    KMEANS_NAME=$(basename "${KMEANS_DIR}")

    ###########################################################################
    # Run K-means inference
    ###########################################################################

    if $RUN_KMEANS_INFERENCE; then
        banner "Running K-means inference with K=${TRAIN_K}"

        uv run python s04_infer_labels.py \
            --clustering-method "kmeans" \
            --features-dir "${FEATURES_DIR}/${INFERENCE_DATASET}" \
            --segments-dir "${SEGMENTS_DIR}/${INFERENCE_DATASET}" \
            --models-dir "${MODELS_DIR}/${TRAIN_DATASET}" \
            --clustering-dir "${KMEANS_DIR}" \
            --reference-dir "${KMEANS_REF_DIR}"
    fi

    ###########################################################################
    # Locate inferred output
    ###########################################################################

    KMEANS_INFERRED_DIR="output/inferred_segments/zerosyl/${INFERENCE_DATASET}/kmeans/zerosyl/inference-test/${TRAIN_DATASET}/${KMEANS_NAME}"

    ###########################################################################
    # Evaluate
    ###########################################################################

    if $RUN_KMEANS_EVALUATE; then
        banner "Evaluating K-means with K=${TRAIN_K}"

        if [[ ! -d "${KMEANS_INFERRED_DIR}" ]]; then
            echo "ERROR: Inferred output not found:"
            echo "  ${KMEANS_INFERRED_DIR}"
            exit 1
        fi

        uv run python s03_evaluate_clustering.py \
            --textgrid-dir "${DATA_DIR}/alignments/${INFERENCE_DATASET}" \
            --segments-dir "${KMEANS_INFERRED_DIR}" \
            --compute-nes
    fi

done