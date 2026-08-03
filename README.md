# Graph-Based Clustering for Unsupervised Speech Unit Discovery

This repository contains a pipeline for extracting segment-level speech representations, clustering them into induced unit inventories, assigning labels to new data, and evaluating the resulting clusters against reference TextGrid annotations.

The main focus is comparing scalable graph-based clustering methods with K-means for unsupervised lexicon or speech-unit discovery. The pipeline supports:

* ZeroSyl-based segmentation and pooled WavLM feature extraction
* Standardisation and PCA projection
* kNN graph clustering with Leiden/CPM
* Full cosine graph clustering
* FAISS K-means clustering
* Label inference from trained clusterings to new datasets
* Clustering evaluation using wNES, iwNES and F1-wNES
* Zipf-style rank-frequency plots of induced clusters and discovered phone sequences

## Repository structure

The main scripts are organised as sequential pipeline stages.

| Script                                 | Purpose                                                                     |
| -------------------------------------- | --------------------------------------------------------------------------- |
| `s01_extract_features_and_segments.py` | Extract WavLM features and ZeroSyl segment boundaries from audio.           |
| `s01b_train_scaler_pca_models.py`      | Train a `StandardScaler` and PCA model on pooled segment features.          |
| `s02_train_graph_clustering.py`        | Train graph clustering using either a kNN graph or full cosine graph.       |
| `s02b_train_faiss_kmeans.py`           | Train FAISS K-means on the same feature representation.                     |
| `s03_evaluate_clustering.py`           | Evaluate labelled segments against TextGrid references.                     |
| `s04_infer_labels.py`                  | Infer labels for new segments using a trained graph or K-means model.       |
| `s04b_read_eval_csv.py`                | Read and summarise saved evaluation CSV files.                              |
| `s05_zipf.py`                          | Plot rank-frequency distributions for induced clusters and phone sequences. |
| `main.sh`                              | Example orchestration script for running selected stages.                   |

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. The repository includes a `pyproject.toml` and `uv.lock`.

Clone the repository and install dependencies with:

```bash
uv sync
```

Run scripts with:

```bash
uv run python <script_name>.py --help
```

For example:

```bash
uv run python s01_extract_features_and_segments.py --help
```

## Expected data layout

The scripts assume audio and alignment data are available outside the repository, for example:

```text
../data/
├── audio/
│   └── LibriSpeech/
│       ├── dev-clean/
│       ├── test-clean/
│       └── train-clean-100/
└── alignments/
    └── LibriSpeech/
        ├── dev-clean/
        ├── test-clean/
        └── train-clean-100/
```

A WavLM checkpoint is also expected, for example:

```text
../checkpoints/WavLM-Large.pt
```

The exact paths can be changed through the command-line arguments or in `run.sh`.

## Pipeline overview

The pipeline has five main stages.

### 1. Extract pooled segment features

`s01_extract_features_and_segments.py` reads audio files, runs WavLM, applies the ZeroSyl pooler, and saves:

* one pooled feature matrix per utterance
* one segment-boundary matrix per utterance

Each feature file has shape:

```text
num_segments × feature_dim
```

Each segment file has shape:

```text
num_segments × 3
```

with columns:

```text
start_frame, end_frame, unit_id
```

The `unit_id` column is initially filled with dummy labels and is populated later by clustering or inference.

Example:

```bash
uv run python s01_extract_features_and_segments.py \
    --wav-dir ../data/audio/LibriSpeech/train-clean-100 \
    --features-dir output/features/zerosyl \
    --segments-dir output/segments/zerosyl \
    --wavlm-ckpt-path ../checkpoints/WavLM-Large.pt \
    --batch-size 4 \
    --num-workers 8
```

### 2. Train scaler and PCA models

`s01b_train_scaler_pca_models.py` trains a `StandardScaler` and PCA model on the pooled segment features.

Example:

```bash
uv run python s01b_train_scaler_pca_models.py \
    --config.features-dir output/features/zerosyl/LibriSpeech/train-clean-100 \
    --config.output-dir output/models/zerosyl/LibriSpeech/train-clean-100 \
    --config.number_of_components 350
```

This produces:

```text
scaler.joblib
pca.joblib
```

These models should be trained on the training set and reused for inference datasets.

### 3. Train graph clustering

`s02_train_graph_clustering.py` loads pooled segment features, applies the trained scaler and PCA model, constructs a graph, and clusters it with Leiden.

A kNN graph is used when `--edges.num-neighbors` is set. A full cosine graph is used when `--edges.num-neighbors` is omitted or set to `None`.

Example kNN graph run:

```bash
uv run python s02_train_graph_clustering.py \
    --features-dir output/features/zerosyl/LibriSpeech/train-clean-100 \
    --segments-dir output/segments/zerosyl/LibriSpeech/train-clean-100 \
    --scaler-path output/models/zerosyl/LibriSpeech/train-clean-100/scaler.joblib \
    --pca-path output/models/zerosyl/LibriSpeech/train-clean-100/pca.joblib \
    --edges.num-neighbors 100 \
    --edges.min-sim 0.55 \
    --cluster.resolution-specified 0.1100 \
    --cluster.leiden-iterations 2 \
    --output-dir output/clustering/zerosyl/inference-test/LibriSpeech/train-clean-100 \
    --show-progress
```

The graph clustering script saves:

* labelled segment files
* clustering artefacts
* metadata including runtime, peak RAM, final cluster count and graph settings

### 4. Train K-means

`s02b_train_faiss_kmeans.py` trains FAISS K-means on the same scaled and PCA-projected segment representations.

Example:

```bash
uv run python s02b_train_faiss_kmeans.py \
    --features-dir output/features/zerosyl/LibriSpeech/train-clean-100 \
    --segments-dir output/segments/zerosyl/LibriSpeech/train-clean-100 \
    --scaler-path output/models/zerosyl/LibriSpeech/train-clean-100/scaler.joblib \
    --pca-path output/models/zerosyl/LibriSpeech/train-clean-100/pca.joblib \
    --output-dir output/clustering/zerosyl/inference-test/LibriSpeech/train-clean-100 \
    --num-clusters 33902 \
    --show-progress
```

The K-means script saves:

* labelled segment files
* centroids
* metadata including runtime, peak RAM and final cluster count

### 5. Infer labels on new data

`s04_infer_labels.py` assigns labels to new segment features using either:

* nearest-centroid assignment for K-means
* nearest-reference-segment voting for graph clustering

For graph clustering, FAISS HNSW is used to retrieve nearest labelled reference segments. The default is majority voting over `k_neighbors=3`.

Example K-means inference:

```bash
uv run python s04_infer_labels.py \
    --clustering-method kmeans++ \
    --features-dir output/features/zerosyl/LibriSpeech/train-* \
    --segments-dir output/segments/zerosyl/LibriSpeech/train-* \
    --models-dir output/models/zerosyl/LibriSpeech/train-clean-100 \
    --reference-dir output/clustering-artifacts/zerosyl/inference-test/LibriSpeech/train-clean-100/<kmeans_run_dir> \
    --clustering-dir output/clustering/zerosyl/inference-test/LibriSpeech/train-clean-100/<kmeans_run_dir>
```

Example graph inference:

```bash
uv run python s04_infer_labels.py \
    --clustering-method graph_knn \
    --features-dir output/features/zerosyl/LibriSpeech/train-* \
    --segments-dir output/segments/zerosyl/LibriSpeech/train-* \
    --models-dir output/models/zerosyl/LibriSpeech/train-clean-100 \
    --reference-dir output/clustering-artifacts/zerosyl/inference-test/LibriSpeech/train-clean-100/<graph_run_dir> \
    --clustering-dir output/clustering/zerosyl/inference-test/LibriSpeech/train-clean-100/<graph_run_dir> \
    --k-neighbors 100 \
    --hnsw-m 32 \
    --hnsw-ef-search 64
```

The inference script is resumable: completed output files are skipped on reruns.

### 6. Evaluate clustering

`s03_evaluate_clustering.py` evaluates labelled segments against TextGrid annotations.

Example:

```bash
uv run python s03_evaluate_clustering.py \
    --textgrid-dir ../data/alignments/LibriSpeech/train-* \
    --segments-dir output/inferred_segments/zerosyl/LibriSpeech/train-* \
    --compute-nes \
    --save-to-csv
```

The main reported metrics are:

| Metric    | Meaning                                 |
| --------- | --------------------------------------- |
| `wnes`    | Weighted normalized edit score.         |
| `iwnes`   | Inverse weighted normalized edit score. |
| `f1_wnes` | Harmonic mean of wNES and iwNES.        |

When `--save-to-csv` is used, results are appended to an `evaluation.csv` file under `output/evaluation`.

### 7. Read evaluation CSV

`s04b_read_eval_csv.py` loads an evaluation CSV, sorts by `f1_wnes`, converts metrics to percentages, and prints the top rows.

Example:

```bash
uv run python s04b_read_eval_csv.py \
    --eval-path output/evaluation/zerosyl/LibriSpeech/train-*/kmeans++_33902/zerosyl/inference-test/LibriSpeech/train-clean-100/evaluation.csv
```

### 8. Plot Zipf distributions

`s05_zipf.py` plots rank-frequency distributions for induced clusters and discovered phone-sequence types.

Example:

```bash
uv run python s05_zipf.py \
    --textgrid-dir ../data/alignments/LibriSpeech/train-* \
    --segments-dir output/inferred_segments/zerosyl/LibriSpeech/train-* \
    --output-path figs \
    --save-as-png \
    --matching-method kmeans
```

The script can plot:

* induced cluster distributions
* discovered phone-sequence distributions
* phone, biphone and triphone distributions

## Running the orchestration script

The `run.sh` script provides a configurable end-to-end experiment runner. Edit the configuration block at the top of the file:

```bash
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
```

Then select which stages to run:

```bash
RUN_EXTRACT=false
RUN_TRAIN_MODELS=false
RUN_GRAPH=false
RUN_KMEANS=false
RUN_INFERENCE=true
RUN_EVALUATE=true
SAVE_TO_CSV=true
RUN_ZIPF=false
```

Run:

```bash
bash run.sh
```

## Output directories

By default, outputs are written under `output/`:

```text
output/
├── features/
├── segments/
├── models/
├── clustering/
├── clustering-artifacts/
├── inferred_segments/
└── evaluation/
```

The usual flow is:

```text
features + segments
    ↓
scaler + PCA models
    ↓
clustering outputs + artefacts
    ↓
inferred segment labels
    ↓
evaluation CSVs and figures
```

## Notes on graph settings

For graph clustering, the most important parameters are:

| Parameter                        | Meaning                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------- |
| `--edges.num-neighbors`          | Number of kNN candidate neighbours. If omitted, full graph construction is used. |
| `--edges.min-sim`                | Minimum cosine similarity threshold, (\tau).                                     |
| `--cluster.resolution-specified` | Leiden/CPM resolution parameter, (\gamma).                                       |
| `--cluster.leiden-iterations`    | Number of Leiden iterations.                                                     |

For inference with graph clustering:

| Parameter          | Meaning                                                     |
| ------------------ | ----------------------------------------------------------- |
| `--k-neighbors`    | Number of nearest reference segments used for label voting. |
| `--hnsw-m`         | HNSW connectivity parameter.                                |
| `--hnsw-ef-search` | HNSW search parameter controlling speed/recall trade-off.   |

## Reproducibility

This repository uses:

* `pyproject.toml` for project metadata and dependencies
* `uv.lock` for locked dependency versions
* explicit random seeds for subsampling where applicable
* saved scaler/PCA models for consistent train/inference transformations

To reproduce the environment, run:

```bash
uv sync
```

Then run the relevant pipeline scripts with `uv run`.

## Citation

If you use this code, please cite the associated work once available.

```bibtex
@misc{slabbert2026graphclustering,
  title  = {Scaling Graph Clustering for Unsupervised Speech Unit Discovery},
  author = {Slabbert, Danel and Kamper, Herman},
  year   = {2026},
  note   = {Code repository}
}
```

## License

Copyright (c) 2026 Danel Slabbert, Stellenbosch University.

Add the appropriate repository license here.
