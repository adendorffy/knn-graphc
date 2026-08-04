import gc
from dataclasses import dataclass

import faiss
import numpy as np
from tqdm import tqdm
from typing import Optional
import os 


@dataclass
class BuildEdgesConfig:
    num_neighbors: Optional[int] = None
    """Number of nearest neighbors per node, excluding self.
    
    Use None for a full cosine-threshold graph.
    Use an int for a kNN graph.
    """

    min_sim: float = 0.0
    """Minimum cosine similarity for an edge."""

    batch_size: int = 500
    """Query batch size for faiss kNN search."""

    device: int = 0
    """CUDA device index for faiss GPU search."""

    use_float16: bool = False
    """Use float16 storage in the faiss GPU index."""

    temp_memory_gb: float = 1.0
    """Maximum VRAM that faiss should allocate to search the index."""


def build_edges_knn(
    features: np.ndarray,
    config: BuildEdgesConfig,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a kNN graph from cosine similarity using faiss GPU exact search."""

    assert features.ndim == 2
    num_nodes, dim = features.shape
    assert config.num_neighbors >= 1
    assert config.num_neighbors <= num_nodes

    faiss.omp_set_num_threads(os.cpu_count())

    features = np.ascontiguousarray(features, dtype=np.float32)
    faiss.normalize_L2(features)
    query_k = min(config.num_neighbors + 1, num_nodes)

    res = faiss.StandardGpuResources()
    res.setTempMemory(int(config.temp_memory_gb * 1024**3))

    index_config = faiss.GpuIndexFlatConfig()
    index_config.device = config.device
    index_config.useFloat16 = config.use_float16
    index = faiss.GpuIndexFlatIP(res, dim, index_config)

    try:
        index.add(features)

        edges_indices: list[np.ndarray] = []
        edges_weights: list[np.ndarray] = []

        for start_idx in tqdm(
            range(0, num_nodes, config.batch_size),
            desc="Computing edges",
            disable=not show_progress,
        ):
            end_idx = min(start_idx + config.batch_size, num_nodes)
            batch = features[start_idx:end_idx]

            distances, indices = index.search(batch, query_k)

            src = np.repeat(np.arange(start_idx, end_idx, dtype=np.int64), query_k)
            dst = indices.reshape(-1).astype(np.int64)
            weights_arr = distances.reshape(-1)

            valid_mask = (src != dst) & (weights_arr >= config.min_sim)

            if valid_mask.any():
                edges_indices.append(
                    np.stack([src[valid_mask], dst[valid_mask]], axis=1)
                )
                edges_weights.append(weights_arr[valid_mask].astype(np.float32))

        if not edges_indices:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)

        edges_indices = np.concatenate(edges_indices, axis=0)
        edges_weights = np.concatenate(edges_weights, axis=0)

        # dedupe undirected edges
        # i.e. if a->b and b->a are present in the graph just use one a-b edge
        edges_indices.sort(axis=1)
        keys = edges_indices[:, 0] * num_nodes + edges_indices[:, 1]
        _, unique_indices = np.unique(keys, return_index=True)

        return edges_indices[unique_indices], edges_weights[unique_indices]

    finally:
        index.reset()
        del index, res, features
        gc.collect()

def build_edges_full(
    features: np.ndarray,
    config: BuildEdgesConfig,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a full cosine-threshold graph using batched matrix multiplication.

    Keeps only upper-triangular edges (i < j), so the graph is undirected
    without needing a later deduplication pass.
    """

    assert features.ndim == 2
    num_nodes, dim = features.shape

    features = np.ascontiguousarray(features, dtype=np.float32)
    faiss.normalize_L2(features)
    
    edges_indices: list[np.ndarray] = []
    edges_weights: list[np.ndarray] = []

    for start_idx in tqdm(
        range(0, num_nodes, config.batch_size),
        desc="Computing full cosine edges",
        disable=not show_progress,
    ):
        end_idx = min(start_idx + config.batch_size, num_nodes)
        batch = features[start_idx:end_idx]

        sims = batch @ features[start_idx:].T  # shape (batch_rows, num_nodes - start_idx)

        local_rows = np.arange(start_idx, end_idx)
        cols = np.arange(start_idx, num_nodes)
        row_grid, col_grid = np.meshgrid(local_rows, cols, indexing="ij")

        # Strict upper triangle within this slice: dst > src
        valid_mask = (col_grid > row_grid) & (sims >= config.min_sim)

        if valid_mask.any():
            src = row_grid[valid_mask].astype(np.int64)
            dst = col_grid[valid_mask].astype(np.int64)
            w = sims[valid_mask].astype(np.float32)
            edges_indices.append(np.stack([src, dst], axis=1))
            edges_weights.append(w)

        del sims, row_grid, col_grid, valid_mask

    if not edges_indices:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)

    edges_indices = np.concatenate(edges_indices, axis=0)
    edges_weights = np.concatenate(edges_weights, axis=0)


    return edges_indices, edges_weights