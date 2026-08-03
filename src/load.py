from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_features(input_dir: Path | str, show_progress: bool = True) -> np.ndarray:
    """
    load 2D features from .npy files using minimal RAM.
    """

    input_dir = Path(input_dir)
    assert input_dir.is_dir()

    feature_paths = sorted(input_dir.rglob("*.npy"))
    assert feature_paths

    total_features = 0
    feature_dim = None

    # first determine the number of features in the dataset
    # we don't have to load the features yet (mmap mode reads shape only)
    for path in tqdm(feature_paths, desc="Scanning shapes", disable=not show_progress):
        mmap_arr = np.load(path, mmap_mode="r")

        assert mmap_arr.ndim == 2
        num_feats, dim = mmap_arr.shape

        if feature_dim is None:
            feature_dim = dim
        assert feature_dim == dim

        total_features += num_feats

    # now make the space for the data
    all_features = np.empty((total_features, feature_dim), dtype=np.float32)

    # and load in place
    current_idx = 0
    for path in tqdm(feature_paths, desc="Loading features", disable=not show_progress):
        arr = np.load(path)
        num_feats = arr.shape[0]
        all_features[current_idx : current_idx + num_feats] = arr.astype(
            np.float32, copy=False
        )
        current_idx += num_feats

    return all_features

def get_subsample_indices(
    total_features: int,
    subsample: float,
    seed: int = 0,
    cache_dir: Path = Path("output/subsample_indices"),
) -> np.ndarray:
    """
    Returns the same subsample indices every time for a given
    (total_features, subsample, seed) combination, regardless of
    what other random calls have happened elsewhere in the process.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"n{total_features}_sub{subsample}_seed{seed}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    rng = np.random.default_rng(seed)  # independent of global np.random state
    subsample_int = int(subsample*total_features)
    indices = rng.choice(total_features, subsample_int, replace=False)
    indices.sort()
    np.save(cache_path, indices)
    return indices