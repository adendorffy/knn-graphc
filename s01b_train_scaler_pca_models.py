from dataclasses import dataclass
from pathlib import Path

import tyro
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.load import load_features

@dataclass(frozen=True)
class ScalingPCAConfig:
    features_dir: Path
    """Output directory for per-utterance pooled segment feature .npy files."""

    output_dir: Path
    """Output directory for serialized scaler and PCA models."""

    number_of_components: int = 350
    """Number of PCA components to keep."""

    show_progress: bool = True
    """Show progress bars and status output during pipeline stages."""

def main(config: ScalingPCAConfig) -> None:
    features = load_features(config.features_dir, show_progress=config.show_progress)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    pca = PCA(n_components=config.number_of_components)
    features = pca.fit(features)
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.output_dir / "scaler.joblib")
    joblib.dump(pca, config.output_dir / "pca.joblib")

if __name__ == "__main__":
    tyro.cli(main)