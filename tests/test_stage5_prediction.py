import pytest
from pathlib import Path
import shutil
from hoop_vision.stage5_prediction import PredictionModel
from hoop_vision.config import Config


@pytest.fixture
def clean_output_dir():
    """Clean output directory before test."""
    output_dir = Path("data/processed/predictions")
    model_dir = Path("models/checkpoints")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    yield
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if model_dir.exists():
        shutil.rmtree(model_dir)


def test_prediction_model_trains_baseline():
    """Test that PredictionModel trains a baseline model."""
    features_dir = Path("data/processed/features")
    if not features_dir.exists() or not list(features_dir.glob("*/features.csv")):
        pytest.skip("No features available")

    config = Config()
    model = PredictionModel(config)

    # For now, just test that we can train (even with dummy labels)
    # In real usage, clips need manual labels
    result = model.train_baseline()

    assert "model_path" in result
    assert "metrics" in result
