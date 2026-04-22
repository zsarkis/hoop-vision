import pytest
from pathlib import Path
import shutil
import json
import pandas as pd
from hoop_vision.stage4_features import FeatureExtractor
from hoop_vision.config import Config


@pytest.fixture
def clean_output_dir():
    """Clean output directory before test."""
    output_dir = Path("data/processed/features")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    yield
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_feature_extractor_creates_features(clean_output_dir):
    """Test that FeatureExtractor creates feature vectors."""
    graphs_dir = Path("data/processed/graphs/test_clip")
    if not graphs_dir.exists() or not (graphs_dir / "graph.gpickle").exists():
        pytest.skip("No graph available")

    config = Config()
    extractor = FeatureExtractor(config)

    clip_id = "test_clip"
    output_path = extractor.process_clip(clip_id)

    assert output_path.exists()
    features_file = output_path / "features.csv"
    assert features_file.exists()


def test_features_have_required_columns(clean_output_dir):
    """Test that features CSV has expected columns."""
    graphs_dir = Path("data/processed/graphs/test_clip")
    if not graphs_dir.exists() or not (graphs_dir / "graph.gpickle").exists():
        pytest.skip("No graph available")

    config = Config()
    extractor = FeatureExtractor(config)

    clip_id = "test_clip"
    output_path = extractor.process_clip(clip_id)

    features_file = output_path / "features.csv"
    df = pd.read_csv(features_file)

    # Should have at least basic columns
    assert "frame_idx" in df.columns
    assert "timestamp" in df.columns
    assert "num_detections" in df.columns
