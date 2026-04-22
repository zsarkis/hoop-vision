# tests/test_config.py
import pytest
from pathlib import Path
from hoop_vision.config import Config


def test_config_loads_defaults():
    """Test that config loads with default values."""
    config = Config()
    assert config.fps == 2
    assert config.output_dir == Path("data/processed")
    assert config.yolo_model == "yolov8n.pt"


def test_config_loads_from_yaml():
    """Test that config loads from YAML file."""
    config = Config(config_path="config.yaml")
    assert config.fps is not None
    assert config.output_dir is not None
