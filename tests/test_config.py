# tests/test_config.py
import pytest
from pathlib import Path
from hoop_vision.config import Config


def test_config_loads_default_yaml_path():
    """Test that config loads from config.yaml when no path is provided."""
    config = Config()
    assert config.fps == 2
    assert config.output_dir == Path("data/processed")
    assert config.yolo_model == "yolov8n.pt"


def test_config_loads_from_yaml():
    """Test that config loads from YAML file."""
    config = Config(config_path="config.yaml")
    assert config.fps == 2
    assert config.yolo_model == "yolov8n.pt"
