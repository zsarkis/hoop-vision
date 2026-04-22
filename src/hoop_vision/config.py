# src/hoop_vision/config.py
"""Configuration management for Hoop Vision pipeline."""

from pathlib import Path
from typing import Optional
import yaml


class Config:
    """Pipeline configuration loaded from YAML."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config.

        Args:
            config_path: Path to YAML config file. If None, uses config.yaml in project root.
        """
        if config_path is None:
            config_path = "config.yaml"

        self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Ingestion
        self.fps = config_data["ingestion"]["fps"]
        self.frames_output_dir = Path(config_data["ingestion"]["output_dir"])

        # Detection
        self.yolo_model = config_data["detection"]["yolo_model"]
        self.confidence_threshold = config_data["detection"]["confidence_threshold"]
        self.nms_threshold = config_data["detection"]["nms_threshold"]
        self.events_output_dir = Path(config_data["detection"]["output_dir"])

        # Graph
        self.max_temporal_distance = config_data["graph"]["max_temporal_distance"]
        self.graphs_output_dir = Path(config_data["graph"]["output_dir"])

        # Features
        self.lookback_window = config_data["features"]["lookback_window"]
        self.features_output_dir = Path(config_data["features"]["output_dir"])

        # Prediction
        self.model_type = config_data["prediction"]["model_type"]
        self.test_split = config_data["prediction"]["test_split"]
        self.random_seed = config_data["prediction"]["random_seed"]
        self.predictions_output_dir = Path(config_data["prediction"]["output_dir"])
        self.model_save_dir = Path(config_data["prediction"]["model_save_dir"])

        # Data
        self.raw_clips_dir = Path(config_data["data"]["raw_clips_dir"])
        self.sample_clips_dir = Path(config_data["data"]["sample_clips_dir"])

    @property
    def output_dir(self) -> Path:
        """Root output directory for processed data."""
        return Path("data/processed")
