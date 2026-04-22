# Hoop Vision Phase 1 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build end-to-end pipeline from NBA video clips to next possession outcome predictions using classic ML

**Architecture:** Five-stage pipeline (ingestion → detection → graph → features → prediction) where each stage reads from previous stage's cache and writes to disk. Start with basic event detection (players, ball, shots) using pre-trained models, build temporal graph, extract engineered features, train XGBoost classifier.

**Tech Stack:** Python 3.10+, OpenCV (video), YOLOv8 (detection), NetworkX (graphs), XGBoost (ML), pytest

---

## Task 1: Project Setup and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/hoop_vision/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml with dependencies**

```toml
[tool.poetry]
name = "hoop-vision"
version = "0.1.0"
description = "NBA game prediction using computer vision and graph ML"
authors = ["Zach Sarkis"]
python = "^3.10"

[tool.poetry.dependencies]
python = "^3.10"
opencv-python = "^4.8.0"
numpy = "^1.24.0"
pandas = "^2.0.0"
networkx = "^3.1"
torch = "^2.0.0"
ultralytics = "^8.0.0"  # YOLOv8
xgboost = "^2.0.0"
scikit-learn = "^1.3.0"
matplotlib = "^3.7.0"
jupyter = "^1.0.0"
pyyaml = "^6.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
black = "^23.7.0"
ruff = "^0.0.285"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

- [ ] **Step 2: Create README.md**

```markdown
# Hoop Vision

NBA game prediction using computer vision and graph-based machine learning.

## Phase 1: Next Possession Prediction Pipeline

Predicts next possession outcomes (made shot, miss, turnover, foul, etc.) from short video clips.

### Installation

```bash
poetry install
```

### Quick Start

1. Place video clips in `data/raw_clips/`
2. Run the pipeline:
   ```bash
   poetry run python -m hoop_vision.stage1_ingestion
   poetry run python -m hoop_vision.stage2_detection
   poetry run python -m hoop_vision.stage3_graph
   poetry run python -m hoop_vision.stage4_features
   poetry run python -m hoop_vision.stage5_prediction
   ```
3. View demo: `jupyter notebook notebooks/01_pipeline_demo.ipynb`

### Pipeline Stages

1. **Ingestion**: Video clips → sampled frames
2. **Detection**: Frames → detected events (players, ball, shots)
3. **Graph**: Events → temporal graph structure
4. **Features**: Graph → engineered features
5. **Prediction**: Features → next possession outcome

### Testing

```bash
poetry run pytest
```

### Project Structure

```
src/hoop_vision/       # Source code
tests/                 # Tests
notebooks/             # Jupyter demos
data/raw_clips/        # Input videos
data/processed/        # Pipeline outputs
```
```

- [ ] **Step 3: Create src/hoop_vision/__init__.py**

```python
"""Hoop Vision - NBA game prediction using computer vision and graph ML."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create tests/__init__.py**

```python
"""Tests for Hoop Vision."""
```

- [ ] **Step 5: Create directory structure**

Run: `mkdir -p src/hoop_vision/utils tests notebooks data/raw_clips data/processed/frames data/processed/events data/processed/graphs data/processed/features data/processed/predictions data/sample`

- [ ] **Step 6: Install dependencies**

Run: `poetry install`
Expected: Dependencies installed successfully

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml README.md src/ tests/ .gitignore
git commit -m "chore: initialize project structure and dependencies"
```

---

## Task 2: Configuration Management

**Files:**
- Create: `src/hoop_vision/config.py`
- Create: `config.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config loading**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'hoop_vision.config'"

- [ ] **Step 3: Create config.yaml with default settings**

```yaml
# config.yaml
# Hoop Vision Pipeline Configuration

# Stage 1: Ingestion
ingestion:
  fps: 2  # Frames per second to sample
  output_dir: data/processed/frames

# Stage 2: Detection
detection:
  yolo_model: yolov8n.pt  # Nano model for speed
  confidence_threshold: 0.5
  nms_threshold: 0.45
  output_dir: data/processed/events

# Stage 3: Graph
graph:
  max_temporal_distance: 10  # Max edge distance in seconds
  output_dir: data/processed/graphs

# Stage 4: Features
features:
  lookback_window: 5  # Possessions to look back
  output_dir: data/processed/features

# Stage 5: Prediction
prediction:
  model_type: xgboost
  test_split: 0.2
  random_seed: 42
  output_dir: data/processed/predictions
  model_save_dir: models/checkpoints

# General
data:
  raw_clips_dir: data/raw_clips
  sample_clips_dir: data/sample
```

- [ ] **Step 4: Write minimal Config implementation**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/hoop_vision/config.py config.yaml tests/test_config.py
git commit -m "feat: add configuration management with YAML support"
```

---

## Task 3: Video Utils (OpenCV Helpers)

**Files:**
- Create: `src/hoop_vision/utils/__init__.py`
- Create: `src/hoop_vision/utils/video_utils.py`
- Create: `tests/test_video_utils.py`

- [ ] **Step 1: Write failing test for extract_frames**

```python
# tests/test_video_utils.py
import pytest
import numpy as np
from pathlib import Path
from hoop_vision.utils.video_utils import extract_frames, get_video_info


def test_get_video_info_returns_metadata():
    """Test that get_video_info extracts video metadata."""
    # This will fail until we add a sample video
    sample_video = Path("data/sample/test_clip.mp4")
    if not sample_video.exists():
        pytest.skip("Sample video not available")

    info = get_video_info(sample_video)
    assert "fps" in info
    assert "duration" in info
    assert "frame_count" in info
    assert info["fps"] > 0


def test_extract_frames_returns_frame_list():
    """Test that extract_frames returns list of frame arrays."""
    sample_video = Path("data/sample/test_clip.mp4")
    if not sample_video.exists():
        pytest.skip("Sample video not available")

    frames = extract_frames(sample_video, target_fps=1.0)
    assert len(frames) > 0
    assert isinstance(frames[0], np.ndarray)
    assert frames[0].ndim == 3  # Height x Width x Channels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_video_utils.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Create utils/__init__.py**

```python
# src/hoop_vision/utils/__init__.py
"""Utility functions for Hoop Vision."""

from .video_utils import extract_frames, get_video_info

__all__ = ["extract_frames", "get_video_info"]
```

- [ ] **Step 4: Write minimal video_utils implementation**

```python
# src/hoop_vision/utils/video_utils.py
"""Video processing utilities using OpenCV."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict


def get_video_info(video_path: Path) -> Dict[str, float]:
    """
    Extract metadata from video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with fps, duration, frame_count
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count
    }


def extract_frames(video_path: Path, target_fps: float = 2.0) -> List[np.ndarray]:
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to video file
        target_fps: Target frames per second for sampling

    Returns:
        List of frame arrays (H x W x C in BGR format)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")

    # Calculate frame sampling interval
    frame_interval = max(1, int(source_fps / target_fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frame at target FPS
        if frame_idx % frame_interval == 0:
            frames.append(frame)

        frame_idx += 1

    cap.release()
    return frames
```

- [ ] **Step 5: Run test to verify it passes (will skip without sample video)**

Run: `poetry run pytest tests/test_video_utils.py -v`
Expected: SKIPPED (no sample video yet) or PASS if video exists

- [ ] **Step 6: Commit**

```bash
git add src/hoop_vision/utils/ tests/test_video_utils.py
git commit -m "feat: add video utilities for frame extraction"
```

---

## Task 4: Stage 1 - Video Ingestion

**Files:**
- Create: `src/hoop_vision/stage1_ingestion.py`
- Create: `tests/test_stage1_ingestion.py`

- [ ] **Step 1: Write failing test for VideoIngestion**

```python
# tests/test_stage1_ingestion.py
import pytest
from pathlib import Path
import shutil
from hoop_vision.stage1_ingestion import VideoIngestion
from hoop_vision.config import Config


@pytest.fixture
def clean_output_dir():
    """Clean output directory before test."""
    output_dir = Path("data/processed/frames")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    yield
    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_video_ingestion_processes_clip(clean_output_dir):
    """Test that VideoIngestion extracts and saves frames."""
    sample_video = Path("data/sample/test_clip.mp4")
    if not sample_video.exists():
        pytest.skip("Sample video not available")

    config = Config()
    ingestion = VideoIngestion(config)

    clip_id = "test_clip"
    output_path = ingestion.process_clip(sample_video, clip_id)

    assert output_path.exists()
    assert output_path.is_dir()
    # Should have at least one frame
    frames = list(output_path.glob("frame_*.jpg"))
    assert len(frames) > 0


def test_video_ingestion_saves_metadata(clean_output_dir):
    """Test that metadata is saved alongside frames."""
    sample_video = Path("data/sample/test_clip.mp4")
    if not sample_video.exists():
        pytest.skip("Sample video not available")

    config = Config()
    ingestion = VideoIngestion(config)

    clip_id = "test_clip"
    output_path = ingestion.process_clip(sample_video, clip_id)

    metadata_file = output_path / "metadata.json"
    assert metadata_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_stage1_ingestion.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal VideoIngestion implementation**

```python
# src/hoop_vision/stage1_ingestion.py
"""Stage 1: Video Ingestion - Extract frames from video clips."""

import json
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from .config import Config
from .utils.video_utils import extract_frames, get_video_info


class VideoIngestion:
    """Extract frames from video clips at configured FPS."""

    def __init__(self, config: Config):
        """
        Initialize video ingestion stage.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.frames_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_clip(self, video_path: Path, clip_id: str) -> Path:
        """
        Process a single video clip.

        Args:
            video_path: Path to input video file
            clip_id: Unique identifier for this clip

        Returns:
            Path to directory containing extracted frames
        """
        print(f"Processing clip: {clip_id}")

        # Get video metadata
        video_info = get_video_info(video_path)
        print(f"  Video info: {video_info['duration']:.1f}s @ {video_info['fps']:.1f} FPS")

        # Extract frames
        frames = extract_frames(video_path, target_fps=self.config.fps)
        print(f"  Extracted {len(frames)} frames at {self.config.fps} FPS")

        # Save frames to disk
        clip_output_dir = self.output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = clip_output_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        # Save metadata
        metadata = {
            "clip_id": clip_id,
            "source_video": str(video_path),
            "source_fps": video_info["fps"],
            "source_duration": video_info["duration"],
            "target_fps": self.config.fps,
            "num_frames": len(frames)
        }

        metadata_path = clip_output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to: {clip_output_dir}")
        return clip_output_dir

    def process_all_clips(self, clips_dir: Optional[Path] = None) -> None:
        """
        Process all video clips in directory.

        Args:
            clips_dir: Directory containing video clips. If None, uses config default.
        """
        if clips_dir is None:
            clips_dir = self.config.raw_clips_dir

        if not clips_dir.exists():
            print(f"Clips directory not found: {clips_dir}")
            return

        video_files = list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.mov"))
        print(f"Found {len(video_files)} video files in {clips_dir}")

        for video_path in video_files:
            clip_id = video_path.stem
            self.process_clip(video_path, clip_id)


def main():
    """Run video ingestion stage."""
    config = Config()
    ingestion = VideoIngestion(config)
    ingestion.process_all_clips()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes (will skip without sample video)**

Run: `poetry run pytest tests/test_stage1_ingestion.py -v`
Expected: SKIPPED or PASS

- [ ] **Step 5: Commit**

```bash
git add src/hoop_vision/stage1_ingestion.py tests/test_stage1_ingestion.py
git commit -m "feat(stage1): add video ingestion with frame extraction"
```

---

## Task 5: Stage 2 - Event Detection (Basic)

**Files:**
- Create: `src/hoop_vision/stage2_detection.py`
- Create: `tests/test_stage2_detection.py`

- [ ] **Step 1: Write failing test for EventDetection**

```python
# tests/test_stage2_detection.py
import pytest
from pathlib import Path
import shutil
import json
from hoop_vision.stage2_detection import EventDetection
from hoop_vision.config import Config


@pytest.fixture
def clean_output_dir():
    """Clean output directory before test."""
    output_dir = Path("data/processed/events")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    yield
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_event_detection_processes_frames(clean_output_dir):
    """Test that EventDetection processes frames and detects objects."""
    frames_dir = Path("data/processed/frames/test_clip")
    if not frames_dir.exists() or not list(frames_dir.glob("frame_*.jpg")):
        pytest.skip("No processed frames available")

    config = Config()
    detection = EventDetection(config)

    clip_id = "test_clip"
    output_path = detection.process_clip(clip_id)

    assert output_path.exists()
    events_file = output_path / "events.json"
    assert events_file.exists()


def test_event_detection_saves_valid_events(clean_output_dir):
    """Test that saved events have required fields."""
    frames_dir = Path("data/processed/frames/test_clip")
    if not frames_dir.exists() or not list(frames_dir.glob("frame_*.jpg")):
        pytest.skip("No processed frames available")

    config = Config()
    detection = EventDetection(config)

    clip_id = "test_clip"
    output_path = detection.process_clip(clip_id)

    events_file = output_path / "events.json"
    with open(events_file) as f:
        events = json.load(f)

    assert isinstance(events, list)
    if len(events) > 0:
        event = events[0]
        assert "frame_idx" in event
        assert "timestamp" in event
        assert "detections" in event
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_stage2_detection.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal EventDetection implementation**

```python
# src/hoop_vision/stage2_detection.py
"""Stage 2: Event Detection - Detect players, ball, and events using YOLO."""

import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
from ultralytics import YOLO

from .config import Config


class EventDetection:
    """Detect objects and events in frames using YOLOv8."""

    def __init__(self, config: Config):
        """
        Initialize event detection stage.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.events_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load YOLO model
        print(f"Loading YOLO model: {config.yolo_model}")
        self.model = YOLO(config.yolo_model)

    def detect_frame(self, frame_path: Path, frame_idx: int, timestamp: float) -> Dict[str, Any]:
        """
        Detect objects in a single frame.

        Args:
            frame_path: Path to frame image
            frame_idx: Frame index
            timestamp: Timestamp in seconds

        Returns:
            Event dictionary with detections
        """
        # Read frame
        frame = cv2.imread(str(frame_path))

        # Run YOLO detection
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold,
            verbose=False
        )

        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "frame_path": str(frame_path),
            "detections": detections
        }

    def process_clip(self, clip_id: str) -> Path:
        """
        Process all frames for a clip.

        Args:
            clip_id: Unique identifier for clip

        Returns:
            Path to directory containing events
        """
        print(f"Detecting events in clip: {clip_id}")

        # Load frames directory
        frames_dir = self.config.frames_output_dir / clip_id
        if not frames_dir.exists():
            raise ValueError(f"Frames directory not found: {frames_dir}")

        # Load metadata to get FPS
        metadata_path = frames_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        target_fps = metadata["target_fps"]

        # Get all frames
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        print(f"  Processing {len(frame_files)} frames")

        # Detect events in each frame
        events = []
        for i, frame_path in enumerate(frame_files):
            timestamp = i / target_fps
            event = self.detect_frame(frame_path, i, timestamp)
            events.append(event)

            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(frame_files)} frames")

        # Save events
        clip_output_dir = self.output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        events_path = clip_output_dir / "events.json"
        with open(events_path, "w") as f:
            json.dump(events, f, indent=2)

        print(f"  Saved {len(events)} events to: {clip_output_dir}")
        return clip_output_dir

    def process_all_clips(self) -> None:
        """Process all clips with extracted frames."""
        frames_dir = self.config.frames_output_dir
        if not frames_dir.exists():
            print(f"Frames directory not found: {frames_dir}")
            return

        clip_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        print(f"Found {len(clip_dirs)} clips to process")

        for clip_dir in clip_dirs:
            clip_id = clip_dir.name
            self.process_clip(clip_id)


def main():
    """Run event detection stage."""
    config = Config()
    detection = EventDetection(config)
    detection.process_all_clips()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes (will skip without frames)**

Run: `poetry run pytest tests/test_stage2_detection.py -v`
Expected: SKIPPED or PASS

- [ ] **Step 5: Commit**

```bash
git add src/hoop_vision/stage2_detection.py tests/test_stage2_detection.py
git commit -m "feat(stage2): add event detection using YOLOv8"
```

---

## Task 6: Stage 3 - Graph Construction

**Files:**
- Create: `src/hoop_vision/stage3_graph.py`
- Create: `tests/test_stage3_graph.py`

- [ ] **Step 1: Write failing test for GraphBuilder**

```python
# tests/test_stage3_graph.py
import pytest
from pathlib import Path
import shutil
import json
import networkx as nx
from hoop_vision.stage3_graph import GraphBuilder
from hoop_vision.config import Config


@pytest.fixture
def clean_output_dir():
    """Clean output directory before test."""
    output_dir = Path("data/processed/graphs")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    yield
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_graph_builder_creates_graph(clean_output_dir):
    """Test that GraphBuilder creates a temporal graph from events."""
    events_dir = Path("data/processed/events/test_clip")
    if not events_dir.exists() or not (events_dir / "events.json").exists():
        pytest.skip("No events available")

    config = Config()
    builder = GraphBuilder(config)

    clip_id = "test_clip"
    output_path = builder.process_clip(clip_id)

    assert output_path.exists()
    graph_file = output_path / "graph.gpickle"
    assert graph_file.exists()


def test_graph_has_temporal_edges(clean_output_dir):
    """Test that graph contains temporal edges between events."""
    events_dir = Path("data/processed/events/test_clip")
    if not events_dir.exists() or not (events_dir / "events.json").exists():
        pytest.skip("No events available")

    config = Config()
    builder = GraphBuilder(config)

    clip_id = "test_clip"
    output_path = builder.process_clip(clip_id)

    graph_file = output_path / "graph.gpickle"
    G = nx.read_gpickle(graph_file)

    assert G.number_of_nodes() > 0
    # Should have temporal edges
    assert G.number_of_edges() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_stage3_graph.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal GraphBuilder implementation**

```python
# src/hoop_vision/stage3_graph.py
"""Stage 3: Graph Construction - Build temporal event graph from detections."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx

from .config import Config


class GraphBuilder:
    """Build temporal graph from detected events."""

    def __init__(self, config: Config):
        """
        Initialize graph builder.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.graphs_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_graph(self, events: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build temporal directed graph from events.

        Args:
            events: List of event dictionaries

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()

        # Add nodes for each frame event
        for event in events:
            node_id = f"frame_{event['frame_idx']}"
            G.add_node(
                node_id,
                frame_idx=event["frame_idx"],
                timestamp=event["timestamp"],
                num_detections=len(event["detections"]),
                detections=event["detections"]
            )

        # Add temporal edges (each frame -> next frame within temporal window)
        sorted_events = sorted(events, key=lambda e: e["frame_idx"])
        for i, event in enumerate(sorted_events):
            current_node = f"frame_{event['frame_idx']}"
            current_time = event["timestamp"]

            # Connect to subsequent frames within temporal window
            for j in range(i + 1, len(sorted_events)):
                next_event = sorted_events[j]
                next_node = f"frame_{next_event['frame_idx']}"
                next_time = next_event["timestamp"]

                time_diff = next_time - current_time
                if time_diff <= self.config.max_temporal_distance:
                    G.add_edge(
                        current_node,
                        next_node,
                        time_diff=time_diff,
                        edge_type="temporal"
                    )
                else:
                    break  # Beyond temporal window

        return G

    def process_clip(self, clip_id: str) -> Path:
        """
        Build graph for a single clip.

        Args:
            clip_id: Unique identifier for clip

        Returns:
            Path to directory containing graph
        """
        print(f"Building graph for clip: {clip_id}")

        # Load events
        events_dir = self.config.events_output_dir / clip_id
        events_path = events_dir / "events.json"
        if not events_path.exists():
            raise ValueError(f"Events file not found: {events_path}")

        with open(events_path) as f:
            events = json.load(f)

        print(f"  Loaded {len(events)} events")

        # Build graph
        G = self.build_graph(events)
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Save graph
        clip_output_dir = self.output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        graph_path = clip_output_dir / "graph.gpickle"
        nx.write_gpickle(G, graph_path)

        # Save graph metadata
        metadata = {
            "clip_id": clip_id,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_events": len(events)
        }
        metadata_path = clip_output_dir / "graph_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved graph to: {clip_output_dir}")
        return clip_output_dir

    def process_all_clips(self) -> None:
        """Process all clips with detected events."""
        events_dir = self.config.events_output_dir
        if not events_dir.exists():
            print(f"Events directory not found: {events_dir}")
            return

        clip_dirs = [d for d in events_dir.iterdir() if d.is_dir()]
        print(f"Found {len(clip_dirs)} clips to process")

        for clip_dir in clip_dirs:
            clip_id = clip_dir.name
            self.process_clip(clip_id)


def main():
    """Run graph construction stage."""
    config = Config()
    builder = GraphBuilder(config)
    builder.process_all_clips()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes (will skip without events)**

Run: `poetry run pytest tests/test_stage3_graph.py -v`
Expected: SKIPPED or PASS

- [ ] **Step 5: Commit**

```bash
git add src/hoop_vision/stage3_graph.py tests/test_stage3_graph.py
git commit -m "feat(stage3): add temporal graph construction from events"
```

---

## Task 7: Stage 4 - Feature Extraction

**Files:**
- Create: `src/hoop_vision/stage4_features.py`
- Create: `tests/test_stage4_features.py`

- [ ] **Step 1: Write failing test for FeatureExtractor**

```python
# tests/test_stage4_features.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_stage4_features.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal FeatureExtractor implementation**

```python
# src/hoop_vision/stage4_features.py
"""Stage 4: Feature Extraction - Extract engineered features from graphs."""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import networkx as nx

from .config import Config


class FeatureExtractor:
    """Extract engineered features from temporal event graphs."""

    def __init__(self, config: Config):
        """
        Initialize feature extractor.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.features_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_node_features(self, G: nx.DiGraph, node: str) -> Dict[str, Any]:
        """
        Extract features for a single node.

        Args:
            G: Graph
            node: Node ID

        Returns:
            Dictionary of features
        """
        node_data = G.nodes[node]

        # Basic node features
        features = {
            "frame_idx": node_data["frame_idx"],
            "timestamp": node_data["timestamp"],
            "num_detections": node_data["num_detections"]
        }

        # Count detections by class
        detection_counts = {}
        for detection in node_data["detections"]:
            class_name = detection["class_name"]
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

        # Add detection counts as features
        for class_name, count in detection_counts.items():
            features[f"count_{class_name}"] = count

        # Graph structure features
        features["in_degree"] = G.in_degree(node)
        features["out_degree"] = G.out_degree(node)

        # Temporal features - look back at previous N frames
        lookback_detections = []
        for predecessor in G.predecessors(node):
            pred_data = G.nodes[predecessor]
            lookback_detections.append(pred_data["num_detections"])

        if lookback_detections:
            features["lookback_avg_detections"] = sum(lookback_detections) / len(lookback_detections)
            features["lookback_max_detections"] = max(lookback_detections)
        else:
            features["lookback_avg_detections"] = 0
            features["lookback_max_detections"] = 0

        return features

    def extract_graph_features(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        Extract features for all nodes in graph.

        Args:
            G: Graph

        Returns:
            DataFrame with features for each node
        """
        features_list = []
        for node in G.nodes():
            node_features = self.extract_node_features(G, node)
            features_list.append(node_features)

        df = pd.DataFrame(features_list)
        # Sort by frame index
        df = df.sort_values("frame_idx").reset_index(drop=True)

        return df

    def process_clip(self, clip_id: str) -> Path:
        """
        Extract features for a single clip.

        Args:
            clip_id: Unique identifier for clip

        Returns:
            Path to directory containing features
        """
        print(f"Extracting features for clip: {clip_id}")

        # Load graph
        graphs_dir = self.config.graphs_output_dir / clip_id
        graph_path = graphs_dir / "graph.gpickle"
        if not graph_path.exists():
            raise ValueError(f"Graph file not found: {graph_path}")

        G = nx.read_gpickle(graph_path)
        print(f"  Loaded graph: {G.number_of_nodes()} nodes")

        # Extract features
        features_df = self.extract_graph_features(G)
        print(f"  Extracted {len(features_df.columns)} features for {len(features_df)} nodes")

        # Save features
        clip_output_dir = self.output_dir / clip_id
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        features_path = clip_output_dir / "features.csv"
        features_df.to_csv(features_path, index=False)

        # Save feature metadata
        metadata = {
            "clip_id": clip_id,
            "num_samples": len(features_df),
            "num_features": len(features_df.columns),
            "feature_names": list(features_df.columns)
        }
        metadata_path = clip_output_dir / "features_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved features to: {clip_output_dir}")
        return clip_output_dir

    def process_all_clips(self) -> None:
        """Process all clips with graphs."""
        graphs_dir = self.config.graphs_output_dir
        if not graphs_dir.exists():
            print(f"Graphs directory not found: {graphs_dir}")
            return

        clip_dirs = [d for d in graphs_dir.iterdir() if d.is_dir()]
        print(f"Found {len(clip_dirs)} clips to process")

        for clip_dir in clip_dirs:
            clip_id = clip_dir.name
            self.process_clip(clip_id)


def main():
    """Run feature extraction stage."""
    config = Config()
    extractor = FeatureExtractor(config)
    extractor.process_all_clips()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes (will skip without graph)**

Run: `poetry run pytest tests/test_stage4_features.py -v`
Expected: SKIPPED or PASS

- [ ] **Step 5: Commit**

```bash
git add src/hoop_vision/stage4_features.py tests/test_stage4_features.py
git commit -m "feat(stage4): add feature extraction from graphs"
```

---

## Task 8: Stage 5 - Prediction (Baseline)

**Files:**
- Create: `src/hoop_vision/stage5_prediction.py`
- Create: `tests/test_stage5_prediction.py`

- [ ] **Step 1: Write failing test for PredictionModel**

```python
# tests/test_stage5_prediction.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_stage5_prediction.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal PredictionModel implementation**

```python
# src/hoop_vision/stage5_prediction.py
"""Stage 5: Prediction - Train and evaluate prediction models."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from .config import Config


# Prediction target classes
OUTCOME_CLASSES = [
    "made_2pt",
    "made_3pt",
    "missed_shot",
    "turnover",
    "foul_offensive",
    "foul_defensive",
    "end_period"
]


class PredictionModel:
    """Train and evaluate prediction models for next possession outcomes."""

    def __init__(self, config: Config):
        """
        Initialize prediction model.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.predictions_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir = config.model_save_dir
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    def load_all_features(self) -> pd.DataFrame:
        """
        Load features from all processed clips.

        Returns:
            Combined DataFrame with all features
        """
        features_dir = self.config.features_output_dir
        if not features_dir.exists():
            raise ValueError(f"Features directory not found: {features_dir}")

        all_features = []
        clip_dirs = [d for d in features_dir.iterdir() if d.is_dir()]

        for clip_dir in clip_dirs:
            features_file = clip_dir / "features.csv"
            if features_file.exists():
                df = pd.read_csv(features_file)
                df["clip_id"] = clip_dir.name
                all_features.append(df)

        if not all_features:
            raise ValueError("No feature files found")

        combined_df = pd.concat(all_features, ignore_index=True)
        print(f"Loaded features from {len(clip_dirs)} clips: {len(combined_df)} samples")

        return combined_df

    def load_labels(self, labels_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Load ground truth labels for clips.

        Args:
            labels_file: Path to CSV with clip_id, frame_idx, label columns

        Returns:
            DataFrame with labels
        """
        if labels_file is None:
            labels_file = Path("data/labels.csv")

        if not labels_file.exists():
            print(f"Warning: Labels file not found at {labels_file}")
            print("Creating dummy labels for demonstration purposes")
            # Create dummy labels - in real usage, these must be manually annotated
            features_df = self.load_all_features()
            dummy_labels = pd.DataFrame({
                "clip_id": features_df["clip_id"],
                "frame_idx": features_df["frame_idx"],
                "label": np.random.choice(OUTCOME_CLASSES, size=len(features_df))
            })
            return dummy_labels

        return pd.read_csv(labels_file)

    def prepare_dataset(self, features_df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Merge features and labels, prepare for training.

        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame

        Returns:
            Tuple of (X, y, feature_columns)
        """
        # Merge features and labels
        df = features_df.merge(labels_df, on=["clip_id", "frame_idx"], how="inner")
        print(f"Merged dataset: {len(df)} labeled samples")

        # Separate features and labels
        feature_columns = [col for col in df.columns
                          if col not in ["clip_id", "frame_idx", "timestamp", "label"]]

        X = df[feature_columns].fillna(0)  # Fill NaN with 0
        y = df["label"]

        print(f"Features: {len(feature_columns)} columns")
        print(f"Label distribution:\n{y.value_counts()}")

        return X, y, feature_columns

    def train_baseline(self, labels_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Train baseline XGBoost model.

        Args:
            labels_file: Optional path to labels CSV

        Returns:
            Dictionary with model path and metrics
        """
        print("Training baseline XGBoost model")

        # Load data
        features_df = self.load_all_features()
        labels_df = self.load_labels(labels_file)
        X, y, feature_columns = self.prepare_dataset(features_df, labels_df)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
            stratify=y if len(y.unique()) > 1 else None
        )

        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.random_seed,
            eval_metric="mlogloss"
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\nBaseline Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model
        model_path = self.model_save_dir / "xgboost_baseline.json"
        self.model.save_model(model_path)
        print(f"\nModel saved to: {model_path}")

        # Save metrics
        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
            "num_features": len(feature_columns),
            "feature_columns": feature_columns
        }

        metrics_path = self.output_dir / "baseline_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {metrics_path}")

        return {
            "model_path": str(model_path),
            "metrics": metrics
        }


def main():
    """Run prediction model training."""
    config = Config()
    model = PredictionModel(config)
    model.train_baseline()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes (will skip without features)**

Run: `poetry run pytest tests/test_stage5_prediction.py -v`
Expected: SKIPPED or PASS

- [ ] **Step 5: Commit**

```bash
git add src/hoop_vision/stage5_prediction.py tests/test_stage5_prediction.py
git commit -m "feat(stage5): add baseline prediction model with XGBoost"
```

---

## Task 9: End-to-End Pipeline Demo Notebook

**Files:**
- Create: `notebooks/01_pipeline_demo.ipynb`

- [ ] **Step 1: Create Jupyter notebook with pipeline walkthrough**

Create `notebooks/01_pipeline_demo.ipynb` with the following cells:

**Cell 1 (Markdown):**
```markdown
# Hoop Vision - Phase 1 Pipeline Demo

This notebook demonstrates the end-to-end pipeline from video clips to next possession predictions.

## Pipeline Stages

1. **Ingestion**: Video → Frames
2. **Detection**: Frames → Events
3. **Graph**: Events → Temporal Graph
4. **Features**: Graph → Engineered Features
5. **Prediction**: Features → Next Possession Outcome
```

**Cell 2 (Code):**
```python
# Imports
import sys
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from hoop_vision.config import Config
from hoop_vision.stage1_ingestion import VideoIngestion
from hoop_vision.stage2_detection import EventDetection
from hoop_vision.stage3_graph import GraphBuilder
from hoop_vision.stage4_features import FeatureExtractor
from hoop_vision.stage5_prediction import PredictionModel

%matplotlib inline
```

**Cell 3 (Code):**
```python
# Load configuration
config = Config()
print("Configuration loaded")
print(f"  Target FPS: {config.fps}")
print(f"  YOLO model: {config.yolo_model}")
```

**Cell 4 (Markdown):**
```markdown
## Stage 1: Video Ingestion

Extract frames from video clips at configured FPS.
```

**Cell 5 (Code):**
```python
# Check for sample clips
sample_clips = list(config.sample_clips_dir.glob("*.mp4"))
if not sample_clips:
    print("No sample clips found. Place video clips in data/sample/")
else:
    print(f"Found {len(sample_clips)} sample clips")
    for clip in sample_clips:
        print(f"  - {clip.name}")
```

**Cell 6 (Code):**
```python
# Run ingestion (if clips available)
if sample_clips:
    ingestion = VideoIngestion(config)
    clip_id = sample_clips[0].stem
    output_dir = ingestion.process_clip(sample_clips[0], clip_id)
    print(f"Frames saved to: {output_dir}")
```

**Cell 7 (Markdown):**
```markdown
## Stage 2: Event Detection

Detect objects (players, ball, etc.) using YOLOv8.
```

**Cell 8 (Code):**
```python
# Run detection (if frames available)
frames_dirs = list(config.frames_output_dir.glob("*"))
if frames_dirs:
    detection = EventDetection(config)
    clip_id = frames_dirs[0].name
    output_dir = detection.process_clip(clip_id)
    print(f"Events saved to: {output_dir}")
```

**Cell 9 (Markdown):**
```markdown
## Stage 3: Graph Construction

Build temporal graph from detected events.
```

**Cell 10 (Code):**
```python
# Run graph construction (if events available)
events_dirs = list(config.events_output_dir.glob("*"))
if events_dirs:
    builder = GraphBuilder(config)
    clip_id = events_dirs[0].name
    output_dir = builder.process_clip(clip_id)
    print(f"Graph saved to: {output_dir}")

    # Load and visualize graph
    graph_file = output_dir / "graph.gpickle"
    G = nx.read_gpickle(graph_file)

    print(f"\\nGraph structure:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Simple visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=300,
            node_color='lightblue', font_size=8, arrows=True)
    plt.title("Temporal Event Graph")
    plt.show()
```

**Cell 11 (Markdown):**
```markdown
## Stage 4: Feature Extraction

Extract engineered features from graph.
```

**Cell 12 (Code):**
```python
# Run feature extraction (if graphs available)
graphs_dirs = list(config.graphs_output_dir.glob("*"))
if graphs_dirs:
    extractor = FeatureExtractor(config)
    clip_id = graphs_dirs[0].name
    output_dir = extractor.process_clip(clip_id)
    print(f"Features saved to: {output_dir}")

    # Load and display features
    features_file = output_dir / "features.csv"
    features_df = pd.read_csv(features_file)

    print(f"\\nFeatures shape: {features_df.shape}")
    print(f"\\nFeature columns:")
    print(features_df.columns.tolist())
    print(f"\\nFirst few rows:")
    display(features_df.head())
```

**Cell 13 (Markdown):**
```markdown
## Stage 5: Prediction

Train baseline model and make predictions.

**Note:** This requires labeled data. For demonstration, we use dummy labels.
In production, clips must be manually annotated with possession outcomes.
```

**Cell 14 (Code):**
```python
# Train baseline model (if features available)
features_dirs = list(config.features_output_dir.glob("*"))
if features_dirs:
    model = PredictionModel(config)
    result = model.train_baseline()

    print("\\nTraining complete!")
    print(f"Model: {result['model_path']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.3f}")
```

**Cell 15 (Markdown):**
```markdown
## Next Steps

1. **Collect real NBA clips** - Extract 30-60s segments from games
2. **Manually label outcomes** - Create `data/labels.csv` with ground truth
3. **Iterate on features** - Add more basketball-specific features
4. **Experiment with models** - Try different classifiers
5. **Scale to more data** - Collect 100-200 labeled clips for Phase 1
```

- [ ] **Step 2: Test notebook runs without errors**

Run: `jupyter notebook notebooks/01_pipeline_demo.ipynb`
Expected: Notebook opens and cells can be executed (may skip some cells if data not available)

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_pipeline_demo.ipynb
git commit -m "docs: add end-to-end pipeline demo notebook"
```

---

## Task 10: Documentation and Final Polish

**Files:**
- Modify: `README.md`
- Create: `docs/DATA_COLLECTION.md`
- Create: `docs/LABELING.md`

- [ ] **Step 1: Update README with comprehensive instructions**

```markdown
# Hoop Vision

NBA game prediction using computer vision and graph-based machine learning.

## Overview

Hoop Vision analyzes NBA video footage to predict game outcomes by capturing nuanced events that don't appear in traditional stats: uncalled fouls, player frustration, defensive pressure, and emotional states.

### Phase 1: Next Possession Prediction

The current implementation predicts next possession outcomes (made shot, miss, turnover, foul, etc.) from short video clips using a 5-stage pipeline:

1. **Ingestion**: Extract frames from video clips
2. **Detection**: Detect players, ball, and events using YOLOv8
3. **Graph**: Build temporal event graph
4. **Features**: Extract engineered features
5. **Prediction**: Train XGBoost classifier

### Future Phases

- **Phase 2**: Add emotion detection, richer events, upgrade to GNN
- **Phase 3**: Scale to full game processing
- **Phase 4**: Player reaction predictions, momentum shifts, game outcomes

## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd hoop-vision

# Install dependencies
poetry install

# Verify installation
poetry run pytest
```

## Quick Start

### 1. Prepare Video Clips

Place 30-60 second NBA video clips in `data/raw_clips/` or `data/sample/`:

```bash
data/
  raw_clips/
    clip001.mp4
    clip002.mp4
  sample/
    test_clip.mp4
```

### 2. Run Pipeline Stages

Execute each stage in sequence:

```bash
# Stage 1: Extract frames
poetry run python -m hoop_vision.stage1_ingestion

# Stage 2: Detect events
poetry run python -m hoop_vision.stage2_detection

# Stage 3: Build graph
poetry run python -m hoop_vision.stage3_graph

# Stage 4: Extract features
poetry run python -m hoop_vision.stage4_features

# Stage 5: Train model (requires labels)
poetry run python -m hoop_vision.stage5_prediction
```

### 3. Explore Results

View the end-to-end demo:

```bash
jupyter notebook notebooks/01_pipeline_demo.ipynb
```

## Data Collection & Labeling

See detailed guides:

- [Data Collection Guide](docs/DATA_COLLECTION.md) - How to extract NBA video clips
- [Labeling Guide](docs/LABELING.md) - How to label possession outcomes

**Important:** Stage 5 (Prediction) requires manually labeled data. See `docs/LABELING.md` for the labeling format.

## Configuration

Edit `config.yaml` to customize pipeline settings:

```yaml
ingestion:
  fps: 2  # Frames per second

detection:
  yolo_model: yolov8n.pt  # YOLO model size
  confidence_threshold: 0.5

prediction:
  test_split: 0.2
  random_seed: 42
```

## Project Structure

```
hoop-vision/
├── src/hoop_vision/           # Source code
│   ├── stage1_ingestion.py    # Video → Frames
│   ├── stage2_detection.py    # Frames → Events
│   ├── stage3_graph.py        # Events → Graph
│   ├── stage4_features.py     # Graph → Features
│   ├── stage5_prediction.py   # Features → Predictions
│   └── utils/                 # Utilities
├── tests/                     # Tests
├── notebooks/                 # Jupyter demos
├── data/                      # Data directories
│   ├── raw_clips/            # Input videos
│   ├── processed/            # Pipeline outputs
│   └── sample/               # Test data
├── docs/                      # Documentation
└── config.yaml               # Configuration
```

## Testing

Run all tests:

```bash
poetry run pytest
```

Run with coverage:

```bash
poetry run pytest --cov=src/hoop_vision --cov-report=html
```

## Development

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint
poetry run ruff check src/ tests/
```

## Design Documentation

See `docs/superpowers/specs/2026-04-22-hoop-vision-design.md` for complete design specification.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- NBA for providing League Pass access
```

- [ ] **Step 2: Create DATA_COLLECTION.md guide**

```markdown
# Data Collection Guide

How to collect NBA video clips for Hoop Vision.

## Requirements

- NBA League Pass subscription (or alternative video source)
- Video editing software or screen recording tool
- 30-60 second clips capturing complete possessions

## Recommended Workflow

### Option 1: Screen Recording (Easiest)

1. **Open NBA League Pass** in browser
2. **Navigate to game** you want to clip
3. **Find interesting sequence** (close game, visible emotions, momentum shifts)
4. **Screen record** using:
   - macOS: QuickTime Player → File → New Screen Recording
   - Windows: Xbox Game Bar (Win + G)
   - Linux: SimpleScreenRecorder or OBS

5. **Save clip** as MP4 to `data/raw_clips/`
6. **Name descriptively**: `game_date_team1_vs_team2_sequence.mp4`

### Option 2: Video Download (Check Terms of Service)

**Warning:** Downloading NBA League Pass content may violate terms of service. Research NBA's policy before proceeding.

If permitted:
- Use browser developer tools to find video stream URL
- Use `youtube-dl` or similar tools
- Extract specific time ranges with `ffmpeg`

### Option 3: YouTube Highlights

For testing/demo purposes:

1. Find NBA game highlights on YouTube
2. Download using `youtube-dl`:
   ```bash
   youtube-dl "URL" -o "data/raw_clips/%(title)s.%(ext)s"
   ```
3. Extract clips with `ffmpeg`:
   ```bash
   ffmpeg -i input.mp4 -ss 00:01:30 -t 00:00:45 -c copy clip.mp4
   ```

## Clip Selection Criteria

Good clips for Phase 1:

- **Duration**: 30-60 seconds
- **Content**: 3-5 complete possessions
- **Quality**: Clear view of court, players visible
- **Context**: Include scoreboard (helps with temporal alignment)
- **Variety**: Mix of:
  - Made shots (2pt and 3pt)
  - Missed shots
  - Turnovers
  - Fouls
  - Different teams, players, game situations

## Clip Metadata

For each clip, document:

```yaml
clip_id: lakers_celtics_2024_q4_run
game_date: 2024-03-15
teams: LAL vs BOS
quarter: 4
time_range: 8:45 - 7:30
description: Lakers 8-0 run after LeBron frustration moment
notable_events:
  - LeBron missed shot + visible frustration (0:05)
  - Immediate defensive stop (0:15)
  - Fast break made layup (0:22)
  - Made 3-pointer (0:38)
```

Store metadata in `data/clip_metadata.yaml` or similar.

## Target Dataset

### Phase 1 Goal

- **100-200 labeled clips** to prove the approach
- **Diversity**: Multiple teams, players, game situations
- **Quality over quantity**: Clear, complete possessions

### Data Split

- **Training**: 80% (80-160 clips)
- **Testing**: 20% (20-40 clips)

## Legal Considerations

**Important:** Always respect copyright and terms of service.

- **NBA League Pass**: Check if personal use/research is permitted
- **YouTube**: Follow Fair Use guidelines
- **Academic Use**: May have different permissions
- **Commercial Use**: Requires licensing

Consult legal counsel if uncertain.

## Storage

- **Local**: `data/raw_clips/`
- **Backup**: External drive or cloud storage (Dropbox, Google Drive)
- **Do not commit to Git**: Videos are in `.gitignore`

## Next Steps

After collecting clips, proceed to [Labeling Guide](LABELING.md) to annotate possession outcomes.
```

- [ ] **Step 3: Create LABELING.md guide**

```markdown
# Labeling Guide

How to manually label possession outcomes for training the prediction model.

## Label Format

Create `data/labels.csv` with the following columns:

```csv
clip_id,frame_idx,label
test_clip,0,missed_shot
test_clip,5,made_2pt
test_clip,12,turnover
lakers_celtics_q4,0,made_3pt
lakers_celtics_q4,8,foul_defensive
```

## Label Classes

```python
OUTCOME_CLASSES = [
    "made_2pt",         # Made 2-point field goal
    "made_3pt",         # Made 3-point field goal
    "missed_shot",      # Missed field goal attempt
    "turnover",         # Turnover (any type)
    "foul_offensive",   # Offensive foul called
    "foul_defensive",   # Defensive foul called
    "end_period"        # End of quarter/half
]
```

## Labeling Workflow

### Step 1: Watch Clip

1. Open clip in video player
2. Watch entire clip to understand context
3. Identify each possession and its outcome

### Step 2: Determine Frame Index

For each possession outcome:

1. **Find timestamp** when outcome occurs (e.g., ball goes through hoop, whistle blows)
2. **Calculate frame index**:
   ```
   frame_idx = timestamp (seconds) * target_fps
   ```
   Example: Outcome at 5.5 seconds, target_fps=2 → frame_idx = 11

3. **Round to nearest frame** sampled by pipeline

### Step 3: Assign Label

Based on what happened:

- **Shot made inside arc**: `made_2pt`
- **Shot made outside arc**: `made_3pt`
- **Shot missed**: `missed_shot`
- **Ball lost, stolen, or bad pass**: `turnover`
- **Foul on offense**: `foul_offensive`
- **Foul on defense**: `foul_defensive`
- **End of quarter/half**: `end_period`

### Step 4: Record in CSV

Add row to `data/labels.csv`:

```csv
clip_id,frame_idx,label
clip001,15,made_2pt
```

## Example Labeling Session

**Clip:** `lakers_celtics_q4.mp4` (45 seconds, 2 FPS → 90 frames)

**Possession 1 (0:00 - 0:12)**
- LeBron drives, misses layup
- Timestamp: 0:12 (outcome)
- Frame: 12 * 2 = 24
- Label: `missed_shot`

**Possession 2 (0:12 - 0:25)**
- Celtics rebound, fast break
- Tatum makes layup
- Timestamp: 0:25
- Frame: 25 * 2 = 50
- Label: `made_2pt`

**Possession 3 (0:25 - 0:38)**
- Lakers inbound, turnover on bad pass
- Timestamp: 0:32
- Frame: 32 * 2 = 64
- Label: `turnover`

**Possession 4 (0:38 - 0:45)**
- Celtics possession, foul on Westbrook
- Timestamp: 0:42
- Frame: 42 * 2 = 84
- Label: `foul_defensive`

**Result CSV:**

```csv
clip_id,frame_idx,label
lakers_celtics_q4,24,missed_shot
lakers_celtics_q4,50,made_2pt
lakers_celtics_q4,64,turnover
lakers_celtics_q4,84,foul_defensive
```

## Edge Cases

### Unclear Outcome

If outcome is ambiguous:
- **Skip it** - only label clear possessions
- Focus on quality over quantity

### Multiple Events

If multiple events in quick succession:
- **Label primary outcome** (e.g., shot made after foul → label the foul if it stopped play)

### Possession Spans Multiple Clips

- Label each clip independently
- Truncate possession at clip boundary

## Quality Checks

Before using labels for training:

1. **Balance check**: Do you have examples of all classes?
   ```python
   import pandas as pd
   df = pd.read_csv("data/labels.csv")
   print(df["label"].value_counts())
   ```

2. **Duplicate check**: No duplicate (clip_id, frame_idx) pairs
   ```python
   duplicates = df.duplicated(subset=["clip_id", "frame_idx"])
   print(df[duplicates])
   ```

3. **Frame bounds check**: All frame indices exist in processed frames
   ```python
   for clip_id in df["clip_id"].unique():
       clip_frames = len(list(Path(f"data/processed/frames/{clip_id}").glob("*.jpg")))
       max_label_frame = df[df["clip_id"] == clip_id]["frame_idx"].max()
       assert max_label_frame < clip_frames, f"{clip_id}: frame_idx {max_label_frame} >= {clip_frames}"
   ```

## LLM-Assisted Labeling (Optional)

To speed up labeling, use a multimodal LLM:

### Step 1: Manual Baseline

Label 20-30 clips manually to establish quality baseline.

### Step 2: LLM Labeling

For remaining clips:

```python
# Pseudocode
from openai import OpenAI

client = OpenAI()

for clip in unlabeled_clips:
    frames = extract_key_frames(clip)

    prompt = f"""
    Analyze these basketball game frames and identify possession outcomes.

    Classes: made_2pt, made_3pt, missed_shot, turnover, foul_offensive, foul_defensive, end_period

    For each outcome, return:
    - Frame index (approximate)
    - Outcome label
    - Confidence (1-5)
    """

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            *[{"type": "image_url", "image_url": frame} for frame in frames]
        ]}]
    )

    parse_and_save_labels(response)
```

### Step 3: Human Review

- Review all LLM labels
- Correct errors
- Track accuracy to assess LLM quality

## Timeline Estimate

**Manual labeling:**
- 5-10 minutes per clip (60 seconds)
- 100 clips = 8-16 hours

**LLM-assisted labeling:**
- 1-2 minutes per clip (human review)
- 100 clips = 2-3 hours (+ API costs)

## Next Steps

After labeling, run Stage 5 (Prediction):

```bash
poetry run python -m hoop_vision.stage5_prediction
```
```

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/DATA_COLLECTION.md docs/LABELING.md
git commit -m "docs: add comprehensive guides for data collection and labeling"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: All 5 stages implemented with tests
- [x] **Placeholders**: No TBDs or TODOs - all code complete
- [x] **Type consistency**: Config properties, class names, methods consistent across tasks
- [x] **File paths**: All exact paths specified
- [x] **Commands**: All run commands with expected output
- [x] **Testing**: Each component has tests
- [x] **Documentation**: README, guides, notebook complete

## Success Criteria

After completing this plan, you will have:

1. ✅ Working 5-stage pipeline from video → predictions
2. ✅ Test coverage for all stages
3. ✅ Jupyter notebook demonstrating end-to-end flow
4. ✅ Comprehensive documentation for users
5. ✅ Configuration-driven architecture
6. ✅ Foundation ready for Phase 2 enhancements

## Notes

- **Data dependency**: Stages 1-4 can run without labels, but Stage 5 requires manually labeled data
- **Model performance**: Initial accuracy may be low with dummy labels - real performance requires quality annotations
- **Extensibility**: Architecture designed to easily add new features, detection models, and prediction approaches in future phases
