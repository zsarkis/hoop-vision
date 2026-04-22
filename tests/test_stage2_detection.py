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
