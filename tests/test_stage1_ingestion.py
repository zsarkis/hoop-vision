# tests/test_stage1_ingestion.py
import pytest
import json
from pathlib import Path
from hoop_vision.config import Config
from hoop_vision.stage1_ingestion import VideoIngestion


def test_video_ingestion_processes_clip(tmp_path):
    """Test that VideoIngestion processes a video clip and extracts frames."""
    sample_video = Path("data/sample/test_clip.mp4")
    if not sample_video.exists():
        pytest.skip("Sample video not available")

    # Create a temporary config
    config = Config()
    config.frames_output_dir = tmp_path / "frames"
    config.fps = 2.0

    # Initialize VideoIngestion
    ingestion = VideoIngestion(config)

    # Process a clip
    clip_id = "test_clip_001"
    ingestion.process_clip(sample_video, clip_id)

    # Check that output directory was created
    clip_dir = config.frames_output_dir / clip_id
    assert clip_dir.exists()

    # Check that frames were saved
    frame_files = list(clip_dir.glob("frame_*.jpg"))
    assert len(frame_files) > 0

    # Check frame naming convention
    for frame_file in frame_files:
        assert frame_file.name.startswith("frame_")
        assert frame_file.name.endswith(".jpg")

    # Check that metadata file exists
    metadata_file = clip_dir / "metadata.json"
    assert metadata_file.exists()


def test_video_ingestion_saves_metadata(tmp_path):
    """Test that VideoIngestion saves correct metadata for processed clips."""
    sample_video = Path("data/sample/test_clip.mp4")
    if not sample_video.exists():
        pytest.skip("Sample video not available")

    # Create a temporary config
    config = Config()
    config.frames_output_dir = tmp_path / "frames"
    config.fps = 2.0

    # Initialize VideoIngestion
    ingestion = VideoIngestion(config)

    # Process a clip
    clip_id = "test_clip_002"
    ingestion.process_clip(sample_video, clip_id)

    # Load and verify metadata
    metadata_file = config.frames_output_dir / clip_id / "metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Check required metadata fields
    assert "clip_id" in metadata
    assert "source_video" in metadata
    assert "source_fps" in metadata
    assert "source_duration" in metadata
    assert "target_fps" in metadata
    assert "num_frames" in metadata

    # Verify metadata values
    assert metadata["clip_id"] == clip_id
    assert metadata["source_video"] == str(sample_video)
    assert metadata["target_fps"] == config.fps
    assert metadata["source_fps"] > 0
    assert metadata["source_duration"] > 0
    assert metadata["num_frames"] > 0

    # Verify num_frames matches actual frame count
    frame_files = list((config.frames_output_dir / clip_id).glob("frame_*.jpg"))
    assert metadata["num_frames"] == len(frame_files)
