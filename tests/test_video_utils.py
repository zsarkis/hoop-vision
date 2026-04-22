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
