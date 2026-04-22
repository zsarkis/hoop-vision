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
