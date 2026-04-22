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
