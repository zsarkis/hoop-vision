# src/hoop_vision/stage1_ingestion.py
"""Stage 1: Video Ingestion - Extract frames from NBA video clips."""

import json
import cv2
from pathlib import Path
from typing import Dict, List
from hoop_vision.config import Config
from hoop_vision.utils.video_utils import extract_frames, get_video_info


class VideoIngestion:
    """Extracts frames from NBA video clips and saves them with metadata."""

    def __init__(self, config: Config):
        """
        Initialize VideoIngestion.

        Args:
            config: Pipeline configuration containing FPS and output directory settings
        """
        self.config = config
        self.output_dir = config.frames_output_dir
        self.target_fps = config.fps

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_clip(self, video_path: Path, clip_id: str) -> Dict:
        """
        Process a single video clip: extract frames and save metadata.

        Args:
            video_path: Path to the video file
            clip_id: Unique identifier for this clip

        Returns:
            Dictionary containing metadata about the processed clip
        """
        # Create output directory for this clip
        clip_dir = self.output_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Get video metadata
        video_info = get_video_info(video_path)

        # Extract frames at target FPS
        frames = extract_frames(video_path, target_fps=self.target_fps)

        # Save frames to disk
        for i, frame in enumerate(frames):
            frame_filename = f"frame_{i:04d}.jpg"
            frame_path = clip_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

        # Create metadata
        metadata = {
            "clip_id": clip_id,
            "source_video": str(video_path),
            "source_fps": video_info["fps"],
            "source_duration": video_info["duration"],
            "target_fps": self.target_fps,
            "num_frames": len(frames)
        }

        # Save metadata to JSON
        metadata_path = clip_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def process_all_clips(self, clips_dir: Path) -> List[Dict]:
        """
        Process all video clips in a directory.

        Args:
            clips_dir: Directory containing video clips (.mp4 or .mov files)

        Returns:
            List of metadata dictionaries for all processed clips
        """
        clips_dir = Path(clips_dir)
        if not clips_dir.exists():
            raise ValueError(f"Clips directory does not exist: {clips_dir}")

        # Find all video files
        video_extensions = ["*.mp4", "*.mov"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(clips_dir.glob(ext))

        # Process each video
        all_metadata = []
        for video_path in sorted(video_files):
            # Use filename (without extension) as clip_id
            clip_id = video_path.stem
            print(f"Processing clip: {clip_id} ({video_path.name})")

            metadata = self.process_clip(video_path, clip_id)
            all_metadata.append(metadata)

            print(f"  -> Extracted {metadata['num_frames']} frames")

        return all_metadata


def main():
    """CLI entry point for video ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1: Extract frames from NBA video clips"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--clips-dir",
        type=str,
        help="Directory containing video clips to process (default: from config)"
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        help="Process a single clip with this ID (requires --video-path)"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to a single video file to process (requires --clip-id)"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)
    ingestion = VideoIngestion(config)

    # Process single clip or all clips
    if args.clip_id and args.video_path:
        # Process single clip
        video_path = Path(args.video_path)
        metadata = ingestion.process_clip(video_path, args.clip_id)
        print(f"\nProcessed clip: {args.clip_id}")
        print(f"Extracted {metadata['num_frames']} frames")
        print(f"Output directory: {config.frames_output_dir / args.clip_id}")

    else:
        # Process all clips in directory
        clips_dir = Path(args.clips_dir) if args.clips_dir else config.raw_clips_dir
        print(f"Processing all clips in: {clips_dir}")

        all_metadata = ingestion.process_all_clips(clips_dir)

        print(f"\n=== Summary ===")
        print(f"Processed {len(all_metadata)} clips")
        print(f"Total frames: {sum(m['num_frames'] for m in all_metadata)}")
        print(f"Output directory: {config.frames_output_dir}")


if __name__ == "__main__":
    main()
