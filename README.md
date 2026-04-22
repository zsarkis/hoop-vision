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
