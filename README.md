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
