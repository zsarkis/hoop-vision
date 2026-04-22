# Hoop Vision - Design Specification

**Date:** 2026-04-22
**Status:** Phase 1 - Initial Design

## Vision

Hoop Vision uses computer vision and graph-based machine learning to predict NBA game outcomes by analyzing video footage. Unlike traditional stats-based approaches, it captures nuanced events that don't appear in box scores: uncalled fouls, visible player frustration, defensive pressure, emotional states, and their downstream effects on performance.

### Full Scope (Long-term)

The complete system will support four types of predictions:

1. **Outcome predictions** - game winners, final scores
2. **In-game predictions** - next possession outcomes, momentum shifts, scoring runs
3. **Player performance** - how emotions/frustration affect subsequent plays
4. **Situational predictions** - how specific game situations will unfold

### Phase 1 Scope (Current Focus)

**Start with:** Next possession outcome prediction (made shot, missed shot, turnover, foul, etc.)

**Why:**
- Simpler classification problem with fast validation feedback
- Proves the pipeline works end-to-end
- Establishes foundation for more complex predictions later

**Design principle:** Build the infrastructure to support the full vision, but validate with the simplest prediction first.

## System Architecture

### Pipeline Overview

Five-stage pipeline where each stage can be developed, tested, and improved independently:

```
Video Clips → Event Detection → Graph Construction → Feature Extraction → Prediction
```

Each stage writes output to disk for inspection, debugging, and iteration without re-running upstream stages.

### Stage 1: Video Ingestion

**Input:** Manually clipped game segments (30-60 second clips around key moments)
**Output:** Sampled frames at configurable rate (start with 1-2 fps)
**Technology:** OpenCV for video handling and frame extraction

**Design notes:**
- Start small with clips to prove the approach before scaling to full games
- Configurable sampling rate allows balancing compute cost vs temporal resolution
- Clips should capture complete possessions (from inbound/rebound through shot/turnover)

**Data sourcing:** NBA League Pass access available, but need to validate legal/technical feasibility of downloading footage. If blocked, explore alternatives (YouTube highlights, publicly available game footage, NBA's official API for video if available).

### Stage 2: Event Detection

**Input:** Frames from Stage 1
**Output:** Detected events with structured data:
- Player positions (bounding boxes, IDs, coordinates)
- Ball location and possession
- Actions (shots, passes, defensive actions, fouls)
- Emotional signals (frustration, celebration, fatigue indicators)

**Technology:**
- Pre-trained object detection models (YOLO variants for players/ball)
- Pose estimation models (MediaPipe, OpenPose for body language)
- Emotion/expression detection (off-the-shelf models, may need fine-tuning)
- Player tracking across frames (DeepSORT or similar)

**Design notes:**
- Start with basic detection (players, ball, shots) and progressively add richer events
- Use pre-trained models where possible to minimize training data needs
- Event detection should be modular - easy to swap in better models later
- Each detected event includes: timestamp, bounding boxes, confidence scores, event type

**Example event output:**
```json
{
  "timestamp": 125.5,
  "type": "shot_attempt",
  "player_id": "lebron_james",
  "bbox": [120, 340, 180, 520],
  "confidence": 0.94,
  "made": false
}
```

### Stage 3: Graph Construction

**Input:** Events from Stage 2
**Output:** Temporal event graph stored as NetworkX or PyTorch Geometric object

**Graph structure:**

**Nodes represent:**
- Possession states (team possession, score differential, time remaining, quarter)
- Player events (shot attempts, defensive actions, emotional displays)
- Key moments (fouls - called and potentially uncalled, timeouts, substitutions)

**Edges represent:**
- Temporal sequence (directed edges showing what happened next)
- Causality (frustration → next possession, defensive pressure → turnover)
- Player relationships (defensive matchups, passing connections)

**Node attributes:**
- Timestamp
- Player ID(s) involved
- Spatial coordinates (court position)
- Detected emotions/body language
- Game context (score, time, foul count, timeout availability)

**Technology:**
- NetworkX for initial development (easy inspection/debugging)
- PyTorch Geometric for production (better integration with GNNs later)

**Design notes:**
- Graph grows temporally as game progresses
- Start with simple node types (shots, rebounds, possessions) and add complexity incrementally
- Structure allows adding new node types (emotions, defensive pressure) without architectural changes
- Historical context embedded in graph structure enables predictions based on recent game flow

**Example mini-graph:**
```
[Missed shot by Player A]
    → [Player A shows frustration]
    → [Defensive rebound by Player B]
    → [Fast break possession]
    → [PREDICT: next possession outcome?]
```

### Stage 4: Feature Extraction

**Input:** Raw graph from Stage 3
**Output:** Enriched graph with computed features

**Feature categories:**

**Momentum indicators:**
- Recent scoring run (last N possessions)
- Shot percentage over last M minutes
- Turnover rate trends

**Player state:**
- Detected emotional state (frustrated, confident, fatigued)
- Time since last substitution (fatigue proxy)
- Personal foul count

**Team context:**
- Score differential and trend
- Timeout availability
- Bonus/penalty situation

**Historical patterns:**
- Player/team shooting percentages (can integrate external APIs like NBA Stats API)
- Head-to-head historical performance
- Home/away splits

**Technology:**
- Custom Python code for feature computation
- Potentially integrate NBA Stats API for historical data enrichment
- Feature engineering guided by basketball domain knowledge

**Design notes:**
- Features should be interpretable (helps debug model predictions)
- Start with simple features, add complexity based on feature importance analysis
- Features become training data for classic ML models in Phase 1
- Later, these features can augment GNN learned representations (hybrid approach)

### Stage 5: Prediction

**Phase 1 approach:** Graph features + Classic ML

**Input:** Engineered features from Stage 4
**Output:** Predicted next possession outcome with confidence scores

**Prediction targets (multi-class classification):**
- Made field goal (2pt)
- Made field goal (3pt)
- Missed field goal
- Turnover
- Foul (offensive)
- Foul (defensive)
- End of quarter/half

**Models to experiment with:**
- XGBoost (likely strongest performer, good with tabular features)
- Random Forest (interpretable, shows feature importance)
- Simple feedforward neural network (baseline)

**Evaluation metrics:**
- Accuracy (overall correctness)
- Per-class precision/recall (some outcomes rarer than others)
- Confusion matrix (understand common misclassifications)
- Calibration (are confidence scores meaningful?)

**Technology:** scikit-learn, XGBoost, PyTorch for neural network baseline

**Phase 2 roadmap:** Upgrade to GNN or hybrid approach once:
- Pipeline proven with 100+ labeled clips
- Baseline performance established
- Feature importance analysis shows what matters
- More training data collected

**GNN benefits for Phase 2:**
- Learn which graph relationships matter (vs hand-engineering)
- Capture long-range dependencies (events several possessions back)
- Better support for player reaction predictions (temporal propagation through graph)

## Data Pipeline

### Data Collection

**Manual clip creation workflow:**
1. Watch NBA League Pass games (or alternative sources)
2. Identify interesting sequences (close games, visible emotions, momentum shifts)
3. Extract 30-60 second clips capturing complete possessions
4. Store clips with metadata: game ID, teams, players, quarter, score

**Labeling:**
- Manual annotation of possession outcomes (ground truth for training)
- Initially: just label the final outcome (made shot, miss, turnover, etc.)
- Later: annotate intermediate events (emotions, defensive pressure)

**LLM-assisted labeling workflow (optional):**
- Manually label 20-30 clips to establish patterns and validation set
- Use multimodal LLM (GPT-4V, Claude with vision) to label remaining clips:
  - Feed key frames from clip + game context to LLM
  - LLM predicts outcome label based on visual analysis
  - Human review and correction of LLM labels
- Benefits: faster dataset creation, less manual work
- Tradeoffs: LLM API costs, potential label noise, need human validation

**Target dataset for Phase 1:** 100-200 labeled clips to prove the approach

**Storage:**
- Raw clips: local filesystem or S3
- Processed frames: local cache
- Graphs and features: pickle files or lightweight database (SQLite)
- Model checkpoints: local filesystem with versioning

### Data Flow

```
Raw clips → Frame extraction → Event detection → Graph construction → Feature extraction
                                                                              ↓
                                                                    Train/test split
                                                                              ↓
                                                                    Model training
                                                                              ↓
                                                                    Evaluation
```

Each stage caches outputs to avoid re-computation. Config-driven pipeline allows experimenting with different sampling rates, detection models, features, etc.

## Technology Stack

### Core Libraries

**Video processing:** OpenCV
**Computer vision:**
- YOLOv8 or YOLOv10 (object detection)
- MediaPipe or OpenPose (pose estimation)
- Pre-trained emotion detection models (HuggingFace)

**Graph processing:**
- NetworkX (development/debugging)
- PyTorch Geometric (production, GNN-ready)

**Machine learning:**
- scikit-learn (classic ML models, preprocessing)
- XGBoost (primary Phase 1 model)
- PyTorch (neural networks, future GNN implementation)

**Data handling:**
- pandas (feature engineering, analysis)
- NumPy (numerical operations)

**Development:**
- Python 3.10+
- Poetry or pip-tools for dependency management
- Jupyter notebooks for exploration and visualization

### Infrastructure

**Phase 1:** Local development
- MacOS development environment
- Local GPU if available (for event detection), otherwise CPU
- Small-scale data (100-200 clips fits in memory/local disk)

**Future scaling considerations:**
- Cloud GPU instances for full game processing (AWS EC2 with GPU, GCP Compute)
- Batch processing pipeline for cost efficiency
- Consider serverless for inference if building a product

## Development Phases

### Phase 1: Prove the Pipeline (Current)

**Goals:**
- End-to-end pipeline working on manually clipped segments
- Basic event detection (players, ball, shots)
- Simple graph construction (temporal sequence of possessions)
- Feature extraction and baseline model training
- Validate prediction accuracy on next possession outcomes

**Success criteria:**
- Pipeline runs end-to-end on test clips
- Prediction accuracy better than random baseline
- Clear understanding of what features matter
- Documented learnings and bottlenecks

**Timeline estimate:** 4-8 weeks (depending on time investment)

### Phase 2: Enrich Events

**Goals:**
- Add emotion/frustration detection
- Improve player tracking and identification
- Expand graph with richer node types
- Experiment with GNN or hybrid models
- Collect more training data (500+ clips)

**Success criteria:**
- Emotion detection working reliably
- Model performance improves with richer features
- GNN baseline established

### Phase 3: Scale to Full Games

**Goals:**
- Process full game footage (solve data sourcing)
- Efficient frame sampling and event detection at scale
- Real-time or near-real-time prediction capability
- Cost optimization for cloud processing

**Success criteria:**
- Can process a full game end-to-end
- Prediction latency acceptable for use case
- Cost per game understood and acceptable

### Phase 4: Advanced Predictions

**Goals:**
- Player reaction tracking (frustration → performance effects)
- Momentum shift predictions
- Outcome predictions (game winners)
- Production-ready system

**Success criteria:**
- All four prediction types working
- Portfolio-ready demonstration
- Potential product viability assessed

## Open Questions & Risks

### Data Sourcing
**Question:** Can we legally/technically download NBA League Pass footage?
**Risk:** If not, need alternative sources (YouTube, official APIs, public datasets)
**Mitigation:** Research NBA's terms of service, explore NBA API offerings, consider partnerships or academic use exceptions

### Model Performance
**Question:** Will classic ML be accurate enough to be interesting?
**Risk:** Predictions might not beat simple baselines (recent shooting %, momentum indicators)
**Mitigation:** Focus on interpretability and learning what matters; even "negative results" are valuable for portfolio

### Compute Costs
**Question:** How expensive is event detection at scale?
**Risk:** Processing full games could be prohibitively expensive
**Mitigation:** Start with clips, optimize models, explore cheaper detection approaches, consider batch processing during off-peak hours

### Player Identification
**Question:** How do we reliably identify specific players from video?
**Risk:** Jersey numbers might not be visible, player recognition is hard
**Mitigation:** Start with team-level predictions (don't need individual IDs), explore jersey number OCR, consider external play-by-play data alignment

### Temporal Alignment
**Question:** How do we align detected events with game clock?
**Risk:** Video timestamps might not match game clock perfectly
**Mitigation:** OCR on scoreboard, align with play-by-play data if available, accept some noise in Phase 1

## Success Metrics

### Technical Metrics
- **Next possession accuracy:** Target >50% to start (random baseline ~14% with 7 classes, simple baseline ~40% using just recent shooting %)
- **Pipeline processing time:** <5 minutes per 60-second clip in Phase 1
- **Event detection recall:** >80% for basic events (shots, rebounds)

### Learning/Portfolio Metrics
- Working end-to-end demonstration with real NBA footage
- Clear explanation of approach and learnings
- Documented progression from simple to complex
- Deployable artifact (Jupyter notebook, web demo, or API)

### Product Metrics (if pursuing)
- Prediction accuracy competitive with betting lines
- Latency acceptable for in-game use
- Cost per prediction economically viable

## Next Steps

1. Set up project repository and development environment
2. Create initial data collection workflow (clip extraction from League Pass or alternatives)
3. Implement Stage 1 (video ingestion and frame extraction)
4. Implement Stage 2 (basic event detection: players, ball, shots)
5. Implement Stage 3 (simple temporal graph)
6. Implement Stage 4 (basic feature extraction)
7. Implement Stage 5 (baseline ML model)
8. Evaluate and iterate

Each stage should be tested independently before moving to the next. Prioritize getting something working end-to-end quickly over perfecting individual stages.
