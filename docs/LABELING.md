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
