# Data Collection Guide

How to collect NBA video clips for Hoop Vision.

## Requirements

- NBA League Pass subscription (or alternative video source)
- Video editing software or screen recording tool
- 30-60 second clips capturing complete possessions

## Recommended Workflow

### Option 1: Screen Recording (Easiest)

1. **Open NBA League Pass** in browser
2. **Navigate to game** you want to clip
3. **Find interesting sequence** (close game, visible emotions, momentum shifts)
4. **Screen record** using:
   - macOS: QuickTime Player → File → New Screen Recording
   - Windows: Xbox Game Bar (Win + G)
   - Linux: SimpleScreenRecorder or OBS

5. **Save clip** as MP4 to `data/raw_clips/`
6. **Name descriptively**: `game_date_team1_vs_team2_sequence.mp4`

### Option 2: Video Download (Check Terms of Service)

**Warning:** Downloading NBA League Pass content may violate terms of service. Research NBA's policy before proceeding.

If permitted:
- Use browser developer tools to find video stream URL
- Use `youtube-dl` or similar tools
- Extract specific time ranges with `ffmpeg`

### Option 3: YouTube Highlights

For testing/demo purposes:

1. Find NBA game highlights on YouTube
2. Download using `youtube-dl`:
   ```bash
   youtube-dl "URL" -o "data/raw_clips/%(title)s.%(ext)s"
   ```
3. Extract clips with `ffmpeg`:
   ```bash
   ffmpeg -i input.mp4 -ss 00:01:30 -t 00:00:45 -c copy clip.mp4
   ```

## Clip Selection Criteria

Good clips for Phase 1:

- **Duration**: 30-60 seconds
- **Content**: 3-5 complete possessions
- **Quality**: Clear view of court, players visible
- **Context**: Include scoreboard (helps with temporal alignment)
- **Variety**: Mix of:
  - Made shots (2pt and 3pt)
  - Missed shots
  - Turnovers
  - Fouls
  - Different teams, players, game situations

## Clip Metadata

For each clip, document:

```yaml
clip_id: lakers_celtics_2024_q4_run
game_date: 2024-03-15
teams: LAL vs BOS
quarter: 4
time_range: 8:45 - 7:30
description: Lakers 8-0 run after LeBron frustration moment
notable_events:
  - LeBron missed shot + visible frustration (0:05)
  - Immediate defensive stop (0:15)
  - Fast break made layup (0:22)
  - Made 3-pointer (0:38)
```

Store metadata in `data/clip_metadata.yaml` or similar.

## Target Dataset

### Phase 1 Goal

- **100-200 labeled clips** to prove the approach
- **Diversity**: Multiple teams, players, game situations
- **Quality over quantity**: Clear, complete possessions

### Data Split

- **Training**: 80% (80-160 clips)
- **Testing**: 20% (20-40 clips)

## Legal Considerations

**Important:** Always respect copyright and terms of service.

- **NBA League Pass**: Check if personal use/research is permitted
- **YouTube**: Follow Fair Use guidelines
- **Academic Use**: May have different permissions
- **Commercial Use**: Requires licensing

Consult legal counsel if uncertain.

## Storage

- **Local**: `data/raw_clips/`
- **Backup**: External drive or cloud storage (Dropbox, Google Drive)
- **Do not commit to Git**: Videos are in `.gitignore`

## Next Steps

After collecting clips, proceed to [Labeling Guide](LABELING.md) to annotate possession outcomes.
