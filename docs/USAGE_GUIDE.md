# Soccer Analytics Platform - Usage Guide

Your MVP is now a full-featured soccer analytics platform!

## What's New

✅ **Works on ANY soccer video** (not just one sample)
✅ **Handles 2+ hour videos** without crashing
✅ **GPU-accelerated** processing
✅ **Player statistics export** for scouting

---

## Quick Start

### 1. Process a Short Video (< 5 minutes)

```bash
python main.py
```

This uses the video specified in `main.py` (currently `test_30sec.mp4`).

### 2. Process a Long Video (2+ hours)

```bash
python main_chunked.py --video input_videos/full_match.mp4 --chunk-size 200
```

This processes in chunks to avoid memory issues.

### 3. Calibrate a New Video

Before processing a new video with a different camera angle:

```bash
python calibrate_field.py input_videos/your_new_video.mp4
```

- Click the 4 corners of the field (top-left, top-right, bottom-right, bottom-left)
- Press `s` to save
- This creates `calibration_your_new_video.json`

Then update `main.py` or `main_chunked.py` to use the calibration:

```python
view_transformer = ViewTransformer(calibration_path='calibration_your_new_video.json')
```

### 4. Export Player Statistics

After processing a video, extract player stats:

```python
from player_stats import PlayerStatsAnalyzer

# Load tracks (from pickle file or from processing)
analyzer = PlayerStatsAnalyzer(tracks, video_fps=25)

# Export to JSON
analyzer.export_to_json('player_stats.json')

# Export to CSV
analyzer.export_to_csv('player_stats.csv')

# Print summary report
print(analyzer.generate_summary_report())
```

---

## File Structure

```
football_analysis/
├── main.py                    # Process short videos
├── main_chunked.py            # Process long videos (2+ hours)
├── calibrate_field.py         # Interactive field calibration
├── clip_video.py              # Clip videos to shorter segments
│
├── input_videos/              # Put your videos here
├── output_videos/             # Processed videos go here
├── models/best.pt             # YOLO model
│
├── trackers/                  # Player/ball detection
├── camera_movement_estimator/ # Camera motion tracking
├── view_transformer/          # Field perspective transformation
├── team_assigner/             # Team color detection
├── player_ball_assigner/      # Ball possession tracking
├── speed_and_distance_estimator/  # Player kinematics
├── player_stats/              # Player statistics export
└── utils/                     # Video I/O utilities
```

---

## Workflow for New Videos

1. **Download/Get Video:**
   ```bash
   # Download from YouTube
   python -m yt_dlp "YOUTUBE_URL" -o input_videos/match.mp4

   # Or clip from existing video
   python clip_video.py input_videos/long_video.mp4 input_videos/clip.mp4 --start 120 --duration 60
   ```

2. **Calibrate Field (if different camera angle):**
   ```bash
   python calibrate_field.py input_videos/match.mp4
   ```

3. **Process Video:**
   ```bash
   # Short video
   python main.py

   # Long video
   python main_chunked.py --video input_videos/match.mp4
   ```

4. **Extract Player Stats:**
   ```python
   from player_stats import PlayerStatsAnalyzer
   import pickle

   # Load tracks
   with open('output_videos/output_chunked_tracks.pkl', 'rb') as f:
       tracks = pickle.load(f)

   # Analyze
   analyzer = PlayerStatsAnalyzer(tracks, video_fps=25)
   analyzer.export_to_json('stats.json')
   analyzer.export_to_csv('stats.csv')
   print(analyzer.generate_summary_report())
   ```

---

## What Each Tool Does

### `main.py`
- Loads entire video into RAM
- Best for: Videos < 5 minutes
- Fastest processing
- Uses caching (stub files) for speed

### `main_chunked.py`
- Processes video in chunks (default: 200 frames)
- Best for: Videos 2+ hours
- Uses minimal RAM
- Required flags: `--video`

### `calibrate_field.py`
- Interactive tool to calibrate field corners
- Creates JSON calibration file
- Use once per camera angle/stadium

### `clip_video.py`
- Extract clips from long videos
- Usage: `python clip_video.py input.mp4 output.mp4 --start 60 --duration 30`

### Player Statistics
- Aggregates all tracking data
- Exports JSON/CSV for analysis
- Includes: distance, speed, possession, positions

---

## Tips

- **GPU**: Make sure you have CUDA PyTorch installed for speed
- **Calibration**: Calibrate once per unique camera angle
- **Chunk Size**: Reduce if running out of memory, increase for speed
- **Stubs**: Delete `stubs/*.pkl` to force fresh processing
- **Jersey Numbers**: Install `easyocr` for jersey number detection
- **Low Quality Videos**: System handles 480p+, but 720p+ recommended for best results
  - Lower quality videos may have reduced detection accuracy
  - Jersey number OCR works best on 720p+ footage
  - Speed/distance calculations work on any resolution (uses field calibration)

---

## Next Steps for Full Platform

To turn this into a production scouting platform:

1. **Database**: Store stats across multiple matches
2. **Web UI**: Streamlit/Flask dashboard for visualization
3. **Player Identification**: OCR for jersey numbers
4. **Heat Maps**: Visualize player positions
5. **Comparison Tools**: Compare players across matches
6. **Export Reports**: PDF scouting reports

---

## Recent Improvements

### v2.0 Updates
- ✅ **Smaller, cleaner icons** - Reduced player/ball icon sizes by 30-40%
- ✅ **Removed camera movement overlay** - Cleaner output video
- ✅ **Accurate speed/distance** - Now uses actual video FPS for calculations
- ✅ **Jersey number recognition** - OCR detects player jersey numbers
- ✅ **Camera cut detection** - Identifies when camera angles switch

## Troubleshooting

**"CUDA not available"**
- Reinstall PyTorch with CUDA:
  ```bash
  pip uninstall torch torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

**"Out of memory"**
- Use `main_chunked.py` instead of `main.py`
- Reduce `--chunk-size`

**"Speed/distance values are wrong"**
- Recalibrate the field with `calibrate_field.py`
- Make sure you clicked the correct corners
- System now automatically uses correct video FPS

**"No players detected"**
- Check that `models/best.pt` exists
- Try reducing `conf=0.1` threshold in tracker.py

**"Jersey numbers not appearing"**
- Install EasyOCR: `pip install easyocr`
- OCR works best on high-quality videos (720p+)
- Jersey numbers are cached - once detected, they persist

**"Tracking IDs change when camera angle switches"**
- This is a known limitation of the ByteTrack algorithm
- Camera cuts cause tracking to reset because players appear in different positions
- For pro match footage with multiple camera angles:
  - Process each camera angle segment separately
  - Use the camera cut detector to identify segments
  - Manually clip video at camera changes before processing

---

Your soccer analytics platform is ready! 🎉
