# Stone AI

> AI-powered soccer analytics platform combining computer vision and structured event data.

Stone AI processes raw match video end-to-end — from player detection through team classification,
event extraction, and statistical analytics — and exposes everything through a REST API and
Streamlit dashboard.

---

## What it does

| Layer | Capability |
|-------|-----------|
| **Detection** | Detects players, ball, and referees frame-by-frame using a fine-tuned YOLOv8x model |
| **Tracking** | Maintains player IDs across frames with BoT-SORT; handles occlusion and camera cuts |
| **Team classification** | Assigns players to teams using SigLIP jersey-color embeddings + UMAP clustering |
| **Spatial calibration** | Maps pixel coordinates to real-world metres via homography (ViewTransformer) |
| **Metrics** | Computes speed, distance, sprint zones, and possession per player per match |
| **Event detection** | Identifies passes, shots, and possession changes as structured timestamped events |
| **xT analytics** | Runs Expected Threat (xT) player valuation using a corpus-fitted 12×16 Karun Singh grid |
| **Visualizations** | Generates shot maps, pass networks, player heatmaps, and xT bar charts via mplsoccer |
| **Open data** | Connects to StatsBomb open data (40+ competitions) via kloppy for benchmark analysis |
| **API** | FastAPI server for submitting jobs, polling status, and fetching results |
| **Dashboard** | Streamlit UI with live job queue, results viewer, and analytics explorer |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/mutindaaa/StoneAI.git
cd StoneAI
pip install -e .

# For CUDA-accelerated PyTorch (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Roboflow Sports (pitch config + ViewTransformer):
pip install git+https://github.com/roboflow/sports.git
```

### 2. Download models

Models are not committed to this repo due to file size. Download and place in `models/`:

| File | Description |
|------|-------------|
| `models/best.pt` | Fine-tuned YOLOv8x for player/ball/referee detection |
| `models/yolov8x.pt` | Base YOLOv8x weights |
| `models/pitch_detection.pt` | Pitch keypoint detector |

### 3. Run the pipeline

```bash
# Process a match video using a config file
python main_v3.py --config match_configs/example_match.json

# Auto-detect teams (no roster needed)
python main_v3.py --config match_configs/auto_template.json --auto

# Generate highlight clips and a reel
python main_v3.py --config match_configs/example_match.json --clips --reel

# Run analytics after pipeline
python main_v3.py --config match_configs/example_match.json --analytics
```

### 4. Run analytics standalone

```bash
# StatsBomb open data (no credentials needed)
python analytics/run_analysis.py --source statsbomb --match_id 3788741

# Video-derived events from the pipeline
python analytics/run_analysis.py --source video \
    --events output_videos/my_match_events.json
```

On first run the xT model is fitted on 50 La Liga 2015/16 matches and cached to
`analytics/xt_grid_cache.npy` (~2 min). Subsequent runs load from cache instantly.

### 5. API server

```bash
uvicorn api.server:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### 6. Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Architecture

```
Input video
    │
    ▼
┌─────────────────────────────────┐
│  Tracker (BoT-SORT + YOLOv8x)  │  → player/ball bounding boxes + IDs
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Team Classifier (SigLIP)       │  → team_id per player per frame
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Camera Movement Estimator      │  → homography matrix per frame
│  View Transformer               │  → pixel → metres mapping
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Metrics + Event Detector       │  → speed, distance, passes, shots
└─────────────────────────────────┘
    │                    │
    ▼                    ▼
Annotated video     events.json + metrics.json
                         │
                         ▼
               ┌─────────────────┐
               │  Analytics      │  → xT, shot maps, pass networks
               │  (kloppy /      │
               │   mplsoccer)    │
               └─────────────────┘
```

---

## Match config

Match configs live in `match_configs/`. Minimal example:

```json
{
  "video_path": "input_videos/my_match.mp4",
  "output_path": "output_videos/my_match_output.mp4",
  "team1_name": "Home FC",
  "team2_name": "Away United"
}
```

See [match_configs/example_match.json](match_configs/example_match.json) for all options.

---

## Project layout

```
stone-ai/
├── main_v3.py                  # Pipeline entry point
├── pyproject.toml              # Package definition & dependencies
├── match_configs/              # Per-match JSON configs
├── analytics/                  # xT, VAEP, mplsoccer visualizations
│   ├── statsbomb_loader.py
│   ├── spadl_pipeline.py       # Corpus-fitted xT grid
│   ├── video_bridge.py         # Video events → SPADL
│   ├── visualizer.py
│   └── run_analysis.py         # CLI runner
├── api/                        # FastAPI server
├── dashboard/                  # Streamlit UI
├── trackers/                   # BoT-SORT wrapper
├── team_assigner/              # SigLIP team classifier
├── camera_movement_estimator/
├── speed_and_distance_estimator/
├── player_ball_assigner/
├── player_stats/
├── radar/
└── models/                     # Model weights (not committed — download separately)
```

---

## License

MIT
