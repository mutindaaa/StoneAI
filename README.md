# Stone AI

**Sport-agnostic analytics platform that extracts professional-grade intelligence from raw match video — built for clubs, academies, and leagues that don't have access to enterprise data providers.**

Stone AI bridges the gap between raw video and actionable analytics. Point it at any match footage — broadcast, Veo, or phone camera — and it produces the same shot maps, player valuation scores, possession metrics, and tactical reports that only top-flight clubs currently get.

---

## What it does

**Video intelligence pipeline**
- Detects and tracks players, ball, and referees using YOLOv8 (custom-trained)
- Classifies players into teams automatically via SigLIP jersey color embeddings + KMeans clustering
- Handles broadcast cameras (panning/zooming) with optical flow compensation
- Transforms pixel coordinates into real-world pitch meters via homography
- Computes per-player speed, distance, and sprint counts
- Detects passes, shots, and possession changes as structured events
- Generates radar mini-map overlay and highlight clip exports
- Scales to full 90-minute matches via chunked streaming (O(1) RAM)

**Data analytics layer**
- Ingests structured event data from StatsBomb, Opta, Wyscout via kloppy
- Converts events to SPADL format and computes VAEP + xT player valuation scores
- Generates shot maps, pass networks, heatmaps, and player radar charts via mplsoccer
- Video-derived events feed into the same analytics engine as structured data
- Streamlit dashboard with match explorer and analytics tab

**Multi-sport architecture**
- Soccer: full pipeline operational
- Basketball: architecture in place, expanding for NBA Africa use case

---

## Quickstart

\\\ash
git clone https://github.com/mutindaaa/StoneAI
cd StoneAI
pip install -r requirements.txt

# Run with a match config (recommended)
python main_v3.py --config match_configs/example_match.json

# Zero-config auto mode
python main_v3.py --video input_videos/game.mp4 --auto --units imperial

# Full pipeline + analytics + highlight clips
python main_v3.py --config match_configs/example_match.json --analytics --clips
\\\

## Models

Models are not committed to this repo due to file size. Download to \models/\:

| Model | Purpose | Size |
|-------|---------|------|
| \est.pt\ | Custom YOLOv8 player/ball detection | 165MB |
| \pitch_detection.pt\ | Pitch keypoint detection for radar | 134MB |
| \yolov8x.pt\ | Base detection model | 131MB |

## Architecture

\\\
Raw video (MP4/AVI)
    ↓
YOLOv8 detection → BoT-SORT tracking → SigLIP team classification
    ↓
Camera movement compensation (optical flow)
    ↓
ViewTransformer (pixel → pitch meters via homography)
    ↓
Speed/distance/possession/event detection
    ↓
Analytics layer (kloppy → SPADL → VAEP/xT → mplsoccer)
    ↓
Streamlit dashboard + JSON metrics + annotated video
\\\

## Target markets

- Soccer clubs and academies in underserved markets (Africa, lower-tier US leagues)
- NBA Africa basketball development infrastructure
- Any club running Veo/BePro cameras that wants pro-level analytics without enterprise pricing

---

Built by [@mutindaaa](https://github.com/mutindaaa)
