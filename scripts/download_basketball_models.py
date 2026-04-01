"""
Prepare basketball-specific detection models.

The basketball pipeline uses two models:
  1. Player detection  — models/best.pt  (already present, soccer-trained but works for players)
  2. Ball detection    — yolov8n.pt      (COCO general model; "sports ball" class detected automatically)

yolov8n.pt is downloaded automatically by ultralytics on first use. This script
pre-fetches it so the first pipeline run doesn't stall.

Usage:
    python scripts/download_basketball_models.py

Note on Roboflow Universe models:
    roboflow-100/basketball-players-fy4c2 and similar projects are annotation
    datasets (images + labels only). They do not ship pre-trained .pt weights.
    Training locally takes 5-15 min with a GPU:
        yolo train data=<dataset>/data.yaml model=yolov8n.pt epochs=30 imgsz=640
    Copy the result to models/basketball_players.pt and update basketball_template.json.
"""

import sys
from pathlib import Path


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("  pip install ultralytics")
        sys.exit(1)

    # Ultralytics auto-downloads yolov8n.pt from GitHub releases on first load.
    # Just instantiating YOLO("yolov8n.pt") triggers the download if not cached.
    print("Fetching yolov8n.pt (COCO general model for ball detection)...")
    model = YOLO("yolov8n.pt")

    # Confirm "sports ball" class is present
    ball_classes = [name for name in model.names.values() if "ball" in name.lower()]
    if ball_classes:
        print(f"  OK - ball-related classes: {ball_classes}")
    else:
        print(f"  WARNING: no ball class found in {list(model.names.values())[:10]}...")

    print()
    print("Ready. Run the basketball pipeline with:")
    print(
        "  python main_v3.py"
        " --video input_videos/basketball_test.mp4"
        " --config match_configs/basketball_template.json"
        " --units imperial"
    )
    print()
    print("The pipeline uses:")
    print("  - models/best.pt     for player/referee detection")
    print("  - yolov8n.pt         for ball detection (sports ball class)")


if __name__ == "__main__":
    main()
