"""
Test overhead soccer detection model vs best.pt on real match footage.

Compares average detections per frame, confidence stats, miss rate, and saves
a side-by-side annotated image at frame 50.

Usage:
    python scripts/test_overhead_model.py
    python scripts/test_overhead_model.py --frames 200 --video input_videos/my_game.mp4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_OLD: str = "models/best.pt"
DEFAULT_MODEL_NEW: str = "models/overhead_soccer.pt"
DEFAULT_VIDEO_CANDIDATES: list[str] = [
    "input_videos/chicago_u19_vs_scwave_sep14.mp4",
    "input_videos/scwave_first_half.mp4",
    "input_videos/test_01_kickoff_5min.mp4",
    "input_videos/test_04_first_20min.mp4",
]
NUM_FRAMES: int = 100
COMPARISON_FRAME: int = 50
COMPARE_CONF: float = 0.25   # detection threshold for comparison
OUTPUT_PATH: str = "output_videos/model_comparison_frame50.png"
PLAYER_CLASSES: set[str] = {"player", "goalkeeper"}
MIN_PLAYERS_GOOD_FRAME: int = 10  # frames with this many players count as "good"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class FrameStats(NamedTuple):
    players_detected: int
    confidence_mean: float
    confidence_min: float
    confidence_max: float


class ModelStats(NamedTuple):
    name: str
    avg_players: float
    conf_mean: float
    conf_min: float
    conf_max: float
    zero_detection_frames: int
    good_detection_frames: int
    total_frames: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_video() -> str:
    """Return first existing video from candidates or raise."""
    for path in DEFAULT_VIDEO_CANDIDATES:
        if os.path.exists(path):
            return path
    mp4s = list(Path("input_videos").glob("*.mp4"))
    if mp4s:
        return str(mp4s[0])
    raise FileNotFoundError(
        "No input video found. Place a .mp4 in input_videos/ or pass --video."
    )


def load_frames(video_path: str, n: int) -> list[np.ndarray]:
    """Load the first n frames from video_path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def run_model_on_frames(
    model,
    frames: list[np.ndarray],
    conf: float,
) -> tuple[list[FrameStats], list]:
    """
    Run model on every frame, return per-frame stats and raw results list.

    Returns:
        stats:   list of FrameStats, one per frame
        results: raw ultralytics Results objects (for annotation)
    """
    raw_results = model.predict(frames, conf=conf, verbose=False)

    stats: list[FrameStats] = []
    for res in raw_results:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            stats.append(FrameStats(0, 0.0, 0.0, 0.0))
            continue

        # Filter to player/goalkeeper classes
        player_confs: list[float] = []
        for cls_id, conf_val in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            cls_name = res.names[int(cls_id)]
            if cls_name in PLAYER_CLASSES:
                player_confs.append(float(conf_val))

        n = len(player_confs)
        if n == 0:
            stats.append(FrameStats(0, 0.0, 0.0, 0.0))
        else:
            stats.append(FrameStats(
                players_detected=n,
                confidence_mean=float(np.mean(player_confs)),
                confidence_min=float(np.min(player_confs)),
                confidence_max=float(np.max(player_confs)),
            ))

    return stats, raw_results


def aggregate_stats(name: str, stats: list[FrameStats]) -> ModelStats:
    """Aggregate per-frame stats into a summary ModelStats."""
    total = len(stats)
    players = [s.players_detected for s in stats]
    confs_mean = [s.confidence_mean for s in stats if s.players_detected > 0]
    confs_min  = [s.confidence_min  for s in stats if s.players_detected > 0]
    confs_max  = [s.confidence_max  for s in stats if s.players_detected > 0]

    return ModelStats(
        name=name,
        avg_players=float(np.mean(players)) if players else 0.0,
        conf_mean=float(np.mean(confs_mean)) if confs_mean else 0.0,
        conf_min=float(np.min(confs_min))   if confs_min  else 0.0,
        conf_max=float(np.max(confs_max))   if confs_max  else 0.0,
        zero_detection_frames=sum(1 for p in players if p == 0),
        good_detection_frames=sum(1 for p in players if p >= MIN_PLAYERS_GOOD_FRAME),
        total_frames=total,
    )


def annotate_frame(model, result, frame: np.ndarray, conf: float) -> np.ndarray:
    """Draw bounding boxes on frame using ultralytics plot()."""
    annotated = result.plot(conf=conf, line_width=2, labels=True)
    return annotated


def make_side_by_side(
    old_frame: np.ndarray,
    new_frame: np.ndarray,
    label_old: str,
    label_new: str,
) -> np.ndarray:
    """Stack two annotated frames side by side with labels."""
    h = max(old_frame.shape[0], new_frame.shape[0])

    def pad(img: np.ndarray) -> np.ndarray:
        ph = h - img.shape[0]
        return np.pad(img, ((0, ph), (0, 0), (0, 0)), constant_values=30)

    left  = pad(old_frame)
    right = pad(new_frame)
    combined = np.concatenate([left, right], axis=1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, x in [(label_old, 10), (label_new, left.shape[1] + 10)]:
        cv2.putText(combined, text, (x, 32), font, 1.0, (0, 0, 0), 4)
        cv2.putText(combined, text, (x, 32), font, 1.0, (255, 255, 255), 2)

    return combined


def print_stats(ms: ModelStats) -> None:
    zero_pct = ms.zero_detection_frames / ms.total_frames * 100
    good_pct = ms.good_detection_frames / ms.total_frames * 100
    print(f"  {ms.name}")
    print(f"    avg players/frame : {ms.avg_players:.1f}")
    print(f"    confidence        : mean={ms.conf_mean:.2f}  min={ms.conf_min:.2f}  max={ms.conf_max:.2f}")
    print(f"    0-detection frames: {ms.zero_detection_frames}/{ms.total_frames} ({zero_pct:.0f}%)")
    print(f"    10+ player frames : {ms.good_detection_frames}/{ms.total_frames} ({good_pct:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLO models on overhead soccer footage")
    parser.add_argument("--video",  default=None, help="Path to input video")
    parser.add_argument("--frames", type=int, default=NUM_FRAMES, help="Number of frames to test")
    parser.add_argument("--model-old", default=DEFAULT_MODEL_OLD)
    parser.add_argument("--model-new", default=DEFAULT_MODEL_NEW)
    parser.add_argument("--conf", type=float, default=COMPARE_CONF)
    args = parser.parse_args()

    # --- Locate video ---
    video_path = args.video or find_video()
    print(f"Video : {video_path}")
    print(f"Frames: {args.frames}")
    print()

    # --- Check models ---
    for path in (args.model_old, args.model_new):
        if not os.path.exists(path):
            print(f"ERROR: Model not found: {path}")
            if path == args.model_new:
                print("  Run training first:")
                print("    python scripts/download_basketball_models.py")
            sys.exit(1)

    # --- Load frames ---
    print(f"Loading {args.frames} frames...")
    frames = load_frames(video_path, args.frames)
    if len(frames) < COMPARISON_FRAME + 1:
        print(f"WARNING: only {len(frames)} frames available (need {COMPARISON_FRAME + 1} for comparison image)")
    print(f"  Loaded {len(frames)} frames  ({frames[0].shape[1]}x{frames[0].shape[0]})")
    print()

    # --- Load models ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    print("Loading models...")
    model_old = YOLO(args.model_old)
    model_new = YOLO(args.model_new)
    print(f"  {args.model_old}  ({sum(p.numel() for p in model_old.model.parameters())/1e6:.1f}M params)")
    print(f"  {args.model_new}  ({sum(p.numel() for p in model_new.model.parameters())/1e6:.1f}M params)")
    print()

    # --- Run inference ---
    print(f"Running {args.model_old} on {len(frames)} frames...")
    stats_old, results_old = run_model_on_frames(model_old, frames, args.conf)
    ms_old = aggregate_stats(Path(args.model_old).name, stats_old)

    print(f"Running {args.model_new} on {len(frames)} frames...")
    stats_new, results_new = run_model_on_frames(model_new, frames, args.conf)
    ms_new = aggregate_stats(Path(args.model_new).name, stats_new)

    # --- Print stats ---
    print()
    print("=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print()
    print_stats(ms_old)
    print()
    print_stats(ms_new)
    print()

    # --- Recommendation ---
    if ms_new.avg_players > ms_old.avg_players * 1.1:
        winner = Path(args.model_new).name
        reason = f"detects {ms_new.avg_players:.1f} vs {ms_old.avg_players:.1f} avg players/frame"
    elif ms_old.avg_players > ms_new.avg_players * 1.1:
        winner = Path(args.model_old).name
        reason = f"detects {ms_old.avg_players:.1f} vs {ms_new.avg_players:.1f} avg players/frame"
    elif ms_new.zero_detection_frames < ms_old.zero_detection_frames:
        winner = Path(args.model_new).name
        reason = f"fewer complete misses ({ms_new.zero_detection_frames} vs {ms_old.zero_detection_frames})"
    elif ms_new.conf_mean > ms_old.conf_mean:
        winner = Path(args.model_new).name
        reason = f"higher mean confidence ({ms_new.conf_mean:.2f} vs {ms_old.conf_mean:.2f})"
    else:
        winner = Path(args.model_old).name
        reason = "similar performance; keep existing model"

    print(f"Recommendation: use {winner} — {reason}")
    print()

    # --- Side-by-side image ---
    if len(frames) > COMPARISON_FRAME:
        cmp_frame = frames[COMPARISON_FRAME]
        ann_old = annotate_frame(model_old, results_old[COMPARISON_FRAME], cmp_frame, args.conf)
        ann_new = annotate_frame(model_new, results_new[COMPARISON_FRAME], cmp_frame, args.conf)

        label_old = f"{Path(args.model_old).name}  ({stats_old[COMPARISON_FRAME].players_detected} players)"
        label_new = f"{Path(args.model_new).name}  ({stats_new[COMPARISON_FRAME].players_detected} players)"
        comparison = make_side_by_side(ann_old, ann_new, label_old, label_new)

        os.makedirs("output_videos", exist_ok=True)
        cv2.imwrite(OUTPUT_PATH, comparison)
        print(f"Side-by-side image saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
