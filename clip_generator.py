"""
Event-Based Highlight Clip Generator

Reads the events JSON produced by the analysis pipeline and cuts a short clip
around each detected event (pass, shot, goal).  Clips are saved as individual
MP4 files in output_dir.

Usage (standalone):
    python clip_generator.py \
        --video  input_videos/scwave_test_5min.mp4 \
        --events output_videos/chicago_u19_vs_scwave_sep14_events.json \
        --output output_videos/clips

Usage (from main_v3.py via --clips flag):
    python main_v3.py --config ... --video ... --clips
"""

import os
import json
import argparse
import cv2


# Event types to clip — skip possession_change (too frequent)
CLIP_EVENT_TYPES = {'pass', 'shot', 'goal', 'corner', 'free_kick'}

# Seconds of footage before and after the event timestamp
PADDING_BEFORE_SEC = 3.0
PADDING_AFTER_SEC  = 4.0


def generate_clips(
    video_path: str,
    events_path: str,
    output_dir: str,
    padding_before: float = PADDING_BEFORE_SEC,
    padding_after:  float = PADDING_AFTER_SEC,
    event_types: set = CLIP_EVENT_TYPES,
) -> list[str]:
    """
    Cut highlight clips around detected events.

    Args:
        video_path:    Source video file.
        events_path:   Path to events JSON (from analysis pipeline).
        output_dir:    Directory to write clip files.
        padding_before: Seconds to include before event timestamp.
        padding_after:  Seconds to include after event timestamp.
        event_types:   Set of event_type strings to clip.

    Returns:
        List of output clip file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(events_path, 'r') as f:
        events = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open video {video_path}")
        return []

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Filter to clippable events, deduplicate overlapping windows
    clippable = [e for e in events if e.get('event_type') in event_types]
    print(f"  Clipping {len(clippable)} events → {output_dir}/")

    output_paths = []
    prev_end_frame = -1  # avoid writing the same frame range twice

    for evt in clippable:
        t = float(evt.get('timestamp_sec', 0))
        start_frame = max(0, int((t - padding_before) * fps))
        end_frame   = min(total_frames - 1, int((t + padding_after) * fps))

        # Skip if heavily overlapping with previous clip
        if start_frame < prev_end_frame - int(fps * 1.0):
            start_frame = prev_end_frame  # extend previous clip instead of duplicate

        if start_frame >= end_frame:
            continue

        evt_type    = evt.get('event_type', 'event')
        team_id     = evt.get('team_id', 'unknown')
        safe_team   = team_id.replace(' ', '_')
        fname       = f"{evt_type}_{t:.1f}s_{safe_team}.mp4"
        out_path    = os.path.join(output_dir, fname)

        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        output_paths.append(out_path)
        prev_end_frame = end_frame
        print(f"    {fname}  ({start_frame/fps:.1f}s – {end_frame/fps:.1f}s)")

    cap.release()
    print(f"  {len(output_paths)} clips saved to {output_dir}/")
    return output_paths


def _build_highlight_reel(clip_paths: list[str], output_path: str, fps: float = 30.0):
    """
    Concatenate all clips into a single highlight reel MP4.
    """
    if not clip_paths:
        return

    # Read dimensions from first clip
    cap0 = cv2.VideoCapture(clip_paths[0])
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for path in clip_paths:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()

    writer.release()
    print(f"  Highlight reel: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut event-based highlight clips')
    parser.add_argument('--video',  required=True, help='Source video path')
    parser.add_argument('--events', required=True, help='Events JSON path')
    parser.add_argument('--output', default='output_videos/clips', help='Output directory')
    parser.add_argument('--reel',   action='store_true', help='Also create a highlight reel')
    parser.add_argument('--before', type=float, default=PADDING_BEFORE_SEC)
    parser.add_argument('--after',  type=float, default=PADDING_AFTER_SEC)
    args = parser.parse_args()

    clips = generate_clips(
        video_path=args.video,
        events_path=args.events,
        output_dir=args.output,
        padding_before=args.before,
        padding_after=args.after,
    )

    if args.reel and clips:
        reel_path = os.path.join(args.output, 'highlight_reel.mp4')
        _build_highlight_reel(clips, reel_path)
