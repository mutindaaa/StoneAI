"""
Video Event Bridge

Converts Stone AI's video-derived events JSON into a SPADL-compatible
DataFrame so video matches can be run through the same VAEP/xT engine
as structured StatsBomb data.

Stone AI events schema:
    {event_id, event_type, timestamp_sec, frame_number,
     player_id, team_id, location_x, location_y, metadata}

SPADL schema:
    {game_id, period_id, time_seconds, team_id, player_id,
     start_x, start_y, end_x, end_y,
     type_id, type_name, result_id, result_name, bodypart_id, bodypart_name}

Coordinate conversion:
    Video events store pixel coords (location_x in [0, frame_width]).
    SPADL expects metres on a 105 × 68 pitch.
    Conversion: x_m = location_x * 105 / frame_width
                y_m = location_y * 68  / frame_height
"""

import json

import pandas as pd

from analytics.config import (
    DEFAULT_FRAME_H as _DEFAULT_FRAME_H,
)
from analytics.config import (
    DEFAULT_FRAME_W as _DEFAULT_FRAME_W,
)
from analytics.config import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    RES_4K_H,
    RES_4K_THRESHOLD,
    RES_4K_W,
    RES_HD_H,
    RES_HD_THRESHOLD,
    RES_HD_W,
)

# Mapping from Stone AI event types → SPADL type_name
_TYPE_MAP = {
    "pass": "pass",
    "shot": "shot",
    "possession_change": "tackle",  # closest SPADL equivalent for a turnover
}


def load_video_events(
    events_path: str,
    frame_w: int = _DEFAULT_FRAME_W,
    frame_h: int = _DEFAULT_FRAME_H,
) -> pd.DataFrame:
    """
    Load Stone AI video events JSON and return a SPADL-compatible DataFrame.

    Args:
        events_path:  Path to *_events.json produced by main_v3.py
        frame_w:      Video frame width in pixels (used for coordinate scaling)
        frame_h:      Video frame height in pixels

    Returns:
        DataFrame with SPADL columns (type_name / result_name based).
        Returns empty DataFrame if file is empty or has no mappable events.
    """
    with open(events_path) as f:
        events = json.load(f)

    if not events:
        return pd.DataFrame()

    rows = []
    for e in events:
        etype = e.get("event_type", "")
        if etype not in _TYPE_MAP:
            continue

        # Prefer position_transformed (metres) from metadata if available
        meta = e.get("metadata", {})
        if "position_x_m" in meta and "position_y_m" in meta:
            x_m = float(meta["position_x_m"])
            y_m = float(meta["position_y_m"])
        else:
            # Fall back to pixel → metres scaling
            x_m = e.get("location_x", 0) * PITCH_LENGTH_M / frame_w
            y_m = e.get("location_y", 0) * PITCH_WIDTH_M / frame_h

        x_m = round(max(0.0, min(PITCH_LENGTH_M, x_m)), 2)
        y_m = round(max(0.0, min(PITCH_WIDTH_M, y_m)), 2)

        rows.append(
            {
                "game_id": 0,
                "period_id": 1,
                "time_seconds": float(e.get("timestamp_sec", 0)),
                "team_id": e.get("team_id", "unknown"),
                "player_id": e.get("player_id", "unknown"),
                "start_x": x_m,
                "start_y": y_m,
                # TODO: Replace with actual receiver position once pass destinations are tracked.
                #   Requires: pass receiver track_id in event data + position_transformed for that id.
                "end_x": x_m,
                "end_y": y_m,
                "type_name": _TYPE_MAP[etype],
                "result_name": "success",
                "bodypart_name": "foot",
                # Preserve original fields for reference
                "_event_type": etype,
                "_event_id": e.get("event_id", ""),
                "_frame": e.get("frame_number", 0),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("time_seconds").reset_index(drop=True)
    return df


def infer_frame_dimensions(events_path: str) -> tuple[int, int]:
    """
    Try to infer original frame dimensions from an events JSON file.
    Looks at the max location_x / location_y values as a heuristic.

    Returns (frame_w, frame_h) — falls back to 1280×720 if indeterminate.
    """
    with open(events_path) as f:
        events = json.load(f)

    if not events:
        return _DEFAULT_FRAME_W, _DEFAULT_FRAME_H

    xs = [e.get("location_x", 0) for e in events if e.get("location_x")]
    ys = [e.get("location_y", 0) for e in events if e.get("location_y")]

    if not xs or not ys:
        return _DEFAULT_FRAME_W, _DEFAULT_FRAME_H

    max_x = max(xs)
    max_y = max(ys)

    # Snap to common resolutions
    if max_x > RES_HD_THRESHOLD:
        w = RES_4K_W if max_x > RES_4K_THRESHOLD else RES_HD_W
    else:
        w = _DEFAULT_FRAME_W

    if max_y > 600:
        h = RES_4K_H if max_y > 800 else RES_HD_H
    else:
        h = _DEFAULT_FRAME_H

    return w, h
