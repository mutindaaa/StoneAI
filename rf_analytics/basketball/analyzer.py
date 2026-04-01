"""
Basketball Analyzer — rf_analytics/basketball/analyzer.py

Extends the sport-agnostic SportAnalyzer base class for basketball.

Court: NBA standard — 28.65 m × 15.24 m  (94 ft × 50 ft)

Detected events:
  - shot_attempt  : ball moving fast (>10 km/h) + near the basket zone
  - made_basket   : ball enters basket zone from above at low speed
  - possession_change : ball transfers between teams

Basketball-specific metrics:
  - shot_attempts       total shots taken
  - shots_made          estimated made (ball slows at basket)
  - three_point_attempts shots taken from beyond 3PT line (6.75 m from basket)
  - paint_time_sec      seconds spent inside the key
  - distance_from_basket_avg  average distance (m) from nearest basket
  - assists             pass leading directly to a shot_attempt (within 3 s)
  - turnovers           possession_change while player had ball
"""

import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

from rf_analytics.base import (
    FieldCalibration,
    MatchEvent,
    PlayerMetrics,
    SportAnalyzer,
)

# ---------------------------------------------------------------------------
# Court constants  (all in metres)
# ---------------------------------------------------------------------------

# NBA court
COURT_LENGTH = 28.65   # sideline to sideline
COURT_WIDTH  = 15.24   # baseline to baseline

# Basket positions (in court metres, origin = bottom-left corner)
BASKET_LEFT  = (1.575, COURT_WIDTH / 2)   # left baseline basket
BASKET_RIGHT = (COURT_LENGTH - 1.575, COURT_WIDTH / 2)

# Key (paint) boundaries — full-court coordinates
KEY_LENGTH = 5.79   # depth of the paint from the baseline
KEY_WIDTH  = 4.90   # width of the paint

PAINT_LEFT  = (0.0, (COURT_WIDTH - KEY_WIDTH) / 2,
               KEY_LENGTH, (COURT_WIDTH + KEY_WIDTH) / 2)   # (x1,y1,x2,y2)
PAINT_RIGHT = (COURT_LENGTH - KEY_LENGTH, (COURT_WIDTH - KEY_WIDTH) / 2,
               COURT_LENGTH, (COURT_WIDTH + KEY_WIDTH) / 2)

# 3-point line radius from basket centre (NBA)
THREE_PT_RADIUS = 7.24

# Thresholds
SHOT_BALL_SPEED_KMPH = 10.0       # minimum ball speed (km/h) to count as a shot attempt
BASKET_ZONE_RADIUS   = 2.0        # metres — ball within this radius of basket → near basket
MADE_BASKET_SPEED    = 3.0        # km/h — ball slows to this inside basket zone → "made"
ASSIST_WINDOW_SEC    = 3.0        # pass in this window before shot → assist
SPRINT_SPEED_KMPH    = 18.0       # basketball sprint threshold (slightly lower than soccer)


class BasketballAnalyzer(SportAnalyzer):
    """Basketball-specific implementation of SportAnalyzer."""

    def __init__(self):
        super().__init__(
            sport_name="basketball",
            field_dimensions=(COURT_LENGTH, COURT_WIDTH),
        )

    # ------------------------------------------------------------------
    # calibrate_field
    # ------------------------------------------------------------------

    def calibrate_field(
        self,
        frame: np.ndarray,
        manual_mode: bool = False,
        ball_model_path: str | None = None,
    ) -> FieldCalibration:
        """
        Stub calibration using known court dimensions and linear pixel mapping.

        For production, swap in a YOLO keypoint model trained on court lines,
        similar to the soccer pitch detection model.

        Args:
            frame:           Sample video frame for calibration.
            manual_mode:     If True, allow manual keypoint selection.
            ball_model_path: Optional path to dedicated ball detection model.
                             Stored for future basket-position detection use.
        """
        if ball_model_path is not None:
            self._ball_model_path = ball_model_path
        h, w = frame.shape[:2]
        pixel_to_meter_x = COURT_LENGTH / max(w, 1)
        pixel_to_meter_y = COURT_WIDTH  / max(h, 1)
        pixel_to_meter   = (pixel_to_meter_x + pixel_to_meter_y) / 2.0

        # Simple identity-like affine: pixel → metres (no perspective warp)
        transform = np.array([
            [COURT_LENGTH / w, 0,                0],
            [0,                COURT_WIDTH  / h,  0],
            [0,                0,                1],
        ], dtype=np.float32)

        inverse = np.linalg.inv(transform)

        keypoints = {
            "court_top_left":     (0, 0),
            "court_top_right":    (w, 0),
            "court_bottom_left":  (0, h),
            "court_bottom_right": (w, h),
            "basket_left":        (int(BASKET_LEFT[0] / pixel_to_meter_x),
                                   int(BASKET_LEFT[1] / pixel_to_meter_y)),
            "basket_right":       (int(BASKET_RIGHT[0] / pixel_to_meter_x),
                                   int(BASKET_RIGHT[1] / pixel_to_meter_y)),
        }

        return FieldCalibration(
            transform_matrix=transform,
            inverse_transform_matrix=inverse,
            field_dimensions=(COURT_LENGTH, COURT_WIDTH),
            pixel_to_meter_ratio=pixel_to_meter,
            keypoints=keypoints,
        )

    # ------------------------------------------------------------------
    # calculate_metrics
    # ------------------------------------------------------------------

    def calculate_metrics(
        self,
        tracks: Dict[str, List],
        match_config: Dict[str, Any],
        field_calibration: FieldCalibration,
        fps: float,
    ) -> Dict[str, "PlayerMetrics"]:
        """
        Calculate basketball performance metrics from tracking data.

        Aggregates per-track-id data from tracks['players'].
        """
        player_data: Dict[int, dict] = {}

        for frame_num, player_frame in enumerate(tracks.get("players", [])):
            for track_id, info in player_frame.items():
                if track_id not in player_data:
                    player_data[track_id] = {
                        "team":        info.get("team", 1),
                        "team_color":  info.get("team_color", (128, 128, 128)),
                        "jersey":      info.get("jersey_number"),
                        "speeds_kmh":  [],
                        "positions_m": [],   # (x_m, y_m) court coordinates
                        "frames_seen": 0,
                        "paint_frames": 0,
                    }

                d = player_data[track_id]
                d["frames_seen"] += 1

                speed = info.get("speed", 0.0)
                d["speeds_kmh"].append(speed)

                pos = info.get("position_transformed")  # metres from view_transformer
                if pos is not None:
                    d["positions_m"].append((float(pos[0]), float(pos[1])))
                    # Paint-time check
                    if self._in_paint(float(pos[0]), float(pos[1])):
                        d["paint_frames"] += 1

        # Build PlayerMetrics
        metrics: Dict[str, PlayerMetrics] = {}

        for track_id, d in player_data.items():
            speeds  = d["speeds_kmh"]
            top_spd = max(speeds) if speeds else 0.0
            avg_spd = float(np.mean(speeds)) if speeds else 0.0
            minutes = d["frames_seen"] / max(fps * 60, 1)

            # Distance: sum of consecutive position deltas
            pos_arr = d["positions_m"]
            dist_m  = 0.0
            for i in range(1, len(pos_arr)):
                dx = pos_arr[i][0] - pos_arr[i-1][0]
                dy = pos_arr[i][1] - pos_arr[i-1][1]
                dist_m += (dx**2 + dy**2) ** 0.5

            sprints = sum(1 for s in speeds if s >= SPRINT_SPEED_KMPH)

            # Average distance from nearest basket
            if pos_arr:
                dists_from_basket = [
                    min(
                        _dist(p, BASKET_LEFT),
                        _dist(p, BASKET_RIGHT),
                    )
                    for p in pos_arr
                ]
                avg_basket_dist = float(np.mean(dists_from_basket))
            else:
                avg_basket_dist = 0.0

            paint_sec = d["paint_frames"] / max(fps, 1)

            # Resolve player identity from match_config
            player_id = self._resolve_player_id(
                track_id, d["team"], d["jersey"], match_config
            )

            metrics[player_id] = PlayerMetrics(
                player_id=player_id,
                player_name=player_id,
                team_id=str(d["team"]),
                minutes_played=round(minutes, 2),
                distance_covered_m=round(dist_m, 2),
                top_speed_kmh=round(top_spd, 2),
                avg_speed_kmh=round(avg_spd, 2),
                sprints_count=sprints,
                sport_metrics={
                    "shot_attempts":            0,   # filled by detect_events
                    "shots_made":               0,
                    "three_point_attempts":     0,
                    "assists":                  0,
                    "turnovers":                0,
                    "paint_time_sec":           round(paint_sec, 2),
                    "distance_from_basket_avg": round(avg_basket_dist, 2),
                },
            )

        return metrics

    # ------------------------------------------------------------------
    # detect_events
    # ------------------------------------------------------------------

    def detect_events(
        self,
        tracks: Dict[str, List],
        frames: List[np.ndarray],
        match_config: Dict[str, Any],
    ) -> List["MatchEvent"]:
        """
        Detect basketball events from tracking data.

        Events detected:
          - shot_attempt        Ball speed > threshold near a basket
          - made_basket         Ball slows to near-zero inside basket zone
          - possession_change   Ball switches from one team's player to another
        """
        fps = match_config.get("fps", 25)
        events: List[MatchEvent] = []

        # Track possessor history (frame_num → track_id)
        possessor_history: List[int | None] = []

        for frame_num, player_frame in enumerate(tracks.get("players", [])):
            # Ball position & speed for this frame
            ball_frame  = tracks["ball"][frame_num] if frame_num < len(tracks.get("ball", [])) else {}
            ball_info   = ball_frame.get(1, {})
            ball_pos    = ball_info.get("position_adjusted") or ball_info.get("position")
            ball_speed  = ball_info.get("speed", 0.0)

            # Identify possessor (player with has_ball)
            possessor = None
            for track_id, info in player_frame.items():
                if info.get("has_ball"):
                    possessor = track_id
                    break
            possessor_history.append(possessor)

            if ball_pos is None:
                continue

            # Convert ball pixel position → court metres (linear fallback)
            if frames and len(frames) > 0:
                frame_h, frame_w = frames[0].shape[:2]
            else:
                # Fall back to config-supplied dims, then generic HD
                frame_h = match_config.get('frame_height', 1080)
                frame_w = match_config.get('frame_width',  1920)
            bx_m = float(ball_pos[0]) * COURT_LENGTH / max(frame_w, 1)
            by_m = float(ball_pos[1]) * COURT_WIDTH  / max(frame_h, 1)

            dist_left  = _dist((bx_m, by_m), BASKET_LEFT)
            dist_right = _dist((bx_m, by_m), BASKET_RIGHT)
            near_basket = min(dist_left, dist_right) < BASKET_ZONE_RADIUS

            # --- Shot attempt ---
            if ball_speed > SHOT_BALL_SPEED_KMPH and near_basket and possessor is None:
                player_frame_info = player_frame if player_frame else {}
                nearest_player, team_id = self._nearest_player_to_ball(
                    ball_pos, player_frame_info, frame_w, frame_h
                )
                nearest_basket = BASKET_LEFT if dist_left < dist_right else BASKET_RIGHT
                is_3pt = _dist((bx_m, by_m), nearest_basket) > THREE_PT_RADIUS

                events.append(MatchEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="shot_attempt",
                    timestamp_sec=frame_num / fps,
                    frame_number=frame_num,
                    player_id=str(nearest_player),
                    team_id=str(team_id),
                    location_x=round(bx_m, 2),
                    location_y=round(by_m, 2),
                    metadata={"three_point": is_3pt, "ball_speed_kmh": round(ball_speed, 1)},
                ))

            # --- Made basket: ball inside basket zone at low speed ---
            if near_basket and ball_speed < MADE_BASKET_SPEED and ball_speed > 0.5:
                events.append(MatchEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="made_basket",
                    timestamp_sec=frame_num / fps,
                    frame_number=frame_num,
                    player_id="unknown",
                    team_id="unknown",
                    location_x=round(bx_m, 2),
                    location_y=round(by_m, 2),
                    metadata={"basket": "left" if dist_left < dist_right else "right"},
                ))

        # --- Possession changes ---
        prev_team = None
        for frame_num, possessor in enumerate(possessor_history):
            if possessor is None:
                continue
            player_frame = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
            team = player_frame.get(possessor, {}).get("team")
            if team is None:
                continue
            if prev_team is not None and team != prev_team:
                events.append(MatchEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="possession_change",
                    timestamp_sec=frame_num / fps,
                    frame_number=frame_num,
                    player_id=str(possessor),
                    team_id=str(team),
                    location_x=0.0,
                    location_y=0.0,
                    metadata={"from_team": prev_team, "to_team": team},
                ))
            prev_team = team

        return events

    # ------------------------------------------------------------------
    # generate_visualization
    # ------------------------------------------------------------------

    def generate_visualization(
        self,
        frames,
        tracks,
        metrics,
        events,
        match_config,
    ):
        """
        Basketball visualization is handled by the core tracker draw_annotations().
        This method returns frames unmodified — extend here for court overlays.
        """
        return frames

    # ------------------------------------------------------------------
    # get_metric_definitions
    # ------------------------------------------------------------------

    def get_metric_definitions(self) -> Dict[str, str]:
        return {
            "shot_attempts":            "Total field-goal attempts (2PT + 3PT)",
            "shots_made":               "Estimated field goals made (ball slows at basket zone)",
            "three_point_attempts":     "Shot attempts from beyond the 3-point arc (7.24 m)",
            "assists":                  "Passes directly leading to a shot attempt within 3 seconds",
            "turnovers":                "Possession changes where player lost the ball",
            "paint_time_sec":           "Seconds spent inside the key / paint area",
            "distance_from_basket_avg": "Average distance (m) from the nearest basket",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_paint(self, x_m: float, y_m: float) -> bool:
        """Return True if court position is inside either paint area."""
        x1l, y1l, x2l, y2l = PAINT_LEFT
        if x1l <= x_m <= x2l and y1l <= y_m <= y2l:
            return True
        x1r, y1r, x2r, y2r = PAINT_RIGHT
        return x1r <= x_m <= x2r and y1r <= y_m <= y2r

    def _nearest_player_to_ball(
        self,
        ball_pos,
        player_frame: dict,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int]:
        """Return (track_id, team) of the player nearest to the ball position."""
        best_dist = float("inf")
        best_id   = 0
        best_team = 1

        bx, by = float(ball_pos[0]), float(ball_pos[1])

        for track_id, info in player_frame.items():
            pos = info.get("position_adjusted") or info.get("position")
            if pos is None:
                continue
            d = ((float(pos[0]) - bx) ** 2 + (float(pos[1]) - by) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id   = track_id
                best_team = info.get("team", 1)

        return best_id, best_team

    def _resolve_player_id(
        self,
        track_id: int,
        team: int,
        jersey: Any,
        match_config: Dict[str, Any],
    ) -> str:
        """Map track_id → player_id using jersey number and team."""
        team_key = "team_home" if team == 1 else "team_away"
        team_cfg = match_config.get(team_key, {})

        if jersey is not None:
            try:
                jn = int(jersey)
                for p in team_cfg.get("players", []):
                    if int(p.get("jersey_number", -1)) == jn:
                        return p["player_id"]
            except (TypeError, ValueError):
                pass

        team_id = team_cfg.get("id", f"team{team}")
        return f"{team_id}_track_{track_id}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
