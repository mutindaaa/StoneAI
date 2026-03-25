"""
Soccer-Specific Analyzer

Implements sport-specific logic for soccer/football analysis including:
- Field calibration using pitch markings
- Soccer metrics (possession, passes, shots, etc.)
- Event detection (goals, shots, tackles, etc.)
- Soccer-specific visualization
"""

import sys
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple

sys.path.append('../..')
from rf_analytics.base import SportAnalyzer, FieldCalibration, PlayerMetrics, MatchEvent
from view_transformer import ViewTransformer
from utils import measure_distance, get_foot_position


class SoccerAnalyzer(SportAnalyzer):
    """Soccer-specific implementation of SportAnalyzer"""

    # Standard soccer field dimensions (meters)
    FIELD_LENGTH = 105.0
    FIELD_WIDTH = 68.0

    def __init__(self):
        super().__init__(
            sport_name="soccer",
            field_dimensions=(self.FIELD_LENGTH, self.FIELD_WIDTH)
        )
        self.view_transformer = ViewTransformer()

    def calibrate_field(self, frame: np.ndarray, manual_mode: bool = False) -> FieldCalibration:
        """
        Calibrate soccer field using pitch markings.

        Detects:
        - Corner flags
        - Penalty box corners
        - Center circle
        - Touchline/goal line intersections

        Args:
            frame: Sample frame showing the field
            manual_mode: If True, show GUI for manual keypoint adjustment

        Returns:
            FieldCalibration with perspective transformation
        """
        # Use existing ViewTransformer logic
        # The ViewTransformer already handles field detection

        # Get transformation matrix (from ViewTransformer)
        # This is already being done in the existing code
        # We're just wrapping it in the new interface

        # For now, return a basic calibration
        # TODO: Enhance ViewTransformer to return full calibration data
        keypoints = {
            'corner_top_left': (0, 0),
            'corner_top_right': (frame.shape[1], 0),
            'corner_bottom_left': (0, frame.shape[0]),
            'corner_bottom_right': (frame.shape[1], frame.shape[0]),
        }

        # Placeholder transform (ViewTransformer handles this internally)
        transform_matrix = np.eye(3)
        inverse_transform = np.eye(3)

        calibration = FieldCalibration(
            transform_matrix=transform_matrix,
            inverse_transform_matrix=inverse_transform,
            field_dimensions=self.field_dimensions,
            pixel_to_meter_ratio=1.0,  # Calculated by ViewTransformer
            keypoints=keypoints
        )

        return calibration

    def calculate_metrics(
        self,
        tracks: Dict[str, List],
        match_config: Dict[str, Any],
        field_calibration: FieldCalibration,
        fps: float
    ) -> Dict[str, PlayerMetrics]:
        """
        Calculate soccer-specific metrics for all players.

        Metrics:
        - Distance covered (m)
        - Top speed (km/h)
        - Average speed (km/h)
        - Sprint count
        - Possession time (sec)
        - Passes attempted/completed
        - Pass accuracy (%)
        - Shots
        - Tackles

        Args:
            tracks: Tracking data with transformed positions and speeds
            match_config: Match configuration with player roster
            field_calibration: Field calibration data
            fps: Video frame rate

        Returns:
            Dictionary mapping player_id to PlayerMetrics
        """
        player_metrics = {}

        # Get player tracks
        player_tracks = tracks.get('players', [])
        ball_tracks = tracks.get('ball', [])

        # Build player ID mapping (tracking_id -> player_id)
        player_id_map = self._build_player_id_map(player_tracks, match_config)

        # Minimum frames to be considered a real player track (1 second)
        # Lower threshold so event-making tracks (passes, shots) are included
        # even when BoT-SORT fragments mid-play
        min_frames = max(10, int(fps * 1.0))

        # Track each unique player
        for tracking_id, player_id in player_id_map.items():
            # Initialize metrics
            total_distance = 0.0
            top_speed = 0.0
            speeds = []
            frames_with_ball = 0
            total_frames = 0

            # Analyze player across all frames
            for frame_num, frame_players in enumerate(player_tracks):
                if tracking_id not in frame_players:
                    continue

                player_data = frame_players[tracking_id]
                total_frames += 1

                # Get speed and distance (already calculated by existing code)
                # Note: 'distance' is cumulative total, not per-frame delta
                speed = player_data.get('speed', 0.0)
                distance = player_data.get('distance', 0.0)

                # Take max since distance is a running cumulative total
                total_distance = max(total_distance, distance)
                top_speed = max(top_speed, speed)
                speeds.append(speed)

                # Check if player has ball
                if player_data.get('has_ball', False):
                    frames_with_ball += 1

            # Skip ghost tracks (noise detections visible for < 0.5 seconds)
            if total_frames < min_frames:
                continue

            # Calculate averages
            avg_speed = np.mean(speeds) if speeds else 0.0
            minutes_played = total_frames / fps / 60.0
            possession_time = frames_with_ball / fps

            # Sprint count (speed > 24 km/h for at least 1 second)
            sprint_threshold = 24.0  # km/h
            sprint_frames = [s > sprint_threshold for s in speeds]
            sprints_count = self._count_sprint_sequences(sprint_frames, fps)

            # Get player name from config
            player_name = self._get_player_name(player_id, match_config)
            team_id = self._get_player_team(player_id, match_config)

            # Speed zones (FIFA-aligned thresholds, in km/h)
            ZONE_THRESHOLDS = [0.0, 2.0, 7.0, 15.0, 20.0, 25.0]
            ZONE_NAMES = ['standing', 'walking', 'jogging', 'running', 'high_speed', 'sprinting']
            speed_zones = {z: 0.0 for z in ZONE_NAMES}
            sprint_distance_m = 0.0
            for spd in speeds:
                for j, lo in enumerate(ZONE_THRESHOLDS):
                    hi = ZONE_THRESHOLDS[j + 1] if j + 1 < len(ZONE_THRESHOLDS) else 9999.0
                    if lo <= spd < hi:
                        speed_zones[ZONE_NAMES[j]] += 1.0 / fps
                        break
                if spd >= 25.0:
                    # Approximate incremental distance from speed: d = v/3.6 / fps
                    sprint_distance_m += spd / 3.6 / fps

            # Soccer-specific metrics
            sport_metrics = {
                'possession_time_sec': round(possession_time, 2),
                'passes_attempted': 0,  # back-filled from events in main_v3.py
                'passes_completed': 0,
                'pass_accuracy': 0.0,
                'shots': 0,  # back-filled from events in main_v3.py
                'shots_on_target': 0,
                'tackles_won': 0,
                'tackles_attempted': 0,
                'speed_zones_sec': {k: round(v, 2) for k, v in speed_zones.items()},
                'sprint_distance_m': round(sprint_distance_m, 1),
            }

            metrics = PlayerMetrics(
                player_id=player_id,
                player_name=player_name,
                team_id=team_id,
                minutes_played=minutes_played,
                distance_covered_m=total_distance,
                top_speed_kmh=top_speed,
                avg_speed_kmh=avg_speed,
                sprints_count=sprints_count,
                sport_metrics=sport_metrics
            )

            player_metrics[player_id] = metrics

        return player_metrics

    def detect_events(
        self,
        tracks: Dict[str, List],
        frames: List[np.ndarray],
        match_config: Dict[str, Any]
    ) -> List[MatchEvent]:
        """
        Detect soccer-specific events.

        Events:
        - Goals (ball crosses goal line)
        - Shots (ball moves towards goal with speed)
        - Passes (ball transferred between teammates)
        - Tackles (ball possession changes)
        - Corners (ball near corner flag)
        - Free kicks (play stops and restarts)

        Args:
            tracks: Tracking data
            frames: Video frames
            match_config: Match configuration

        Returns:
            List of detected events
        """
        events = []
        event_counter = 0

        ball_tracks = tracks.get('ball', [])
        player_tracks = tracks.get('players', [])

        # Build possession sequence: frame_num -> (track_id, team_id)
        possession = {}
        for frame_num, frame_players in enumerate(player_tracks):
            for track_id, pdata in frame_players.items():
                if pdata.get('has_ball', False):
                    possession[frame_num] = (track_id, pdata.get('team', 0))
                    break

        fps = 30.0  # fallback; caller should pass this via match_config ideally
        min_pass_frames = int(fps * 0.5)   # ball must travel >0.5s to count as pass
        shot_speed_threshold = 18.0        # km/h ball speed threshold for shots
        min_shot_frames = int(fps * 0.2)   # ball must move toward goal for 0.2s

        # --- Pass detection ---
        # A pass occurs when ball possession transfers from player A to teammate B
        prev_possessor = None
        prev_frame = -1

        for frame_num in sorted(possession.keys()):
            track_id, team_id = possession[frame_num]

            if prev_possessor is not None:
                prev_track, prev_team = prev_possessor

                # Same team, different player, gap not too long = pass
                gap = frame_num - prev_frame
                if (prev_track != track_id and
                        prev_team == team_id and
                        min_pass_frames <= gap <= int(fps * 4)):

                    # Filter out same-player fragment handoffs:
                    # If passer and receiver are within 3m they're the same
                    # physical player tracked under two different IDs.
                    passer_pos = (
                        player_tracks[prev_frame].get(prev_track, {}).get('position_transformed')
                        if prev_frame < len(player_tracks) else None
                    )
                    receiver_pos = (
                        player_tracks[frame_num].get(track_id, {}).get('position_transformed')
                        if frame_num < len(player_tracks) else None
                    )
                    if passer_pos and receiver_pos:
                        dx = passer_pos[0] - receiver_pos[0]
                        dy = passer_pos[1] - receiver_pos[1]
                        if (dx * dx + dy * dy) < 9.0:  # < 3m apart
                            prev_possessor = (track_id, team_id)
                            prev_frame = frame_num
                            continue  # same physical player, skip

                    # Location of ball at pass origin
                    ball_data = ball_tracks[prev_frame].get(1, {})
                    bbox = ball_data.get('bbox', [0, 0, 0, 0])
                    loc = ball_data.get('position_transformed') or (
                        ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    )

                    player_id = self._get_player_id_for_track(
                        prev_track, player_tracks, match_config
                    )
                    team_str = self._get_player_team(player_id, match_config) or str(prev_team)

                    events.append(MatchEvent(
                        event_id=f"evt_{event_counter:04d}",
                        event_type="pass",
                        timestamp_sec=prev_frame / fps,
                        frame_number=prev_frame,
                        player_id=player_id,
                        team_id=team_str,
                        location_x=float(loc[0]) if loc else 0.0,
                        location_y=float(loc[1]) if loc else 0.0,
                        metadata={
                            'from_track': prev_track,
                            'to_track': track_id,
                            'gap_frames': gap,
                        }
                    ))
                    event_counter += 1

                # Possession changed team = tackle / interception
                elif prev_team != team_id:
                    ball_data = ball_tracks[prev_frame].get(1, {})
                    bbox = ball_data.get('bbox', [0, 0, 0, 0])
                    loc = ball_data.get('position_transformed') or (
                        ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    )
                    player_id = self._get_player_id_for_track(
                        track_id, player_tracks, match_config
                    )
                    team_str = self._get_player_team(player_id, match_config) or str(team_id)

                    events.append(MatchEvent(
                        event_id=f"evt_{event_counter:04d}",
                        event_type="possession_change",
                        timestamp_sec=frame_num / fps,
                        frame_number=frame_num,
                        player_id=player_id,
                        team_id=team_str,
                        location_x=float(loc[0]) if loc else 0.0,
                        location_y=float(loc[1]) if loc else 0.0,
                        metadata={'from_team': prev_team, 'to_team': team_id}
                    ))
                    event_counter += 1

            prev_possessor = (track_id, team_id)
            prev_frame = frame_num

        # --- Shot detection ---
        # A shot: ball moves with high speed AND no player has_ball for several frames
        # Uses ball speed from speed data in tracks (if available)
        ball_free_streak = 0
        shot_candidate_start = None

        for frame_num, frame_ball in enumerate(ball_tracks):
            ball_data = frame_ball.get(1, {})
            ball_speed = ball_data.get('speed', 0.0)
            has_possessor = frame_num in possession

            if ball_speed > shot_speed_threshold and not has_possessor:
                ball_free_streak += 1
                if shot_candidate_start is None:
                    shot_candidate_start = frame_num
            else:
                if ball_free_streak >= min_shot_frames and shot_candidate_start is not None:
                    # Who last had the ball before the shot?
                    shooter_frame = shot_candidate_start - 1
                    while shooter_frame >= 0 and shooter_frame not in possession:
                        shooter_frame -= 1

                    if shooter_frame >= 0:
                        shooter_track, shooter_team = possession[shooter_frame]
                        bbox = ball_tracks[shot_candidate_start].get(1, {}).get('bbox', [0,0,0,0])
                        loc = ball_tracks[shot_candidate_start].get(1, {}).get(
                            'position_transformed'
                        ) or (((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))

                        player_id = self._get_player_id_for_track(
                            shooter_track, player_tracks, match_config
                        )
                        team_str = self._get_player_team(player_id, match_config) or str(shooter_team)

                        events.append(MatchEvent(
                            event_id=f"evt_{event_counter:04d}",
                            event_type="shot",
                            timestamp_sec=shot_candidate_start / fps,
                            frame_number=shot_candidate_start,
                            player_id=player_id,
                            team_id=team_str,
                            location_x=float(loc[0]) if loc else 0.0,
                            location_y=float(loc[1]) if loc else 0.0,
                            metadata={
                                'ball_speed_kmh': round(ball_speed, 1),
                                'duration_frames': ball_free_streak,
                            }
                        ))
                        event_counter += 1

                ball_free_streak = 0
                shot_candidate_start = None

        return events

    def _get_player_id_for_track(
        self,
        tracking_id: int,
        player_tracks: List[Dict],
        match_config: Dict[str, Any]
    ) -> str:
        """Look up player_id for a tracking_id using jersey + team color from tracks.

        Returns a player_id consistent with _build_player_id_map() so that
        back-filling event counts into player_metrics always matches keys.
        """
        _team_int_to_id = {
            1: match_config.get('team_home', {}).get('id', 'team_home'),
            2: match_config.get('team_away', {}).get('id', 'team_away'),
        }
        siglip_team_int = None

        for frame_players in player_tracks:
            if tracking_id not in frame_players:
                continue
            pdata = frame_players[tracking_id]

            # Capture SigLIP team int on first occurrence (stable, reliable)
            if siglip_team_int is None:
                siglip_team_int = pdata.get('team')

            # Best case: jersey number detected → try full roster match
            jersey = pdata.get('jersey_number')
            team_color = pdata.get('team_color')
            if jersey is not None and team_color is not None:
                team_id = self._match_team_by_color(team_color, match_config)
                if team_id:
                    pid = self.map_tracking_id_to_player(
                        tracking_id, team_color, jersey, match_config
                    )
                    if pid:
                        return pid

        # Fall back to SigLIP team integer — gives consistent {team_id}_unknown_{id}
        if siglip_team_int is not None:
            team_id = _team_int_to_id.get(siglip_team_int, 'unknown')
            return f"{team_id}_unknown_{tracking_id}"

        return f"unknown_{tracking_id}"

    def generate_visualization(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List],
        metrics: Dict[str, PlayerMetrics],
        events: List[MatchEvent],
        match_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Generate soccer-specific visualization.

        Overlays:
        - Player ellipses (color-coded by team)
        - Jersey numbers (instead of tracking IDs)
        - Speed and distance (below each player)
        - Ball triangle
        - Possession bar (team ball control %)
        - Event markers (goals, shots, etc.)

        Args:
            frames: Original frames
            tracks: Tracking data
            metrics: Player metrics
            events: Detected events
            match_config: Match configuration

        Returns:
            Annotated frames
        """
        # The existing tracker.draw_annotations() handles most of this
        # We're just wrapping it in the new interface

        # For now, return frames as-is
        # The main.py will still use the existing visualization
        # Once we migrate fully, we'll move that logic here

        return frames

    def get_metric_definitions(self) -> Dict[str, str]:
        """
        Get definitions of all soccer metrics.

        Returns:
            Dictionary of metric name -> description
        """
        return {
            # Universal metrics
            'minutes_played': 'Total time player was on the field (minutes)',
            'distance_covered_m': 'Total distance covered by player (meters)',
            'top_speed_kmh': 'Maximum speed achieved (km/h)',
            'avg_speed_kmh': 'Average speed when moving (km/h)',
            'sprints_count': 'Number of sprint sequences (>24 km/h for >1 second)',

            # Soccer-specific metrics
            'possession_time_sec': 'Time player had ball possession (seconds)',
            'passes_attempted': 'Number of passes attempted',
            'passes_completed': 'Number of successful passes',
            'pass_accuracy': 'Percentage of successful passes',
            'shots': 'Total shots attempted',
            'shots_on_target': 'Shots on target',
            'tackles_won': 'Number of successful tackles',
            'tackles_attempted': 'Total tackles attempted',
        }

    # Helper methods

    def _build_player_id_map(
        self,
        player_tracks: List[Dict],
        match_config: Dict[str, Any]
    ) -> Dict[int, str]:
        """
        Build mapping from tracking_id to player_id using jersey numbers and team colors.

        Args:
            player_tracks: Player tracking data
            match_config: Match configuration with roster

        Returns:
            Dictionary mapping tracking_id -> player_id
        """
        player_id_map = {}

        # Build team_id lookup: 1 -> home team id, 2 -> away team id
        _team_int_to_id = {
            1: match_config.get('team_home', {}).get('id', 'team_home'),
            2: match_config.get('team_away', {}).get('id', 'team_away'),
        }

        # Pass 1: collect jersey numbers, team colors, and SigLIP team integers per tracking ID
        tracking_jerseys = {}   # tracking_id -> list of (jersey_number, team_color)
        tracking_colors = {}    # tracking_id -> first team_color seen (for fallback)
        tracking_teams  = {}    # tracking_id -> SigLIP team int (1 or 2) — most reliable source

        for frame_players in player_tracks:
            for tracking_id, player_data in frame_players.items():
                jersey_number = player_data.get('jersey_number')
                team_color    = player_data.get('team_color')
                team_int      = player_data.get('team')  # set by SigLIP (1 or 2)

                # Record SigLIP team assignment (prefer first seen, it's stable)
                if team_int is not None and tracking_id not in tracking_teams:
                    tracking_teams[tracking_id] = team_int

                # Always record any team color we see for this ID
                if team_color is not None and tracking_id not in tracking_colors:
                    tracking_colors[tracking_id] = team_color

                if jersey_number is not None and team_color is not None:
                    if tracking_id not in tracking_jerseys:
                        tracking_jerseys[tracking_id] = []
                    tracking_jerseys[tracking_id].append((jersey_number, team_color))

        # Pass 2: collect ALL unique tracking IDs that appeared in any frame
        all_tracking_ids = set()
        for frame_players in player_tracks:
            all_tracking_ids.update(frame_players.keys())

        # Pass 3: assign player IDs
        for tracking_id in all_tracking_ids:
            jersey_colors = tracking_jerseys.get(tracking_id, [])
            siglip_team = tracking_teams.get(tracking_id)  # 1 or 2, most reliable source

            if jersey_colors and siglip_team is not None:
                # Jersey detected by OCR + SigLIP team available: use SigLIP for team
                # (color matching is unreliable — jersey colors vary with lighting)
                jersey_number = max(set([j for j, c in jersey_colors]), key=[j for j, c in jersey_colors].count)
                team_id = _team_int_to_id.get(siglip_team, 'unknown')
                # Find player in roster by jersey number + team
                player_id = None
                for team_key in ['team_home', 'team_away']:
                    if match_config.get(team_key, {}).get('id') == team_id:
                        for p in match_config[team_key].get('players', []):
                            if int(p.get('jersey_number', -1)) == int(jersey_number):
                                player_id = p['player_id']
                                break
                        break
                if player_id is None:
                    player_id = f"{team_id}_unknown_{tracking_id}"
            elif jersey_colors:
                # OCR jersey but no SigLIP team — fall back to color matching
                jersey_number = max(set([j for j, c in jersey_colors]), key=[j for j, c in jersey_colors].count)
                team_color = jersey_colors[0][1]
                player_id = self.map_tracking_id_to_player(
                    tracking_id, team_color, jersey_number, match_config
                )
            elif siglip_team is not None:
                # No OCR jersey — use SigLIP team integer directly
                team_id = _team_int_to_id.get(siglip_team, 'unknown')
                player_id = f"{team_id}_unknown_{tracking_id}"
            elif tracking_id in tracking_colors:
                # Fallback: color-based team guess
                team_color = tracking_colors[tracking_id]
                team_id = self._match_team_by_color(team_color, match_config) or 'unknown'
                player_id = f"{team_id}_unknown_{tracking_id}"
            else:
                player_id = f"unknown_{tracking_id}"

            player_id_map[tracking_id] = player_id

        return player_id_map

    def _count_sprint_sequences(self, sprint_frames: List[bool], fps: float) -> int:
        """
        Count number of sprint sequences (consecutive frames above sprint threshold).

        A sprint must last at least 1 second to count.

        Args:
            sprint_frames: Boolean list indicating sprint frames
            fps: Video frame rate

        Returns:
            Number of sprint sequences
        """
        min_sprint_frames = int(fps)  # 1 second
        sprint_count = 0
        current_sprint_length = 0

        for is_sprint in sprint_frames:
            if is_sprint:
                current_sprint_length += 1
            else:
                if current_sprint_length >= min_sprint_frames:
                    sprint_count += 1
                current_sprint_length = 0

        # Check final sprint
        if current_sprint_length >= min_sprint_frames:
            sprint_count += 1

        return sprint_count

    def _get_player_name(self, player_id: str, match_config: Dict[str, Any]) -> str:
        """Get player name from match config"""
        for team_key in ['team_home', 'team_away']:
            if team_key not in match_config:
                continue

            for player in match_config[team_key].get('players', []):
                if player.get('player_id') == player_id:
                    return player.get('name', player_id)

        return player_id

    def _get_player_team(self, player_id: str, match_config: Dict[str, Any]) -> str:
        """Get team ID from match config"""
        for team_key in ['team_home', 'team_away']:
            if team_key not in match_config:
                continue

            team_id = match_config[team_key].get('id', '')

            # Direct roster match
            for player in match_config[team_key].get('players', []):
                if player.get('player_id') == player_id:
                    return team_id

            # Prefix match for auto-generated IDs like "chicago_fc_united_unknown_12345"
            if team_id and player_id.startswith(f"{team_id}_"):
                return team_id

        return 'unknown'
