"""
Sport-Agnostic Base Class for Multi-Sport Analytics Platform

This abstract class defines the interface that all sport-specific analyzers must implement.
It enables the platform to support multiple sports (soccer, basketball, football, etc.)
while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np


@dataclass
class FieldCalibration:
    """Field/court calibration data for perspective transformation"""
    transform_matrix: np.ndarray
    inverse_transform_matrix: np.ndarray
    field_dimensions: Tuple[float, float]  # (length, width) in meters
    pixel_to_meter_ratio: float
    keypoints: Dict[str, Tuple[int, int]]  # Named keypoints (e.g., "corner_top_left")


@dataclass
class PlayerMetrics:
    """Player performance metrics for a single match"""
    player_id: str
    player_name: str
    team_id: str

    # Universal metrics (all sports)
    minutes_played: float
    distance_covered_m: float
    top_speed_kmh: float
    avg_speed_kmh: float
    sprints_count: int

    # Sport-specific metrics (stored as dict)
    sport_metrics: Dict[str, Any]


@dataclass
class MatchEvent:
    """Represents a significant event in the match"""
    event_id: str
    event_type: str  # "goal", "shot", "pass", "basket", etc.
    timestamp_sec: float
    frame_number: int
    player_id: str
    team_id: str
    location_x: float  # Field/court coordinates
    location_y: float
    metadata: Dict[str, Any]


class SportAnalyzer(ABC):
    """
    Abstract base class for sport-specific analyzers.

    Each sport (soccer, basketball, football, etc.) extends this class
    and implements its specific logic for field calibration, metrics
    calculation, event detection, and visualization.
    """

    def __init__(self, sport_name: str, field_dimensions: Tuple[float, float]):
        """
        Initialize sport analyzer

        Args:
            sport_name: Name of the sport (e.g., "soccer", "basketball")
            field_dimensions: (length, width) in meters
        """
        self.sport_name = sport_name
        self.field_dimensions = field_dimensions

    @abstractmethod
    def calibrate_field(self, frame: np.ndarray, manual_mode: bool = False) -> FieldCalibration:
        """
        Detect field/court boundaries and calculate perspective transformation.

        This is critical for accurate distance/speed calculations.

        Args:
            frame: Sample video frame showing the field/court
            manual_mode: If True, allow user to manually select keypoints

        Returns:
            FieldCalibration object with transformation matrices
        """
        pass

    @abstractmethod
    def calculate_metrics(
        self,
        tracks: Dict[str, List],
        match_config: Dict[str, Any],
        field_calibration: FieldCalibration,
        fps: float
    ) -> Dict[str, PlayerMetrics]:
        """
        Calculate sport-specific performance metrics for all players.

        Args:
            tracks: Object tracking data from core engine
            match_config: Match configuration (teams, players, etc.)
            field_calibration: Field calibration data
            fps: Video frame rate

        Returns:
            Dictionary mapping player_id to PlayerMetrics
        """
        pass

    @abstractmethod
    def detect_events(
        self,
        tracks: Dict[str, List],
        frames: List[np.ndarray],
        match_config: Dict[str, Any]
    ) -> List[MatchEvent]:
        """
        Detect sport-specific events (goals, shots, baskets, etc.)

        Args:
            tracks: Object tracking data
            frames: Video frames
            match_config: Match configuration

        Returns:
            List of detected events
        """
        pass

    @abstractmethod
    def generate_visualization(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List],
        metrics: Dict[str, PlayerMetrics],
        events: List[MatchEvent],
        match_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Generate annotated video frames with sport-specific overlays.

        Args:
            frames: Original video frames
            tracks: Object tracking data
            metrics: Calculated player metrics
            events: Detected events
            match_config: Match configuration

        Returns:
            List of annotated frames ready for video output
        """
        pass

    @abstractmethod
    def get_metric_definitions(self) -> Dict[str, str]:
        """
        Return definitions of all sport-specific metrics.

        Returns:
            Dictionary mapping metric name to description
        """
        pass

    # Common utility methods (implemented here, used by all sports)

    def map_tracking_id_to_player(
        self,
        tracking_id: int,
        team_color: Tuple[int, int, int],
        jersey_number: int,
        match_config: Dict[str, Any]
    ) -> str:
        """
        Map a tracking ID to actual player identity using jersey number and team.

        This solves the "tracking confusion" problem by linking anonymous
        tracking IDs to real player identities.

        Args:
            tracking_id: ByteTrack assigned ID
            team_color: Detected team color (BGR)
            jersey_number: Detected jersey number (if available)
            match_config: Match configuration with player roster

        Returns:
            player_id from match_config
        """
        # Determine which team based on color
        team_id = self._match_team_by_color(team_color, match_config)

        if team_id is None:
            return f"unknown_{tracking_id}"

        # Find the team roster — match_config uses 'team_home'/'team_away' keys, not the team id
        team_players = []
        for team_key in ['team_home', 'team_away']:
            if match_config.get(team_key, {}).get('id') == team_id:
                team_players = match_config[team_key].get('players', [])
                break

        # Match jersey number — OCR returns strings, config has ints; normalise both
        try:
            jersey_int = int(jersey_number)
        except (TypeError, ValueError):
            return f"{team_id}_unknown_{tracking_id}"

        for player in team_players:
            if int(player.get('jersey_number', -1)) == jersey_int:
                return player['player_id']

        # Jersey number detected but not on roster (sub, misread, etc.)
        return f"{team_id}_unknown_{tracking_id}"

    def _match_team_by_color(
        self,
        detected_color: Tuple[int, int, int],
        match_config: Dict[str, Any]
    ) -> str:
        """
        Match detected team color to configured team.

        Args:
            detected_color: BGR color tuple
            match_config: Match configuration

        Returns:
            team_id or None
        """
        # Simple color distance matching
        # In production, use more sophisticated color matching
        min_distance = float('inf')
        matched_team = None

        for team_key in ['team_home', 'team_away']:
            if team_key not in match_config:
                continue

            team_color_name = match_config[team_key].get('color_primary', '').lower()
            team_color_bgr = self._color_name_to_bgr(team_color_name)

            distance = np.linalg.norm(np.array(detected_color) - np.array(team_color_bgr))

            if distance < min_distance:
                min_distance = distance
                matched_team = match_config[team_key]['id']

        return matched_team

    def _color_name_to_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """
        Convert color name to BGR tuple.

        Args:
            color_name: Color name (e.g., "red", "blue")

        Returns:
            BGR tuple
        """
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
        }
        return color_map.get(color_name, (128, 128, 128))
