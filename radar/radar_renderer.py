"""
Radar Bird's-Eye View Renderer

Generates a live overhead pitch mini-map using roboflow/sports annotators.
Composited as a semi-transparent overlay at the bottom-center of each frame.

Two modes:
  - Keypoint mode:  Uses a pitch detection YOLO model (models/pitch_detection.pt)
                    to detect 32 field keypoints per frame, builds a homography,
                    and projects player positions accurately onto the pitch diagram.
  - Fallback mode:  Uses linear pixel-to-cm scaling when no pitch model is present.
                    Less geometrically precise but works out of the box.

To enable keypoint mode, download the YOLOv8 pitch detection model from:
  https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi
  and place it at: models/pitch_detection.pt
"""

import os
import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer as RFViewTransformer
import supervision as sv


class RadarRenderer:
    """
    Renders a bird's-eye radar overlay onto each video frame.

    Usage:
        renderer = RadarRenderer(pitch_model_path='models/pitch_detection.pt')

        for frame_num, frame in enumerate(output_frames):
            if renderer.keypoint_mode:
                renderer.update_frame_transformer(video_frames[frame_num])
            output_frames[frame_num] = renderer.render_radar(
                frame=frame,
                player_dict=tracks['players'][frame_num],
                team_colors_bgr=team_colors,
                ball_dict=tracks['ball'][frame_num],
            )
    """

    # Radar overlay occupies this fraction of frame height
    OVERLAY_HEIGHT_RATIO = 0.25
    # Alpha blend: 0 = fully transparent, 1 = fully opaque
    OVERLAY_ALPHA = 0.5
    # Scale factor applied to SoccerPitchConfiguration cm coordinates for rendering
    # 0.065 → ~780px wide for a 12000cm pitch (comfortable mini-map size)
    PITCH_SCALE = 0.065

    def __init__(self, pitch_model_path: str = None, device: str = 'cpu'):
        self.config = SoccerPitchConfiguration()
        self.device = device
        self._current_transformer = None  # RFViewTransformer for current frame

        # Decide mode
        self.keypoint_mode = (
            pitch_model_path is not None and os.path.exists(pitch_model_path)
        )

        if self.keypoint_mode:
            from ultralytics import YOLO
            self._pitch_model = YOLO(pitch_model_path)
            print(f"  RadarRenderer: keypoint mode ({pitch_model_path})")
        else:
            self._pitch_model = None
            if pitch_model_path:
                print(f"  RadarRenderer: fallback mode (pitch model not found at {pitch_model_path})")
            else:
                print("  RadarRenderer: fallback mode (no pitch model path provided)")

        # Pre-render the empty pitch background once — reused every frame
        self._pitch_bg = self._build_pitch_background()

    def _build_pitch_background(self) -> np.ndarray:
        """Draw the static pitch diagram (green field + white lines)."""
        pitch = draw_pitch(
            config=self.config,
            background_color=sv.Color.from_hex('#1a7a1a'),
            line_color=sv.Color.WHITE,
            scale=self.PITCH_SCALE,
        )
        # draw_pitch may return PIL Image — convert to BGR numpy if needed
        if not isinstance(pitch, np.ndarray):
            pitch = np.array(pitch)
        if pitch.ndim == 3 and pitch.shape[2] == 3:
            pitch = cv2.cvtColor(pitch, cv2.COLOR_RGB2BGR)
        return pitch

    # ------------------------------------------------------------------ #
    #  Keypoint mode helpers
    # ------------------------------------------------------------------ #

    def update_frame_transformer(self, frame: np.ndarray):
        """
        Detect pitch keypoints in the current frame and build a homography.
        Call this once per frame when in keypoint mode.
        """
        if not self.keypoint_mode:
            return

        result = self._pitch_model.predict(frame, conf=0.5, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            self._current_transformer = None
            return

        kp_xy = result.keypoints.xy[0].cpu().numpy()          # (32, 2) pixel coords
        kp_conf = result.keypoints.conf[0].cpu().numpy()      # (32,) confidences

        mask = kp_conf > 0.5
        if mask.sum() < 4:
            self._current_transformer = None
            return

        source = kp_xy[mask]                                   # (M, 2) pixels
        target = np.array(self.config.vertices, dtype=np.float32)[mask]  # (M, 2) cm

        try:
            self._current_transformer = RFViewTransformer(
                source=source, target=target
            )
        except Exception:
            self._current_transformer = None

    # ------------------------------------------------------------------ #
    #  Position transformation
    # ------------------------------------------------------------------ #

    def _pixel_to_pitch(self, position_adjusted, frame_w: int, frame_h: int):
        """
        Convert a pixel position (camera-motion-corrected) to pitch cm coordinates.

        In keypoint mode: uses the homography built from detected field keypoints.
        In fallback mode: linear scale from pixel → cm using full-pitch assumption.

        Returns [x_cm, y_cm] or None if the position is outside valid bounds.
        """
        if position_adjusted is None:
            return None

        x_px, y_px = float(position_adjusted[0]), float(position_adjusted[1])

        if self.keypoint_mode and self._current_transformer is not None:
            pt = np.array([[x_px, y_px]], dtype=np.float32)
            transformed = self._current_transformer.transform_points(pt)
            if transformed is None or len(transformed) == 0:
                return None
            x_cm, y_cm = float(transformed[0][0]), float(transformed[0][1])
        else:
            # Fallback: assume the full pitch width/length fits in the frame
            x_cm = x_px * self.config.width / max(frame_w, 1)
            y_cm = y_px * self.config.length / max(frame_h, 1)

        # Clamp to pitch boundaries
        x_cm = max(0.0, min(x_cm, float(self.config.width)))
        y_cm = max(0.0, min(y_cm, float(self.config.length)))
        return [x_cm, y_cm]

    # ------------------------------------------------------------------ #
    #  Main render
    # ------------------------------------------------------------------ #

    def render_radar(
        self,
        frame: np.ndarray,
        player_dict: dict,
        team_colors_bgr: dict,
        ball_dict: dict = None,
    ) -> np.ndarray:
        """
        Render the radar for one frame and composite it onto the frame.

        Args:
            frame:            BGR frame (H, W, 3) — not modified in place
            player_dict:      tracks['players'][frame_num]
            team_colors_bgr:  {1: (B, G, R), 2: (B, G, R)}
            ball_dict:        tracks['ball'][frame_num]  (optional)

        Returns:
            New BGR frame with radar overlay at bottom-center.
        """
        frame_h, frame_w = frame.shape[:2]
        pitch_img = self._pitch_bg.copy()

        # Separate players by team
        team_xy = {1: [], 2: []}

        for track_id, player in player_dict.items():
            pos = player.get('position_adjusted')
            xy = self._pixel_to_pitch(pos, frame_w, frame_h)
            if xy is None:
                continue
            team_id = player.get('team', 1)
            if team_id in team_xy:
                team_xy[team_id].append(xy)

        # Draw each team's players
        for team_id, positions in team_xy.items():
            if not positions:
                continue
            bgr = team_colors_bgr.get(team_id, (200, 200, 200))
            color = sv.Color(r=int(bgr[2]), g=int(bgr[1]), b=int(bgr[0]))
            pitch_img = draw_points_on_pitch(
                config=self.config,
                xy=np.array(positions, dtype=np.float32),
                face_color=color,
                edge_color=sv.Color.WHITE,
                radius=16,
                thickness=2,
                pitch=pitch_img,
                scale=self.PITCH_SCALE,
            )
            if not isinstance(pitch_img, np.ndarray):
                pitch_img = np.array(pitch_img)
            if pitch_img.ndim == 3 and pitch_img.shape[2] == 3:
                pitch_img = cv2.cvtColor(pitch_img, cv2.COLOR_RGB2BGR)

        # Draw ball
        if ball_dict:
            for _, ball in ball_dict.items():
                pos = ball.get('position_adjusted')
                xy = self._pixel_to_pitch(pos, frame_w, frame_h)
                if xy is None:
                    continue
                pitch_img = draw_points_on_pitch(
                    config=self.config,
                    xy=np.array([xy], dtype=np.float32),
                    face_color=sv.Color.WHITE,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    thickness=2,
                    pitch=pitch_img,
                    scale=self.PITCH_SCALE,
                )
                if not isinstance(pitch_img, np.ndarray):
                    pitch_img = np.array(pitch_img)
                if pitch_img.ndim == 3 and pitch_img.shape[2] == 3:
                    pitch_img = cv2.cvtColor(pitch_img, cv2.COLOR_RGB2BGR)

        return self._overlay_radar(frame, pitch_img)

    # ------------------------------------------------------------------ #
    #  Compositing
    # ------------------------------------------------------------------ #

    def _overlay_radar(self, frame: np.ndarray, radar_img: np.ndarray) -> np.ndarray:
        """
        Resize radar and blend it at the bottom-center of the frame.
        """
        frame_h, frame_w = frame.shape[:2]

        # Target height = 25% of frame; maintain aspect ratio
        target_h = max(1, int(frame_h * self.OVERLAY_HEIGHT_RATIO))
        radar_h, radar_w = radar_img.shape[:2]
        if radar_h == 0 or radar_w == 0:
            return frame

        target_w = int(target_h * radar_w / radar_h)
        if target_w == 0:
            return frame

        radar_resized = cv2.resize(
            radar_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # Position: bottom-center, 10px padding from bottom edge
        x_offset = max(0, (frame_w - target_w) // 2)
        y_offset = max(0, frame_h - target_h - 10)

        # Clamp to frame bounds
        x_end = min(frame_w, x_offset + target_w)
        y_end = min(frame_h, y_offset + target_h)
        actual_w = x_end - x_offset
        actual_h = y_end - y_offset

        output = frame.copy()
        roi = output[y_offset:y_end, x_offset:x_end]
        radar_crop = radar_resized[:actual_h, :actual_w]

        cv2.addWeighted(
            radar_crop, self.OVERLAY_ALPHA,
            roi, 1.0 - self.OVERLAY_ALPHA,
            0, roi
        )
        output[y_offset:y_end, x_offset:x_end] = roi
        return output
