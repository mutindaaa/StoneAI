import numpy as np
import cv2
import json
import os

class ViewTransformer():
    # Default overhead tactical camera vertices expressed as fractions of frame
    # dimensions so they scale correctly to ANY resolution (720p, 1080p, 4K, etc.).
    # Values derived from a standard half-pitch overhead view; override with a
    # calibration JSON for different camera setups.
    _TACTICAL_VERTICES_REL = np.array([
        [0.057, 0.958],   # bottom-left
        [0.138, 0.255],   # top-left
        [0.474, 0.241],   # top-right
        [0.854, 0.847],   # bottom-right
    ])

    def __init__(self, calibration_path=None, frame_width=None, frame_height=None,
                 visible_field_meters=40.0):
        """
        Initialize ViewTransformer with optional calibration file.

        For broadcast footage (panning/zooming camera), perspective transform
        is unreliable. Instead we use a pixel-to-meter scale estimate based on
        visible field area. Set frame_width/frame_height to enable this mode.

        Args:
            calibration_path:      Path to JSON calibration file (optional).
            frame_width:           Video frame width in pixels.
            frame_height:          Video frame height in pixels.
            visible_field_meters:  How many metres of field are visible across the
                                   frame width. Used by the pixel-fallback only.
                                   Soccer broadcast ≈ 40 m; basketball full-court ≈ 28 m.
        """
        court_width = 68    # Standard soccer field width (meters)
        court_length = 23.32

        self.use_perspective = False
        self.meters_per_pixel = None
        self._visible_field_meters = visible_field_meters

        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path, 'r') as f:
                calibration = json.load(f)

            self.pixel_vertices = np.array(calibration['pixel_vertices'])
            self.target_vertices = np.array(calibration['target_vertices'])

            if 'field_dimensions' in calibration:
                court_width = calibration['field_dimensions']['width']
                court_length = calibration['field_dimensions']['height']

            # Validate calibration vertices fit within the actual frame
            if frame_width and frame_height:
                max_x = self.pixel_vertices[:, 0].max()
                max_y = self.pixel_vertices[:, 1].max()
                if max_x > frame_width * 1.1 or max_y > frame_height * 1.1:
                    print(f"  Warning: Calibration ({max_x:.0f},{max_y:.0f}) exceeds frame ({frame_width}x{frame_height})")
                    print("  Falling back to pixel-based distance estimation")
                    self._setup_pixel_fallback(frame_width)
                    return

            self.pixel_vertices = self.pixel_vertices.astype(np.float32)
            self.target_vertices = self.target_vertices.astype(np.float32)
            self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
            self.use_perspective = True
            print(f"  Loaded calibration: {court_length}m x {court_width}m")

        else:
            # No calibration file — use pixel-based distance estimation.
            # This covers every player on the frame regardless of position, whereas
            # a perspective polygon would silently drop players outside its bounds.
            if frame_width:
                self._setup_pixel_fallback(frame_width)
            else:
                # No frame info at all — can't estimate distances
                self.use_perspective = False
                self.meters_per_pixel = None

    def _setup_pixel_fallback(self, frame_width):
        """
        Pixel-based distance estimation for broadcast / unknown-angle footage.

        Uses visible_field_meters (set in __init__) to compute metres-per-pixel.
        More accurate values can be set via the 'visible_field_meters' config key:
          - Soccer broadcast camera:      ~40 m
          - Soccer tactical full-pitch:   ~68 m
          - Basketball full-court:        ~28 m
          - Basketball half-court:        ~15 m
        """
        self.meters_per_pixel = self._visible_field_meters / max(frame_width, 1)
        self.use_perspective = False
        print(f"  Pixel-based estimation: {self.meters_per_pixel:.4f} m/px "
              f"(~{self._visible_field_meters:.0f}m visible at {frame_width}px wide)")

    def transform_point(self, point):
        if self.use_perspective:
            p = (int(point[0]), int(point[1]))
            is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
            if not is_inside:
                return None
            reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)  # type: ignore[call-overload]
            transform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
            return transform_point.reshape(-1, 2)
        else:
            # Scale pixel coords to approximate meters
            if self.meters_per_pixel is None:
                return None
            return np.array([[point[0] * self.meters_per_pixel,
                               point[1] * self.meters_per_pixel]])

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
