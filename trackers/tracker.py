from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path, ball_model_path=None):
        self.model = YOLO(model_path)
        # Optional dedicated ball detection model (e.g. basketball-specific)
        self.ball_model = YOLO(ball_model_path) if ball_model_path else None
        # BoT-SORT: appearance Re-ID built in — handles occlusions and camera cuts
        # Falls back to ByteTrack motion-only matching if Re-ID embeddings unavailable
        self.tracker = sv.ByteTrack()   # kept for ball/referee (no Re-ID needed)
        self._use_botsort = True        # use ultralytics BoT-SORT for players

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Fill missing ball frames and smooth jitter.

        For wide-angle/overhead cameras (Veo, BePro) the ball is ~5-10px at
        1280x720, leading to many missed detections and jittery positions.

        Steps:
        1. Linear interpolation across missing frames
        2. Back-fill any leading gaps
        3. Rolling median (window=7) to smooth positional jitter without lag
        """
        raw_bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(raw_bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Step 1: Replace empty rows with NaN so pandas interpolates them
        df = df.replace(0, np.nan)
        df[['x1', 'y1', 'x2', 'y2']] = df[['x1', 'y1', 'x2', 'y2']].interpolate(
            method='linear', limit_direction='both'
        )
        df = df.bfill().ffill()

        # Step 2: Rolling median to smooth jitter (wide-angle cameras bounce ~3-8px)
        # window=7 → ~0.23 s at 30fps, negligible lag for analysis purposes
        smooth_window = 7
        for col in ['x1', 'y1', 'x2', 'y2']:
            df[col] = df[col].rolling(window=smooth_window, center=True, min_periods=1).median()

        result = [{1: {"bbox": row}} for row in df.to_numpy().tolist()]
        return result

    def detect_frames(self, frames):
        """Run detection only (no tracking) — used for ball/referee."""
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1, verbose=False)
            detections += detections_batch
        return detections

    def _detect_ball_frames(self, frames):
        """
        Run dedicated ball model detection (basketball-specific).

        Used when self.ball_model is set. Returns the same list-of-Results
        format as detect_frames() so callers can swap in this path transparently.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.ball_model.predict(
                frames[i:i + batch_size], conf=0.1, verbose=False
            )
            detections += batch
        return detections

    def track_frames(self, frames):
        """
        Run BoT-SORT tracking via ultralytics model.track().
        Processes frame-by-frame (persist=True keeps state between calls).
        Returns list of ultralytics Results objects with .boxes.id populated.
        """
        # Resolve yaml path relative to this file so it works from any cwd
        _yaml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "botsort_custom.yaml")
        if not os.path.exists(_yaml):
            _yaml = "botsort.yaml"  # fallback to ultralytics default

        results = []
        for frame in frames:
            result = self.model.track(
                frame,
                tracker=_yaml,
                persist=True,
                conf=0.3,   # raised from 0.1 — eliminates ghost detections that cause fragmentation
                iou=0.5,    # non-max suppression IOU threshold
                verbose=False,
            )
            results.append(result[0])
        return results

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {"players": [], "referees": [], "ball": []}

        # BoT-SORT tracking for all classes (players, referees, goalkeepers)
        tracked_results = self.track_frames(frames)

        # Plain detection for ball (tiny object — tracker IDs not needed, just position)
        # Use dedicated ball model when available (basketball), else fall back to player model
        if self.ball_model is not None:
            ball_det_results = self._detect_ball_frames(frames)
        else:
            ball_det_results = self.detect_frames(frames)

        for frame_num, tracked in enumerate(tracked_results):
            detected = ball_det_results[frame_num]
            cls_names = tracked.names

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # --- Players & referees from BoT-SORT ---
            boxes = tracked.boxes
            if boxes is not None and boxes.id is not None:
                for box, cls_id, track_id in zip(
                    boxes.xyxy.tolist(),
                    boxes.cls.tolist(),
                    boxes.id.tolist(),
                ):
                    cls_id = int(cls_id)
                    track_id = int(track_id)
                    class_name = cls_names[cls_id]

                    # Merge goalkeeper into player class
                    if class_name == "goalkeeper":
                        class_name = "player"

                    if class_name == "player":
                        tracks["players"][frame_num][track_id] = {"bbox": box}
                    elif class_name == "referee":
                        tracks["referees"][frame_num][track_id] = {"bbox": box}

            # --- Ball from detection (always track_id=1) ---
            det_sv = sv.Detections.from_ultralytics(detected)
            ball_cls_names = self.ball_model.names if self.ball_model is not None else cls_names
            if self.ball_model is not None:
                # Dedicated ball model: filter for any class whose name contains "ball"
                # Works for custom models ("ball"/"Ball") and yolov8n COCO ("sports ball")
                best_conf, best_bbox = -1.0, None
                for i, cls_id in enumerate(det_sv.class_id):
                    if "ball" in ball_cls_names[int(cls_id)].lower():
                        conf = float(det_sv.confidence[i]) if det_sv.confidence is not None else 1.0
                        if conf > best_conf:
                            best_conf = conf
                            best_bbox = det_sv.xyxy[i].tolist()
                if best_bbox is not None:
                    tracks["ball"][frame_num][1] = {"bbox": best_bbox}
            else:
                # Player model: filter by class name "ball"
                for i, cls_id in enumerate(det_sv.class_id):
                    if cls_names[cls_id] == "ball":
                        tracks["ball"][frame_num][1] = {"bbox": det_sv.xyxy[i].tolist()}
                        break  # take first/best ball detection per frame

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def get_object_tracks_chunked(
        self,
        video_path: str,
        scale_factor: float = 1.0,
        chunk_size: int = 500,
        stub_path: str = None,
    ):
        """
        Memory-efficient tracking for long videos (e.g. 90-minute matches).

        Processes chunk_size frames at a time so RAM stays flat regardless of
        video length. BoT-SORT state persists across chunks via persist=True.
        """
        if stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        tracks = {"players": [], "referees": [], "ball": []}
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        print(f"  Chunked tracking: {total} frames in chunks of {chunk_size}")

        while True:
            chunk = []
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                if scale_factor != 1.0:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(
                        frame,
                        (int(w * scale_factor), int(h * scale_factor)),
                        interpolation=cv2.INTER_AREA,
                    )
                chunk.append(frame)

            if not chunk:
                break

            tracked_results = self.track_frames(chunk)
            if self.ball_model is not None:
                ball_det_results = self._detect_ball_frames(chunk)
            else:
                ball_det_results = self.detect_frames(chunk)

            for tracked, detected in zip(tracked_results, ball_det_results):
                cls_names = tracked.names

                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                frame_idx = len(tracks["players"]) - 1

                boxes = tracked.boxes
                if boxes is not None and boxes.id is not None:
                    for box, cls_id, track_id in zip(
                        boxes.xyxy.tolist(),
                        boxes.cls.tolist(),
                        boxes.id.tolist(),
                    ):
                        cls_id     = int(cls_id)
                        track_id   = int(track_id)
                        class_name = cls_names[cls_id]
                        if class_name == "goalkeeper":
                            class_name = "player"
                        if class_name == "player":
                            tracks["players"][frame_idx][track_id] = {"bbox": box}
                        elif class_name == "referee":
                            tracks["referees"][frame_idx][track_id] = {"bbox": box}

                det_sv = sv.Detections.from_ultralytics(detected)
                ball_cls_names = self.ball_model.names if self.ball_model is not None else cls_names
                if self.ball_model is not None:
                    best_conf, best_bbox = -1.0, None
                    for i, cls_id in enumerate(det_sv.class_id):
                        if "ball" in ball_cls_names[int(cls_id)].lower():
                            conf = float(det_sv.confidence[i]) if det_sv.confidence is not None else 1.0
                            if conf > best_conf:
                                best_conf = conf
                                best_bbox = det_sv.xyxy[i].tolist()
                    if best_bbox is not None:
                        tracks["ball"][frame_idx][1] = {"bbox": best_bbox}
                else:
                    for i, cls_id in enumerate(det_sv.class_id):
                        if cls_names[int(cls_id)] == "ball":
                            tracks["ball"][frame_idx][1] = {"bbox": det_sv.xyxy[i].tolist()}
                            break

            processed += len(chunk)
            print(f"  {processed}/{total} frames tracked ({processed/total*100:.0f}%)", end="\r")

        cap.release()
        print()

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"  Tracks cached to {stub_path}")

        return tracks

    def draw_ellipse(self,frame,bbox,color,track_id=None,jersey_number=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Smaller, cleaner ellipse
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width*0.7), int(0.25*width)),  # 30% smaller
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Smaller ID rectangle
        rectangle_width = 28
        rectangle_height=16
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +12
        y2_rect = (y2+ rectangle_height//2) +12

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)

            # Use jersey number if available, otherwise use track_id
            display_text = str(jersey_number) if jersey_number else str(track_id)

            x1_text = x1_rect+8
            if len(display_text) > 2:
                x1_text -=6

            # Smaller, cleaner text
            cv2.putText(
                frame,
                display_text,
                (int(x1_text),int(y1_rect+12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,  # Smaller font
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        # Smaller, cleaner triangle
        triangle_points = np.array([
            [x,y],
            [x-6,y-12],  # 40% smaller
            [x+6,y-12],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 1)  # Thinner border

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, match_config=None):
        h, w = frame.shape[:2]

        # Overlay box — fully relative to frame dimensions
        pad   = max(6, int(w * 0.006))
        box_w = int(w * 0.29)
        box_h = int(h * 0.11)
        x1    = w - box_w - pad
        x2    = w - pad
        y1    = pad
        y2    = y1 + box_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total  = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total if total > 0 else 0.5
        team_2 = team_2_num_frames / total if total > 0 else 0.5

        # Use actual team names from config when available
        cfg = match_config or {}
        name_1 = cfg.get('team_home', {}).get('name', 'Home')
        name_2 = cfg.get('team_away', {}).get('name', 'Away')
        # Truncate long names so they fit in the box
        max_chars = max(8, box_w // 12)
        name_1 = name_1[:max_chars]
        name_2 = name_2[:max_chars]

        font_scale = max(0.4, w / 2400)
        line_gap   = int(box_h * 0.4)
        tx = x1 + pad
        cv2.putText(frame, f"{name_1}: {team_1*100:.0f}%", (tx, y1 + line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2)
        cv2.putText(frame, f"{name_2}: {team_2*100:.0f}%", (tx, y1 + line_gap * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2)

        return frame

    def annotate_frame(self, frame, frame_num, tracks, team_ball_control, match_config=None):
        """
        Annotate a single frame — streaming-friendly alternative to draw_annotations().
        Call this inside a VideoCapture loop to avoid loading all frames into RAM.
        """
        frame = frame.copy()
        player_dict  = tracks["players"][frame_num]
        ball_dict    = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        for track_id, player in player_dict.items():
            color      = player.get("team_color", (0, 0, 255))
            # Display priority: OCR jersey_number > position_number (1-11) > raw track_id
            jersey_num = player.get("jersey_number") or player.get("position_number")
            frame = self.draw_ellipse(frame, player["bbox"], color, track_id, jersey_num)
            if player.get("has_ball", False):
                frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

        for _, referee in referee_dict.items():
            frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        for _, ball in ball_dict.items():
            frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, match_config)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, match_config=None):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                jersey_number = player.get("jersey_number") or player.get("position_number")
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id, jersey_number)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, match_config)

            output_video_frames.append(frame)

        return output_video_frames