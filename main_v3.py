"""
Multi-Sport Analytics Platform - Main Processing Script v3.0

This is the production version that uses:
- Match configuration files (user provides team/player data)
- Sport-agnostic architecture (easily expand to other sports)
- Performance optimizations (GPU, downscaling, efficient rendering)
- Accurate player identification (no more tracking confusion)
"""

import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

from utils import read_video, save_video, get_video_properties
from trackers import Tracker
from team_assigner import TeamAssigner, SigLIPTeamClassifier, assign_position_numbers
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from jersey_number_detector import JerseyNumberDetector
from rf_analytics import SoccerAnalyzer, BasketballAnalyzer


class MatchProcessor:
    """Processes a single match video with configuration"""

    def __init__(self, config_path: str):
        """
        Initialize match processor with configuration file.

        Args:
            config_path: Path to match configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.sport = self.config.get('sport', 'soccer')

        # Initialize sport-specific analyzer
        if self.sport == 'soccer':
            self.sport_analyzer = SoccerAnalyzer()
        elif self.sport == 'basketball':
            self.sport_analyzer = BasketballAnalyzer()
        else:
            raise ValueError(f"Sport '{self.sport}' not yet supported. Available: soccer, basketball")

        # Performance options
        self.use_gpu = self.config.get('processing_options', {}).get('use_gpu', True)
        self.downscale = self.config.get('processing_options', {}).get('downscale_for_processing', False)
        self.target_height = self.config.get('processing_options', {}).get('target_processing_height', 720)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load match configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def process(self, progress_callback=None):
        """
        Main processing pipeline.

        Args:
            progress_callback: Optional callable(step: str, fraction: float) for
                               reporting progress to an external caller (e.g. API server).

        Steps:
        1. Load video and get properties
        2. Run object detection and tracking
        3. Assign teams using color clustering
        4. Map tracking IDs to real player identities (using jersey numbers + team colors)
        5. Calculate sport-specific metrics
        6. Detect sport-specific events
        7. Generate annotated video
        8. Save output and metrics
        """
        def _progress(step: str, fraction: float):
            if progress_callback:
                progress_callback(step, fraction)
        print(f"\n{'='*60}")
        print(f"Multi-Sport Analytics Platform v3.0")
        print(f"{'='*60}")
        print(f"Match ID: {self.config.get('match_id', 'auto')}")
        print(f"Sport: {self.sport.upper()}")
        print(f"Competition: {self.config.get('competition', 'Unknown Competition')}")
        home_name = self.config.get('team_home', {}).get('name', 'Team A')
        away_name = self.config.get('team_away', {}).get('name', 'Team B')
        print(f"Teams: {home_name} vs {away_name}")
        print(f"{'='*60}\n")

        # Step 1: Load video
        _progress("Loading video", 0.0)
        print("[1/9] Loading video...")
        video_path = self.config.get('video_path', '')

        # Get video properties without loading frames
        video_props = get_video_properties(video_path)
        video_fps = video_props['fps']
        video_width = video_props['width']
        video_height = video_props['height']
        total_frames = video_props['total_frames']
        duration_min = video_props['duration_sec'] / 60
        print(f"  ✓ {total_frames} frames  |  {video_width}x{video_height} @ {video_fps} FPS  |  {duration_min:.1f} min")

        # Decide load strategy based on video length
        # Short videos (<5 min) → load all into RAM as before
        # Long videos (≥5 min) → chunked processing to avoid RAM crash
        CHUNK_THRESHOLD_MIN = 5.0
        use_chunked = duration_min >= CHUNK_THRESHOLD_MIN

        # Performance optimization: downscale factor
        scale_factor = 1.0
        if self.downscale and video_height > self.target_height:
            scale_factor = self.target_height / video_height
            proc_w = int(video_width * scale_factor)
            proc_h = int(video_height * scale_factor)
            print(f"  ✓ Downscaling to {proc_h}p for processing (scale: {scale_factor:.2f})")
        else:
            proc_w, proc_h = video_width, video_height

        if use_chunked:
            print(f"  ✓ Long video detected — using chunked processing (saves RAM)")
            video_frames = None   # do NOT load all frames; tracker will read in chunks
        else:
            video_frames = read_video(video_path)
            print(f"  ✓ Loaded {len(video_frames)} frames into memory")
            if scale_factor != 1.0:
                video_frames = self._downscale_frames(video_frames, scale_factor)

        # Ensure required directories exist (safe for any working directory)
        os.makedirs('stubs', exist_ok=True)
        os.makedirs('output_videos', exist_ok=True)

        # Create video-specific stub paths so each video gets its own cached data
        video_stem = Path(video_path).stem
        track_stub = f'stubs/{video_stem}_tracks.pkl'
        camera_stub = f'stubs/{video_stem}_camera.pkl'

        # Step 2: Object detection and tracking
        _progress("Object detection & tracking", 1 / 9)
        print("\n[2/9] Running object detection and tracking...")
        model_path = self.config.get('processing_options', {}).get('model_path', 'models/best.pt')
        tracker = Tracker(model_path)

        if use_chunked and not os.path.exists(track_stub):
            # Chunked tracking: process in 500-frame chunks, never hold full video in RAM
            tracks = tracker.get_object_tracks_chunked(
                video_path=video_path,
                scale_factor=scale_factor,
                chunk_size=500,
                stub_path=track_stub,
            )
        elif use_chunked:
            # Stub exists — load from pickle directly without reading frames into RAM
            import pickle
            with open(track_stub, 'rb') as _f:
                tracks = pickle.load(_f)
            print(f"  ✓ Loaded tracks from stub (chunked mode): {track_stub}")
        else:
            tracks = tracker.get_object_tracks(
                video_frames,
                read_from_stub=True,
                stub_path=track_stub,
            )

        num_tracked_frames = len(tracks['players'])
        print(f"  ✓ {num_tracked_frames} frames tracked")

        # Add positions to tracks (needed for camera movement compensation)
        tracker.add_position_to_tracks(tracks)

        # ---------------------------------------------------------------
        # Always read the first frame (cheap: 1 frame only) for:
        #   - CameraMovementEstimator initialisation
        #   - ViewTransformer (only needs frame dims, already known)
        # ---------------------------------------------------------------
        first_frame = self._read_first_frame(video_path, scale_factor, proc_w, proc_h)

        # Step 3: Camera movement estimation
        _progress("Camera movement estimation", 2 / 9)
        print("\n[3/9] Estimating camera movement...")
        static_cam = self.config.get('processing_options', {}).get('static_camera', False)
        camera_estimator = CameraMovementEstimator(first_frame)

        if static_cam:
            # Veo/BePro fixed camera — no optical flow needed
            camera_movement_per_frame = [[0, 0]] * num_tracked_frames
            print("  ✓ static_camera=true — optical flow skipped (Veo/BePro fixed camera)")
        elif use_chunked:
            # Long video: stream optical flow frame-by-frame (O(1) RAM)
            camera_movement_per_frame = camera_estimator.get_camera_movement_streaming(
                video_path, num_tracked_frames, stub_path=camera_stub
            )
        else:
            # Short video: use full in-memory path
            camera_movement_per_frame = camera_estimator.get_camera_movement(
                video_frames,
                read_from_stub=True,
                stub_path=camera_stub,
            )
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
        print("  ✓ Camera movement compensated")

        # Step 4: Field calibration and position transformation
        _progress("Field calibration", 3 / 9)
        print("\n[4/9] Calibrating field and transforming positions...")
        visible_field_m = self.config.get('processing_options', {}).get('visible_field_meters', 40.0)
        view_transformer = ViewTransformer(
            frame_width=video_width,
            frame_height=video_height,
            visible_field_meters=visible_field_m,
        )
        view_transformer.add_transformed_position_to_tracks(tracks)
        print("  ✓ Positions transformed to field coordinates")

        # Interpolate ball positions
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Step 5: Team assignment via SigLIP + UMAP + KMeans
        _progress("Team assignment (SigLIP)", 4 / 9)
        print("\n[5/9] Assigning teams (SigLIP embeddings)...")
        team_classifier = SigLIPTeamClassifier(device='cuda' if self.use_gpu else 'cpu')
        team_stub = f'stubs/{video_stem}_teams.pkl'

        if os.path.exists(team_stub):
            team_classifier.load(team_stub)
        elif use_chunked:
            # Long video: stream frames instead of holding all in RAM
            team_classifier.fit_from_video(video_path, tracks, stride=60)
            team_classifier.save(team_stub)
        else:
            team_classifier.fit(video_frames, tracks, stride=60)
            team_classifier.save(team_stub)

        if use_chunked:
            team_colors = team_classifier.derive_team_colors_from_video(video_path, tracks)
        else:
            team_colors = team_classifier.derive_team_colors_from_tracks(video_frames, tracks)

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id in player_track:
                team_id = team_classifier.get_player_team(player_id)
                tracks['players'][frame_num][player_id]['team'] = team_id
                tracks['players'][frame_num][player_id]['team_color'] = team_colors[team_id]

        print("  ✓ Teams assigned")

        # Position-number fallback: assign 1-11 by field position when no roster
        # Runs only when team roster is empty (zero-config / auto mode)
        player_id_map = self.sport_analyzer._build_player_id_map(
            tracks['players'], self.config)
        # Quick sanity: check if position_transformed is available
        _has_pos = any(
            pdata.get('position_transformed') is not None
            for frame in tracks['players'][:10]
            for pdata in frame.values()
        )
        if not _has_pos:
            print("  ⚠ position_transformed not yet available — skipping position numbering")
        position_nums = assign_position_numbers(
            tracks['players'], player_id_map, self.config) if _has_pos else {}
        if position_nums:
            # Reverse: tracking_id -> position number
            tid_to_pos = {tid: position_nums.get(pid)
                          for tid, pid in player_id_map.items()
                          if position_nums.get(pid) is not None}
            # Write to 'position_number' (display-only field) — NOT 'jersey_number'.
            # jersey_number is reserved for OCR results used in analytics player mapping.
            # draw_ellipse shows: jersey_number (OCR) > position_number > track_id.
            for frame in tracks['players']:
                for tid, pdata in frame.items():
                    if tid in tid_to_pos:
                        pdata['position_number'] = tid_to_pos[tid]
            print(f"  ✓ Position numbers assigned (1-11) for {len(tid_to_pos)} tracks")

        # Step 6: Jersey number detection (if enabled)
        _progress("Jersey number detection", 5 / 9)
        enable_ocr = self.config.get('processing_options', {}).get('enable_jersey_ocr', True)
        sample_interval = self.config.get('processing_options', {}).get('jersey_ocr_sample_interval', 30)
        jersey_detector = None
        if enable_ocr:
            print("\n[6/9] Detecting jersey numbers...")
            jersey_detector = JerseyNumberDetector(use_ocr=True)
            if use_chunked:
                # Long video: only sample a sparse set of frames into RAM
                sampled = self._sample_frames_sparse(video_path, sample_interval, scale_factor, proc_w, proc_h)
                # Build a minimal indexable proxy so jersey detector can use sparse frames
                jersey_detector.add_jersey_numbers_to_tracks(sampled, tracks, sample_interval=1)
            else:
                jersey_detector.add_jersey_numbers_to_tracks(video_frames, tracks, sample_interval=sample_interval)
            print(f"  ✓ Detected {len(jersey_detector.player_jersey_dict)} jersey numbers")
        else:
            print("\n[6/9] Skipping jersey number detection (disabled in config)")

        # Step 7: Speed and distance calculation
        _progress("Speed & distance calculation", 6 / 9)
        print("\n[7/9] Calculating speed and distance...")
        unit_system = self.config.get('unit_system', 'metric')
        speed_estimator = SpeedAndDistance_Estimator(frame_rate=video_fps, unit_system=unit_system)
        speed_estimator.add_speed_and_distance_to_tracks(tracks)
        print("  ✓ Speed and distance calculated")

        # Step 8: Ball possession assignment
        _progress("Ball possession assignment", 7 / 9)
        print("\n[8/9] Assigning ball possession...")
        player_assigner = PlayerBallAssigner(frame_width=proc_w)
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
            if ball_bbox is None:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
                continue
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

        team_ball_control = np.array(team_ball_control)
        print("  ✓ Ball possession assigned")

        # Step 9: Generate visualization
        _progress("Rendering output video", 8 / 9)
        print("\n[9/9] Generating output video...")

        # Radar setup (optional, same for both paths)
        radar_obj = None
        if self.config.get('processing_options', {}).get('enable_radar', False):
            print("  Setting up radar overlay...")
            from radar import RadarRenderer
            pitch_model_path = self.config.get('processing_options', {}).get(
                'pitch_model_path', 'models/pitch_detection.pt'
            )
            radar_obj = RadarRenderer(
                pitch_model_path=pitch_model_path,
                device='cuda' if self.use_gpu else 'cpu',
            )

        match_id = self.config.get('match_id', 'auto')
        output_path = f"output_videos/{match_id}_output.mp4"

        if use_chunked:
            # Long video: stream render — O(1) RAM, writes directly to VideoWriter
            self._render_video_streaming(
                video_path=video_path,
                tracks=tracks,
                team_ball_control=team_ball_control,
                output_path=output_path,
                video_fps=video_fps,
                tracker=tracker,
                speed_estimator=speed_estimator,
                scale_factor=scale_factor,
                team_colors=team_colors,
                video_width=video_width,
                video_height=video_height,
                radar=radar_obj,
                match_config=self.config,
            )
        else:
            # Short video: in-memory path (existing behaviour)
            output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, self.config)
            speed_estimator.draw_speed_and_distance(output_frames, tracks)

            if radar_obj is not None:
                for frame_num in range(len(output_frames)):
                    if radar_obj.keypoint_mode:
                        radar_obj.update_frame_transformer(video_frames[frame_num])
                    output_frames[frame_num] = radar_obj.render_radar(
                        frame=output_frames[frame_num],
                        player_dict=tracks['players'][frame_num],
                        team_colors_bgr=team_colors,
                        ball_dict=tracks['ball'][frame_num],
                    )
                print("  ✓ Radar overlay applied")

            if scale_factor != 1.0:
                print(f"  Upscaling back to {video_width}x{video_height}...")
                output_frames = self._upscale_frames(output_frames, video_width, video_height)

            save_video(output_frames, output_path, fps=video_fps)

        print(f"  ✓ Video saved to: {output_path}")

        # Calculate and save metrics using sport analyzer
        print("\n[Bonus] Calculating player metrics...")
        # Use first_frame (always available, read at step 3 init) — safe for chunked path
        field_calibration = self.sport_analyzer.calibrate_field(first_frame)
        player_metrics = self.sport_analyzer.calculate_metrics(
            tracks,
            self.config,
            field_calibration,
            video_fps
        )

        # Detect match events (passes, shots, possession changes)
        print("[Bonus] Detecting match events...")
        # Inject runtime video properties so analyzers don't need to guess or hard-code
        config_with_fps = {
            **self.config,
            'fps': video_fps,
            'frame_width': proc_w,
            'frame_height': proc_h,
        }
        # video_frames may be None on chunked path; detect_events only uses tracks + config
        events = self.sport_analyzer.detect_events(tracks, video_frames or [], config_with_fps)
        passes = [e for e in events if e.event_type == 'pass']
        shots  = [e for e in events if e.event_type == 'shot']
        print(f"  ✓ {len(passes)} passes  |  {len(shots)} shots  |  {len(events)} total events")

        # Back-fill pass/shot counts into player metrics
        for event in passes:
            pid = event.player_id
            if pid in player_metrics:
                player_metrics[pid].sport_metrics['passes_attempted'] += 1
                player_metrics[pid].sport_metrics['passes_completed'] += 1  # refined later
        for event in shots:
            pid = event.player_id
            if pid in player_metrics:
                player_metrics[pid].sport_metrics['shots'] += 1

        # Save metrics and events to JSON
        metrics_path = f"output_videos/{match_id}_metrics.json"
        events_path  = f"output_videos/{match_id}_events.json"
        self._save_metrics(player_metrics, metrics_path)
        self._save_events(events, events_path)
        print(f"  ✓ Metrics saved to: {metrics_path}")
        print(f"  ✓ Events  saved to: {events_path}")

        # Print top tracks summary
        self._print_metrics_summary(player_metrics)

        _progress("Complete", 1.0)
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}\n")

        return {
            'output_video': output_path,
            'metrics': metrics_path,
            'events': events_path,
            'player_metrics': player_metrics,
            'match_events': events,
        }

    # ------------------------------------------------------------------
    # Streaming helpers — O(1) RAM regardless of video length
    # ------------------------------------------------------------------

    def _read_first_frame(self, video_path, scale_factor, proc_w, proc_h):
        """Read only the first frame from the video (for estimator init)."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read first frame from {video_path}")
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        return frame

    def _sample_frames_sparse(self, video_path, sample_interval, scale_factor, proc_w, proc_h):
        """
        Read every sample_interval-th frame into a list (sparse sampling).
        Memory = (total_frames / sample_interval) × frame_size — safe for jersey OCR.
        For a 1h49m video at interval=30: ~5,400 frames ≈ 32GB at 1080p.
        Reduce interval or use enable_jersey_ocr=false for very long videos.
        """
        cap = cv2.VideoCapture(video_path)
        sampled = []
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % sample_interval == 0:
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                sampled.append(frame)
            frame_num += 1
        cap.release()
        return sampled

    def _render_video_streaming(
        self,
        video_path,
        tracks,
        team_ball_control,
        output_path,
        video_fps,
        tracker,
        speed_estimator,
        scale_factor,
        team_colors,
        video_width,
        video_height,
        radar=None,
        match_config=None,
    ):
        """
        Render annotated video frame-by-frame — O(1) RAM.

        Reads source video with VideoCapture, annotates each frame, writes
        directly to VideoWriter. Never holds more than ~5 frames in RAM.
        Safe for any video length (1 minute or 90 minutes).
        """
        cap = cv2.VideoCapture(video_path)
        writer = None
        total = len(tracks['players'])
        frame_num = 0

        print(f"  Streaming render: {total} frames → {output_path}")

        while frame_num < total:
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale to processing resolution
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor),
                                           int(frame.shape[0] * scale_factor)),
                                   interpolation=cv2.INTER_AREA)

            # Annotate: ellipses, ball triangle, team control overlay
            frame = tracker.annotate_frame(frame, frame_num, tracks, team_ball_control, match_config)

            # Speed / distance text
            speed_estimator.annotate_frame(frame, frame_num, tracks)

            # Radar overlay (optional)
            if radar is not None:
                if radar.keypoint_mode:
                    radar.update_frame_transformer(frame)
                frame = radar.render_radar(
                    frame=frame,
                    player_dict=tracks['players'][frame_num],
                    team_colors_bgr=team_colors,
                    ball_dict=tracks['ball'][frame_num],
                )

            # Upscale back to original resolution
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (video_width, video_height),
                                   interpolation=cv2.INTER_LINEAR)

            # Lazy-init VideoWriter on first annotated frame
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, video_fps, (w, h))

            writer.write(frame)
            frame_num += 1

            if frame_num % 500 == 0:
                pct = frame_num / total * 100
                print(f"  {frame_num}/{total} rendered ({pct:.0f}%)", end='\r')

        cap.release()
        if writer:
            writer.release()
        print(f"\n  Streaming render complete: {frame_num} frames written")

    def _downscale_frames(self, frames, scale_factor):
        """Downscale frames for faster processing"""
        downscaled = []
        for frame in frames:
            height, width = frame.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            downscaled.append(resized)
        return downscaled

    def _upscale_frames(self, frames, target_width, target_height):
        """Upscale frames back to original resolution"""
        upscaled = []
        for frame in frames:
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            upscaled.append(resized)
        return upscaled

    def _save_metrics(self, player_metrics, output_path):
        """Save player metrics to JSON file, converting units per config."""
        unit_system = self.config.get('unit_system', 'metric')
        use_imperial = unit_system == 'imperial'
        dist_unit = 'yards' if use_imperial else 'm'
        speed_unit = 'mph' if use_imperial else 'km/h'

        # --- Build per-player output and accumulate team totals ---
        metrics_dict = {}
        team_totals: dict = {}  # team_id -> aggregated stats

        for player_id, metrics in player_metrics.items():
            sm = metrics.sport_metrics.copy()

            # Compute pass accuracy now that passes are back-filled
            att = sm.get('passes_attempted', 0)
            comp = sm.get('passes_completed', 0)
            sm['pass_accuracy'] = round(comp / att, 3) if att > 0 else 0.0

            if use_imperial:
                distance = round(metrics.distance_covered_m * 1.09361, 1)
                top_speed = round(metrics.top_speed_kmh * 0.62137, 1)
                avg_speed = round(metrics.avg_speed_kmh * 0.62137, 1)
                if 'sprint_distance_m' in sm:
                    sm['sprint_distance'] = round(sm.pop('sprint_distance_m') * 1.09361, 1)
                    sm['sprint_distance_unit'] = 'yards'
            else:
                distance = round(metrics.distance_covered_m, 1)
                top_speed = round(metrics.top_speed_kmh, 1)
                avg_speed = round(metrics.avg_speed_kmh, 1)
                if 'sprint_distance_m' in sm:
                    sm['sprint_distance'] = sm.pop('sprint_distance_m')
                    sm['sprint_distance_unit'] = 'm'

            metrics_dict[player_id] = {
                'player_name': metrics.player_name,
                'team_id': metrics.team_id,
                'unit_system': unit_system,
                'minutes_played': round(metrics.minutes_played, 2),
                'distance_covered': distance,
                'distance_unit': dist_unit,
                'top_speed': top_speed,
                'avg_speed': avg_speed,
                'speed_unit': speed_unit,
                'sprints_count': metrics.sprints_count,
                'sport_metrics': sm,
            }

            # Accumulate team totals
            tid = metrics.team_id or 'unknown'
            if tid not in team_totals:
                team_totals[tid] = {
                    'total_distance': 0.0,
                    'max_speed': 0.0,
                    'total_passes': 0,
                    'total_possession_sec': 0.0,
                    'total_sprint_distance': 0.0,
                    'track_count': 0,
                    'speed_zones_sec': {z: 0.0 for z in
                        ['standing', 'walking', 'jogging', 'running', 'high_speed', 'sprinting']},
                }
            tt = team_totals[tid]
            tt['total_distance'] = round(tt['total_distance'] + distance, 1)
            tt['max_speed'] = round(max(tt['max_speed'], top_speed), 1)
            tt['total_passes'] += sm.get('passes_attempted', 0)
            tt['total_possession_sec'] = round(
                tt['total_possession_sec'] + sm.get('possession_time_sec', 0.0), 2)
            tt['total_sprint_distance'] = round(
                tt['total_sprint_distance'] + sm.get('sprint_distance', 0.0), 1)
            tt['track_count'] += 1
            for z, v in sm.get('speed_zones_sec', {}).items():
                if z in tt['speed_zones_sec']:
                    tt['speed_zones_sec'][z] = round(tt['speed_zones_sec'][z] + v, 2)

        # Finalise team summaries with units
        for tid, tt in team_totals.items():
            tt['distance_unit'] = dist_unit
            tt['speed_unit'] = speed_unit
            tt['sprint_distance_unit'] = dist_unit

        # Build output: summary block first, then per-player entries
        output = {
            '_summary': {
                'unit_system': unit_system,
                'teams': team_totals,
            },
            'players': metrics_dict,
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    def _save_events(self, events, output_path):
        """Save match events to JSON file"""
        events_list = []
        for e in events:
            events_list.append({
                'event_id': e.event_id,
                'event_type': e.event_type,
                'timestamp_sec': round(e.timestamp_sec, 2),
                'frame_number': e.frame_number,
                'player_id': e.player_id,
                'team_id': e.team_id,
                'location_x': round(e.location_x, 2),
                'location_y': round(e.location_y, 2),
                'metadata': e.metadata,
            })
        with open(output_path, 'w') as f:
            json.dump(events_list, f, indent=2)

    def _print_metrics_summary(self, player_metrics):
        """Print a clean summary table of the top player tracks by distance"""
        if not player_metrics:
            return

        # Sort by distance descending, take top 25
        sorted_players = sorted(
            player_metrics.items(),
            key=lambda x: x[1].distance_covered_m,
            reverse=True
        )[:25]

        # Count named vs unknown
        named = sum(1 for pid, _ in player_metrics.items() if not pid.startswith(('unknown_', 'tottenham_unknown_', 'chelsea_unknown_')))
        total = len(player_metrics)

        print(f"\n{'='*60}")
        print(f"  Top Player Track Segments  ({total} segments total, {named} named)")
        print(f"{'='*60}")
        unit_system = self.config.get('unit_system', 'metric')
        use_imperial = unit_system == 'imperial'
        dist_lbl = 'Dist(yd)' if use_imperial else 'Dist(m)'
        spd_lbl  = 'Top mph' if use_imperial else 'Top km/h'
        spd2_lbl = 'Avg mph' if use_imperial else 'Avg km/h'

        print(f"  {'Player':<35} {'Min':>5}  {dist_lbl:>8}  {spd_lbl:>9}  {spd2_lbl:>9}")
        print(f"  {'-'*35} {'-'*5}  {'-'*8}  {'-'*9}  {'-'*9}")

        for player_id, m in sorted_players:
            name = m.player_name if m.player_name != player_id else player_id
            display = name if len(name) <= 35 else name[:32] + '...'
            dist = m.distance_covered_m * 1.09361 if use_imperial else m.distance_covered_m
            spd  = m.top_speed_kmh * 0.62137 if use_imperial else m.top_speed_kmh
            aspd = m.avg_speed_kmh * 0.62137 if use_imperial else m.avg_speed_kmh
            print(f"  {display:<35} {m.minutes_played:>5.1f}  {dist:>8.1f}  {spd:>9.1f}  {aspd:>9.1f}")

        if total > 25:
            print(f"  ... and {total - 25} more segments (see metrics JSON)")
        print(f"{'='*60}")

        # Team totals
        from collections import defaultdict
        team_agg = defaultdict(lambda: {'dist': 0.0, 'passes': 0, 'possession': 0.0, 'tracks': 0, 'max_spd': 0.0})
        for _, m in player_metrics.items():
            t = m.team_id or 'unknown'
            d = m.distance_covered_m * 1.09361 if use_imperial else m.distance_covered_m
            s = m.top_speed_kmh * 0.62137 if use_imperial else m.top_speed_kmh
            team_agg[t]['dist'] += d
            team_agg[t]['max_spd'] = max(team_agg[t]['max_spd'], s)
            team_agg[t]['passes'] += m.sport_metrics.get('passes_attempted', 0)
            team_agg[t]['possession'] += m.sport_metrics.get('possession_time_sec', 0.0)
            team_agg[t]['tracks'] += 1
        dist_u = 'yd' if use_imperial else 'm'
        spd_u = 'mph' if use_imperial else 'km/h'
        print(f"\n  Team Totals:")
        for t, agg in sorted(team_agg.items()):
            print(f"    {t:<30}  dist={agg['dist']:.0f}{dist_u}  "
                  f"max={agg['max_spd']:.1f}{spd_u}  "
                  f"passes={agg['passes']}  "
                  f"poss={agg['possession']:.1f}s  "
                  f"({agg['tracks']} track segments)")
        print()


def _make_auto_config(video_path: str, sport: str = 'soccer', unit_system: str = 'imperial') -> Dict[str, Any]:
    """Generate a minimal config for --auto mode (no roster required)."""
    import datetime
    stem = Path(video_path).stem
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return {
        'match_id': f'auto_{stem}_{ts}',
        'sport': sport,
        'competition': 'Match',
        'unit_system': unit_system,
        'video_path': video_path,
        'team_home': {'id': 'team_a', 'name': 'Team A', 'players': []},
        'team_away': {'id': 'team_b', 'name': 'Team B', 'players': []},
        'processing_options': {
            'use_gpu': True,
            'downscale_for_processing': True,
            'target_processing_height': 720,
            'static_camera': False,
            'visible_field_meters': 68.0,
            'enable_jersey_ocr': False,
            'enable_radar': False,
        }
    }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-Sport Analytics Platform v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With full match config:
  python main_v3.py --config match_configs/chicago_fc_united_u19.json

  # Quick run with no roster (auto mode, imperial units):
  python main_v3.py --video input_videos/game.mp4 --auto

  # Quick run with metric units:
  python main_v3.py --video input_videos/game.mp4 --auto --units metric
""")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to match configuration JSON file'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Video path (overrides config video_path, or use with --auto)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto mode: no config file required, generates minimal config from video filename'
    )
    parser.add_argument(
        '--units',
        type=str,
        choices=['imperial', 'metric'],
        default='imperial',
        help='Unit system for output (default: imperial for USA — mph, yards)'
    )
    parser.add_argument(
        '--sport',
        type=str,
        choices=['soccer', 'basketball'],
        default='soccer',
        help='Sport type for auto mode (default: soccer)'
    )
    parser.add_argument(
        '--clips',
        action='store_true',
        help='Generate highlight clips for each detected event after analysis'
    )
    parser.add_argument(
        '--reel',
        action='store_true',
        help='Also concatenate all event clips into a single highlight reel (requires --clips)'
    )
    parser.add_argument(
        '--analytics',
        action='store_true',
        help='Run data analytics (xT, shot map, pass network) on detected events after pipeline'
    )

    args = parser.parse_args()

    # Auto mode: build a minimal config from video path alone
    if args.auto:
        if not args.video:
            print("Error: --auto requires --video <path>")
            sys.exit(1)
        if not Path(args.video).exists():
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        import tempfile, json as _json
        auto_cfg = _make_auto_config(args.video, sport=args.sport, unit_system=args.units)
        # Write to a temp file so MatchProcessor can load it normally
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        _json.dump(auto_cfg, tmp)
        tmp.close()
        config_path = tmp.name
        print(f"Auto mode: generated config for '{Path(args.video).name}'")
    else:
        config_path = args.config or 'match_configs/example_match.json'
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            print("\nUsage:")
            print("  python main_v3.py --config match_configs/example_match.json")
            print("  python main_v3.py --video input_videos/game.mp4 --auto")
            sys.exit(1)

    # Process match
    processor = MatchProcessor(config_path)

    # Override video path if provided (and not already set by auto mode)
    if args.video and not args.auto:
        processor.config['video_path'] = args.video
        print(f"Using video override: {args.video}")

    # Apply CLI unit system preference (overrides config file)
    if args.units:
        processor.config['unit_system'] = args.units

    results = processor.process()

    print("\nResults:")
    print(f"  Video: {results['output_video']}")
    print(f"  Metrics: {results['metrics']}")

    # Optional: run data analytics (xT, shot map, pass network)
    if args.analytics:
        import subprocess
        analytics_out = f"output_videos/analytics"
        print(f"\n[Analytics] Running data analytics on events → {analytics_out}/")
        subprocess.run([
            sys.executable,
            "analytics/run_analysis.py",
            "--source", "video",
            "--events", results["events"],
            "--output", analytics_out,
        ], check=False)

    # Optional: generate event-based highlight clips
    if args.clips:
        from clip_generator import generate_clips, _build_highlight_reel
        clips_dir = f"output_videos/{Path(results['events']).stem}_clips"
        print(f"\n[Clips] Generating highlight clips → {clips_dir}/")
        clip_paths = generate_clips(
            video_path=processor.config['video_path'],
            events_path=results['events'],
            output_dir=clips_dir,
        )
        if args.reel and clip_paths:
            reel_path = f"output_videos/{Path(results['events']).stem}_highlight_reel.mp4"
            _build_highlight_reel(clip_paths, reel_path)
            print(f"  Highlight reel: {reel_path}")


if __name__ == '__main__':
    main()
