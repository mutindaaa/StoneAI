#!/usr/bin/env python3
"""
Chunked Video Processing for Long Videos (2+ hours)

This version processes videos in chunks to avoid memory issues.
Use this for videos longer than a few minutes.

Usage:
    python main_chunked.py --video input_videos/long_match.mp4 --chunk-size 200
"""

from utils.video_utils import (read_video_chunks, get_video_properties,
                                save_video_streaming)
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from jersey_number_detector import JerseyNumberDetector
import argparse
import pickle
import os

def process_chunk(tracker, chunk_frames, chunk_start_frame, video_fps,
                 team_assigner=None, camera_estimator=None, view_transformer=None,
                 speed_estimator=None, player_assigner=None, team_ball_control=None,
                 jersey_detector=None):
    """
    Process a single chunk of frames

    Returns: (annotated_frames, tracks, updated_team_ball_control)
    """
    # Get tracks for this chunk
    print(f"  Processing frames {chunk_start_frame} to {chunk_start_frame + len(chunk_frames)}...")

    chunk_tracks = tracker.get_object_tracks(chunk_frames, read_from_stub=False)

    # Add positions
    tracker.add_position_to_tracks(chunk_tracks)

    # Camera movement for this chunk
    if camera_estimator is None:
        camera_estimator = CameraMovementEstimator(chunk_frames[0])

    chunk_camera_movement = camera_estimator.get_camera_movement(chunk_frames,
                                                                 read_from_stub=False)
    camera_estimator.add_adjust_positions_to_tracks(chunk_tracks, chunk_camera_movement)

    # View transformation
    if view_transformer is None:
        view_transformer = ViewTransformer()

    view_transformer.add_transformed_position_to_tracks(chunk_tracks)

    # Interpolate ball positions
    chunk_tracks["ball"] = tracker.interpolate_ball_positions(chunk_tracks["ball"])

    # Speed and distance (with correct FPS for accurate calculations)
    if speed_estimator is None:
        speed_estimator = SpeedAndDistance_Estimator(frame_rate=video_fps)

    speed_estimator.add_speed_and_distance_to_tracks(chunk_tracks)

    # Assign teams (only on first chunk, then reuse)
    if team_assigner is None:
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(chunk_frames[0], chunk_tracks['players'][0])

    for frame_num, player_track in enumerate(chunk_tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(chunk_frames[frame_num],
                                                track['bbox'],
                                                player_id)
            chunk_tracks['players'][frame_num][player_id]['team'] = team
            chunk_tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Jersey number detection
    if jersey_detector is None:
        jersey_detector = JerseyNumberDetector(use_ocr=True)

    jersey_detector.add_jersey_numbers_to_tracks(chunk_frames, chunk_tracks, sample_interval=30)

    # Ball possession
    if player_assigner is None:
        player_assigner = PlayerBallAssigner()

    if team_ball_control is None:
        team_ball_control = []

    for frame_num, player_track in enumerate(chunk_tracks['players']):
        if len(chunk_tracks['ball'][frame_num]) > 0:
            ball_bbox = chunk_tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                chunk_tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(chunk_tracks['players'][frame_num][assigned_player]['team'])
            elif len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
        elif len(team_ball_control) > 0:
            team_ball_control.append(team_ball_control[-1])
        else:
            team_ball_control.append(0)

    team_ball_control_array = np.array(team_ball_control)

    # Draw annotations
    output_frames = tracker.draw_annotations(chunk_frames, chunk_tracks, team_ball_control_array)
    # Camera movement calculated above (used for tracking, not displayed)
    speed_estimator.draw_speed_and_distance(output_frames, chunk_tracks)

    return (output_frames, chunk_tracks, team_ball_control,
            team_assigner, camera_estimator, view_transformer,
            speed_estimator, player_assigner, jersey_detector)

def main():
    parser = argparse.ArgumentParser(description='Process long soccer videos in chunks')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', default='output_videos/output_chunked.mp4',
                       help='Output video path')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Frames per chunk (default: 200)')
    parser.add_argument('--calibration', help='Field calibration JSON file')

    args = parser.parse_args()

    # Get video properties
    props = get_video_properties(args.video)
    print(f"\nVideo Properties:")
    print(f"  Resolution: {props['width']}x{props['height']}")
    print(f"  FPS: {props['fps']}")
    print(f"  Total frames: {props['total_frames']}")
    print(f"  Duration: {props['duration_sec']:.1f} seconds ({props['duration_sec']/60:.1f} minutes)")
    print(f"  Chunk size: {args.chunk_size} frames")
    print(f"  Total chunks: {props['total_frames'] // args.chunk_size + 1}")

    # Initialize tracker
    print("\nInitializing YOLO tracker...")
    tracker = Tracker('models/best.pt')

    # Initialize streaming video writer
    video_writer = save_video_streaming(args.output, props['fps'],
                                        props['width'], props['height'])

    # State that persists across chunks
    team_assigner = None
    camera_estimator = None
    view_transformer = None
    speed_estimator = None
    player_assigner = None
    team_ball_control = None
    jersey_detector = None
    all_tracks = {"players": [], "ball": [], "referees": []}

    print("\nProcessing video in chunks...")
    chunk_count = 0

    # Process video in chunks
    for chunk_frames, chunk_start in read_video_chunks(args.video, args.chunk_size):
        chunk_count += 1
        print(f"\nChunk {chunk_count}:")

        (output_frames, chunk_tracks, team_ball_control,
         team_assigner, camera_estimator, view_transformer,
         speed_estimator, player_assigner, jersey_detector) = process_chunk(
            tracker, chunk_frames, chunk_start, props['fps'],
            team_assigner, camera_estimator, view_transformer,
            speed_estimator, player_assigner, team_ball_control,
            jersey_detector
        )

        # Write frames to output
        for frame in output_frames:
            video_writer.write(frame)

        # Accumulate tracks
        all_tracks["players"].extend(chunk_tracks["players"])
        all_tracks["ball"].extend(chunk_tracks["ball"])
        all_tracks["referees"].extend(chunk_tracks["referees"])

        print(f"  Processed {len(chunk_frames)} frames, total so far: {len(all_tracks['players'])}")

    # Release video writer
    video_writer.release()

    print(f"\n\u2713 Processing complete!")
    print(f"  Total frames processed: {len(all_tracks['players'])}")
    print(f"  Output saved to: {args.output}")

    # Optionally save tracks for analysis
    tracks_path = args.output.replace('.mp4', '_tracks.pkl')
    with open(tracks_path, 'wb') as f:
        pickle.dump(all_tracks, f)
    print(f"  Tracks saved to: {tracks_path}")

if __name__ == '__main__':
    main()
