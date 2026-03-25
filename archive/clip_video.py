#!/usr/bin/env python3
"""
Simple video clipper - extracts a 30-second clip from a video

Usage:
    python clip_video.py input_video.mp4 output_clip.mp4 --start 0 --duration 30
"""

import cv2
import sys
import argparse

def clip_video(input_path, output_path, start_seconds=0, duration_seconds=30):
    """
    Extract a clip from a video

    Args:
        input_path: Path to input video
        output_path: Path to save output clip
        start_seconds: Start time in seconds
        duration_seconds: Duration of clip in seconds
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Calculate frame range
    start_frame = int(start_seconds * fps)
    end_frame = start_frame + int(duration_seconds * fps)

    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video {output_path}")
        return False

    print(f"Extracting frames {start_frame} to {end_frame}...")

    # Read and write frames
    frame_count = 0
    current_frame = start_frame

    while current_frame < end_frame and current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1
        current_frame += 1

        # Progress indicator
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{end_frame - start_frame} frames...")

    # Release resources
    cap.release()
    out.release()

    print(f"Done! Saved {frame_count} frames to {output_path}")
    print(f"Duration: {frame_count / fps:.2f} seconds")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clip a video to specified duration')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output video file')
    parser.add_argument('--start', type=float, default=0, help='Start time in seconds (default: 0)')
    parser.add_argument('--duration', type=float, default=30, help='Duration in seconds (default: 30)')

    args = parser.parse_args()

    success = clip_video(args.input, args.output, args.start, args.duration)
    sys.exit(0 if success else 1)
