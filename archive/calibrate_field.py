#!/usr/bin/env python3
"""
Interactive Field Calibration Tool

Usage:
    python calibrate_field.py input_videos/your_video.mp4

Instructions:
    1. Click on the 4 corners of the playing field in this order:
       - Top-left corner
       - Top-right corner
       - Bottom-right corner
       - Bottom-left corner
    2. Press 'r' to reset if you make a mistake
    3. Press 's' to save the calibration
    4. Press 'q' to quit without saving
"""

import cv2
import numpy as np
import json
import sys
import argparse
from pathlib import Path

class FieldCalibrator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.points = []
        self.display_frame = None
        self.original_frame = None

        # Load first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        self.original_frame = frame.copy()
        self.display_frame = frame.copy()

        # Field dimensions in meters (standard soccer field)
        self.field_width = 68.0  # meters
        self.field_height = 23.32  # meters (partial view typically)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                print(f"Point {len(self.points)}: ({x}, {y})")

                # Draw the point
                cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)

                # Draw labels
                labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                cv2.putText(self.display_frame, labels[len(self.points)-1],
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)

                # Draw lines between points
                if len(self.points) > 1:
                    cv2.line(self.display_frame,
                            tuple(self.points[-2]),
                            tuple(self.points[-1]),
                            (0, 255, 0), 2)

                # Close the polygon if 4 points
                if len(self.points) == 4:
                    cv2.line(self.display_frame,
                            tuple(self.points[-1]),
                            tuple(self.points[0]),
                            (0, 255, 0), 2)
                    print("\nAll 4 points selected!")
                    print("Press 's' to save, 'r' to reset, 'q' to quit")

                cv2.imshow('Field Calibration', self.display_frame)

    def reset(self):
        """Reset all points"""
        self.points = []
        self.display_frame = self.original_frame.copy()
        cv2.imshow('Field Calibration', self.display_frame)
        print("\nPoints reset. Click 4 corners again:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")

    def save_calibration(self, output_path=None):
        """Save calibration to JSON file"""
        if len(self.points) != 4:
            print("Error: Need exactly 4 points")
            return False

        if output_path is None:
            # Save in same directory as video with .json extension
            video_name = Path(self.video_path).stem
            output_path = f"calibration_{video_name}.json"

        calibration = {
            "video_path": str(self.video_path),
            "pixel_vertices": self.points,
            "target_vertices": [
                [0, 0],
                [self.field_width, 0],
                [self.field_width, self.field_height],
                [0, self.field_height]
            ],
            "field_dimensions": {
                "width": self.field_width,
                "height": self.field_height
            }
        }

        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)

        print(f"\nCalibration saved to: {output_path}")
        return True

    def run(self):
        """Run the interactive calibration"""
        window_name = 'Field Calibration'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Add instructions overlay
        instructions = self.display_frame.copy()
        cv2.rectangle(instructions, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.rectangle(instructions, (10, 10), (500, 150), (255, 255, 255), 2)

        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(instructions, "Click 4 field corners in order:",
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "1. Top-left     2. Top-right",
                   (20, y_offset+25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "3. Bottom-right 4. Bottom-left",
                   (20, y_offset+50), font, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "Press 'r' to reset, 's' to save, 'q' to quit",
                   (20, y_offset+85), font, 0.5, (255, 255, 255), 1)

        self.display_frame = instructions
        cv2.imshow(window_name, self.display_frame)

        print("\nField Calibration Tool")
        print("=" * 50)
        print("Click on the 4 corners of the playing field:")
        print("1. Top-left corner")
        print("2. Top-right corner")
        print("3. Bottom-right corner")
        print("4. Bottom-left corner")
        print("\nControls:")
        print("  r - Reset points")
        print("  s - Save calibration")
        print("  q - Quit")
        print("=" * 50)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting without saving...")
                break
            elif key == ord('r'):
                self.reset()
            elif key == ord('s'):
                if self.save_calibration():
                    break
                else:
                    print("Cannot save - need 4 points first!")

        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Calibrate soccer field for perspective transformation')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output calibration file (default: calibration_<video_name>.json)')

    args = parser.parse_args()

    try:
        calibrator = FieldCalibrator(args.video)
        calibrator.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
