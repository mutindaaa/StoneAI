"""
Camera Cut Detector

Detects abrupt camera angle changes in video footage.
This helps identify when tracking needs to be reset to avoid ID confusion.
"""

import cv2
import numpy as np

class CameraCutDetector:
    def __init__(self, threshold=0.5):
        """
        Initialize camera cut detector

        Args:
            threshold: Histogram difference threshold (0-1, default 0.5)
                      Higher = only detect more dramatic cuts
        """
        self.threshold = threshold
        self.cut_frames = []

    def calculate_histogram_difference(self, frame1, frame2):
        """
        Calculate histogram difference between two frames

        Args:
            frame1, frame2: Video frames

        Returns:
            Normalized difference score (0-1)
        """
        # Convert to HSV for better color comparison
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # Calculate histograms for each channel
        hist1_h = cv2.calcHist([hsv1], [0], None, [50], [0, 180])
        hist1_s = cv2.calcHist([hsv1], [1], None, [60], [0, 256])
        hist1_v = cv2.calcHist([hsv1], [2], None, [60], [0, 256])

        hist2_h = cv2.calcHist([hsv2], [0], None, [50], [0, 180])
        hist2_s = cv2.calcHist([hsv2], [1], None, [60], [0, 256])
        hist2_v = cv2.calcHist([hsv2], [2], None, [60], [0, 256])

        # Normalize histograms
        cv2.normalize(hist1_h, hist1_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_s, hist1_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_v, hist1_v, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_h, hist2_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_s, hist2_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_v, hist2_v, 0, 1, cv2.NORM_MINMAX)

        # Compare histograms using Chi-Square
        diff_h = cv2.compareHist(hist1_h, hist2_h, cv2.HISTCMP_CHISQR)
        diff_s = cv2.compareHist(hist1_s, hist2_s, cv2.HISTCMP_CHISQR)
        diff_v = cv2.compareHist(hist1_v, hist2_v, cv2.HISTCMP_CHISQR)

        # Average and normalize
        avg_diff = (diff_h + diff_s + diff_v) / 3.0

        # Normalize to 0-1 range (empirically determined max value ~100)
        normalized_diff = min(avg_diff / 100.0, 1.0)

        return normalized_diff

    def detect_cuts(self, frames):
        """
        Detect camera cuts in a sequence of frames

        Args:
            frames: List of video frames

        Returns:
            List of frame indices where cuts occur
        """
        self.cut_frames = []

        for i in range(1, len(frames)):
            diff = self.calculate_histogram_difference(frames[i-1], frames[i])

            if diff > self.threshold:
                self.cut_frames.append(i)
                print(f"Camera cut detected at frame {i} (diff: {diff:.3f})")

        return self.cut_frames

    def is_cut_frame(self, frame_num):
        """
        Check if a frame is a camera cut

        Args:
            frame_num: Frame index

        Returns:
            True if frame is a cut frame
        """
        return frame_num in self.cut_frames

    def get_cut_segments(self, total_frames):
        """
        Get video segments separated by cuts

        Args:
            total_frames: Total number of frames in video

        Returns:
            List of (start_frame, end_frame) tuples for each segment
        """
        segments = []
        start = 0

        for cut_frame in self.cut_frames:
            segments.append((start, cut_frame - 1))
            start = cut_frame

        # Add final segment
        segments.append((start, total_frames - 1))

        return segments
