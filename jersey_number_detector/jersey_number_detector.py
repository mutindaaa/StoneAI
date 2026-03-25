"""
Jersey Number Detection using OCR

Detects jersey numbers from player bounding boxes using EasyOCR.
Caches results to avoid re-processing the same player.
"""

import cv2
import numpy as np
import re

class JerseyNumberDetector:
    def __init__(self, use_ocr=True):
        """
        Initialize jersey number detector

        Args:
            use_ocr: Whether to use OCR (requires easyocr package)
                     If False, will only use cached values
        """
        self.use_ocr = use_ocr
        self.reader = None
        self.player_jersey_dict = {}  # Cache: player_id -> jersey_number

        if use_ocr:
            try:
                import easyocr
                # Initialize EasyOCR with English (numbers)
                # gpu=True will use GPU if available
                self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                print("Jersey number OCR initialized (GPU enabled)")
            except ImportError:
                print("Warning: easyocr not installed. Jersey numbers will not be detected.")
                print("Install with: pip install easyocr")
                self.use_ocr = False
            except Exception as e:
                print(f"Warning: Could not initialize OCR: {e}")
                self.use_ocr = False

    def extract_jersey_region(self, frame, bbox):
        """
        Extract the region of the player's jersey where the number is likely to be

        Args:
            frame: Video frame
            bbox: Player bounding box [x1, y1, x2, y2]

        Returns:
            Cropped image of jersey region
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Crop player image
        player_img = frame[y1:y2, x1:x2]

        if player_img.size == 0:
            return None

        # Focus on upper-middle part of player (chest/back where jersey number is)
        height = player_img.shape[0]
        width = player_img.shape[1]

        # Take middle 40% of width, upper 30-60% of height
        y_start = int(height * 0.25)
        y_end = int(height * 0.55)
        x_start = int(width * 0.3)
        x_end = int(width * 0.7)

        jersey_region = player_img[y_start:y_end, x_start:x_end]

        return jersey_region

    def preprocess_for_ocr(self, img):
        """
        Preprocess image for better OCR results

        Args:
            img: Input image

        Returns:
            Preprocessed image
        """
        if img is None or img.size == 0:
            return None

        # Resize for better OCR (make it larger)
        scale = 3
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def detect_jersey_number(self, frame, bbox, player_id):
        """
        Detect jersey number from player bounding box

        Args:
            frame: Video frame
            bbox: Player bounding box
            player_id: Player tracking ID

        Returns:
            Jersey number as string, or None if not detected
        """
        # Check cache first
        if player_id in self.player_jersey_dict:
            return self.player_jersey_dict[player_id]

        # If OCR not available, return None
        if not self.use_ocr or self.reader is None:
            return None

        # Extract jersey region
        jersey_region = self.extract_jersey_region(frame, bbox)
        if jersey_region is None or jersey_region.size == 0:
            return None

        # Preprocess for OCR
        processed = self.preprocess_for_ocr(jersey_region)
        if processed is None:
            return None

        try:
            # Run OCR
            results = self.reader.readtext(processed, allowlist='0123456789')

            # Extract numbers from results
            detected_numbers = []
            for (bbox, text, confidence) in results:
                # Filter: only keep 1-2 digit numbers with high confidence
                # Threshold raised to 0.6 to reduce misreads on low-res broadcast
                # Reject "0" alone — almost always a misread of background noise
                text_clean = re.sub(r'\D', '', text)  # Remove non-digits
                if len(text_clean) in [1, 2] and confidence > 0.6 and text_clean != '0':
                    detected_numbers.append((text_clean, confidence))

            # If we found numbers, take the one with highest confidence
            if detected_numbers:
                best_number = max(detected_numbers, key=lambda x: x[1])[0]

                # Cache the result
                self.player_jersey_dict[player_id] = best_number
                return best_number

        except Exception as e:
            # OCR failed silently, don't spam logs
            pass

        return None

    def add_jersey_numbers_to_tracks(self, frames, tracks, sample_interval=30):
        """
        Add jersey numbers to tracking data

        Args:
            frames: Video frames
            tracks: Tracking data dictionary
            sample_interval: Only process every Nth frame to save time (default: 30)
        """
        if not self.use_ocr or self.reader is None:
            return

        for frame_num, frame in enumerate(frames):
            # Only sample every Nth frame to reduce OCR overhead
            if frame_num % sample_interval != 0:
                continue

            player_tracks = tracks['players'][frame_num]

            for player_id, track_info in player_tracks.items():
                # Skip if we already have this player's number
                if player_id in self.player_jersey_dict:
                    continue

                bbox = track_info['bbox']
                jersey_number = self.detect_jersey_number(frame, bbox, player_id)

                # Don't need to store in tracks, it's in the cache

        # Now add cached numbers to all frames
        for frame_num in range(len(tracks['players'])):
            for player_id, track_info in tracks['players'][frame_num].items():
                if player_id in self.player_jersey_dict:
                    tracks['players'][frame_num][player_id]['jersey_number'] = self.player_jersey_dict[player_id]
