"""
SigLIP-based Team Classifier

Replaces raw K-means BGR clustering with:
1. SigLIP vision embeddings (google/siglip-base-patch16-224)
2. UMAP dimensionality reduction
3. KMeans clustering into 2 teams

Far more robust than pixel-color clustering — works even when kit colors
are similar (e.g. white vs light grey), handles varying lighting, and is
immune to pitch-green contamination.
"""

import pickle
import numpy as np
import cv2
from collections import defaultdict


class SigLIPTeamClassifier:
    """
    Classifies players into two teams using SigLIP image embeddings.

    Usage:
        classifier = SigLIPTeamClassifier(device='cuda')
        classifier.fit(video_frames, tracks, stride=60)
        team_colors = classifier.derive_team_colors_from_tracks(video_frames, tracks)

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team_id = classifier.get_player_team(player_id)
                track['team'] = team_id
                track['team_color'] = team_colors[team_id]
    """

    SIGLIP_MODEL = "google/siglip-base-patch16-224"
    CROP_SIZE = 224  # SigLIP input resolution

    def __init__(self, device='cuda'):
        self.device = device
        self._cluster_labels = {}   # track_id -> 0 or 1 (raw KMeans label)
        self._team_label_map = {}   # raw label -> team_id (1 or 2)
        self._fitted = False

        # Lazy-load heavy dependencies
        self._processor = None
        self._model = None
        self._load_siglip()

    def _load_siglip(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModel
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL)
            self._model = AutoModel.from_pretrained(self.SIGLIP_MODEL)
            self._model = self._model.to(self.device)
            self._model.eval()
            print(f"  SigLIP loaded on {self.device}")
        except Exception as e:
            print(f"  Warning: SigLIP unavailable ({e}). Falling back to K-means color clustering.")
            self._processor = None
            self._model = None

    def _extract_crop(self, frame, bbox):
        """Extract and resize a player crop from a frame, clamped to frame bounds."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize to SigLIP input size
        crop = cv2.resize(crop, (self.CROP_SIZE, self.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        return crop

    def _extract_embeddings(self, crops):
        """
        Run SigLIP on a list of BGR crops.
        Returns numpy array of shape (N, embedding_dim).
        """
        import torch

        if not crops:
            return np.empty((0, 768), dtype=np.float32)

        # Convert BGR -> RGB PIL-like arrays
        rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]

        # Process in batches of 32
        all_embeddings = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(rgb_crops), batch_size):
                batch = rgb_crops[i:i + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Extract vision features
                outputs = self._model.vision_model(**inputs)
                # Use pooled output (CLS token equivalent)
                embeddings = outputs.pooler_output  # (batch, dim)
                all_embeddings.append(embeddings.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _kmeans_color_fallback(self, video_frames, tracks, stride=60):
        """
        Fallback: K-means on mean BGR colors when SigLIP unavailable.
        Mirrors the old TeamAssigner logic but tracks which track_id maps to which team.
        """
        from sklearn.cluster import KMeans

        crops_by_track = defaultdict(list)
        fps_skip = max(1, int(stride))
        start_frame = 0  # no skip needed for fallback

        for frame_num in range(start_frame, len(video_frames), fps_skip):
            frame = video_frames[frame_num]
            for track_id, player_data in tracks['players'][frame_num].items():
                crop = self._extract_crop(frame, player_data['bbox'])
                if crop is not None:
                    # Use top half of crop for jersey color
                    top_half = crop[:crop.shape[0] // 2]
                    mean_bgr = top_half.reshape(-1, 3).mean(axis=0)
                    crops_by_track[track_id].append(mean_bgr)

        if not crops_by_track:
            return

        # Compute mean color per track
        track_ids = list(crops_by_track.keys())
        mean_colors = np.array([np.mean(crops_by_track[tid], axis=0) for tid in track_ids])

        if len(track_ids) < 2:
            for tid in track_ids:
                self._cluster_labels[tid] = 0
            return

        n_clusters = min(2, len(track_ids))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(mean_colors)

        for tid, label in zip(track_ids, labels):
            self._cluster_labels[tid] = int(label)

        self._fitted = True

    def fit(self, video_frames, tracks, stride=60):
        """
        Fit the classifier on sampled frames.

        Args:
            video_frames: List of BGR frames
            tracks: Tracking data dict with 'players' key
            stride: Sample every N frames (default: 60 ≈ every 2 sec at 30fps)
        """
        if self._model is None:
            print("  SigLIP unavailable — using K-means color fallback")
            self._kmeans_color_fallback(video_frames, tracks, stride)
            return

        from sklearn.cluster import KMeans
        import umap

        # Skip first 5 seconds to avoid single-team kickoff closeups
        # (assumes at least some frames exist)
        fps_estimate = 30
        skip_frames = fps_estimate * 5

        # Collect crops per track_id across sampled frames
        crops_by_track = defaultdict(list)

        for frame_num in range(skip_frames, len(video_frames), stride):
            frame = video_frames[frame_num]
            for track_id, player_data in tracks['players'][frame_num].items():
                crop = self._extract_crop(frame, player_data['bbox'])
                if crop is not None:
                    crops_by_track[track_id].append(crop)

        if not crops_by_track:
            print("  Warning: no crops collected for SigLIP — trying without skip")
            for frame_num in range(0, len(video_frames), stride):
                frame = video_frames[frame_num]
                for track_id, player_data in tracks['players'][frame_num].items():
                    crop = self._extract_crop(frame, player_data['bbox'])
                    if crop is not None:
                        crops_by_track[track_id].append(crop)

        if not crops_by_track:
            print("  Warning: still no crops — skipping team classification")
            return

        # Compute one representative embedding per track (mean of all crops)
        track_ids = []
        all_crops = []
        crop_to_track = []  # which track_id each crop belongs to

        for track_id, crop_list in crops_by_track.items():
            track_ids.append(track_id)
            # Use up to 5 crops per track to avoid large tracks dominating
            sampled = crop_list[::max(1, len(crop_list) // 5)][:5]
            all_crops.extend(sampled)
            crop_to_track.extend([track_id] * len(sampled))

        print(f"  Extracting SigLIP embeddings for {len(all_crops)} crops ({len(track_ids)} tracks)...")
        embeddings = self._extract_embeddings(all_crops)

        # Average per track
        track_embeddings = {}
        for i, tid in enumerate(crop_to_track):
            if tid not in track_embeddings:
                track_embeddings[tid] = []
            track_embeddings[tid].append(embeddings[i])

        final_track_ids = list(track_embeddings.keys())
        final_embeddings = np.array([np.mean(track_embeddings[tid], axis=0) for tid in final_track_ids])

        if len(final_track_ids) < 2:
            for tid in final_track_ids:
                self._cluster_labels[tid] = 0
            self._fitted = True
            return

        # UMAP: reduce to 3D for more robust separation
        n_neighbors = min(15, len(final_track_ids) - 1)
        reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
        reduced = reducer.fit_transform(final_embeddings)

        # KMeans: cluster into 2 teams
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(reduced)

        for tid, label in zip(final_track_ids, labels):
            self._cluster_labels[tid] = int(label)

        self._fitted = True
        print(f"  SigLIP classifier fitted: {sum(1 for l in labels if l==0)} tracks in cluster 0, {sum(1 for l in labels if l==1)} in cluster 1")

    def get_player_team(self, track_id) -> int:
        """
        Return team ID (1 or 2) for a given track_id.
        Returns 1 as default if track was never seen during fitting.
        """
        raw = self._cluster_labels.get(track_id)
        if raw is None:
            return 1
        # Bridge: KMeans labels are 0-indexed, our convention is 1-indexed
        return raw + 1

    def derive_team_colors_from_tracks(self, video_frames, tracks) -> dict:
        """
        Derive a representative BGR color for each team by averaging
        the mean jersey color of all crops assigned to that team.

        Returns:
            {1: (B, G, R), 2: (B, G, R)}  — plain Python tuples (not numpy)
        """
        team_crops = defaultdict(list)
        stride = 60  # sample interval

        for frame_num in range(0, len(video_frames), stride):
            frame = video_frames[frame_num]
            for track_id, player_data in tracks['players'][frame_num].items():
                team_id = self.get_player_team(track_id)
                crop = self._extract_crop(frame, player_data['bbox'])
                if crop is not None:
                    # Use top half of the crop (jersey area)
                    top_half = crop[:crop.shape[0] // 2]
                    mean_bgr = top_half.reshape(-1, 3).mean(axis=0)
                    team_crops[team_id].append(mean_bgr)

        team_colors = {}
        defaults = {1: (200, 200, 200), 2: (50, 50, 200)}  # grey vs blue fallback

        for team_id in [1, 2]:
            if team_crops[team_id]:
                avg = np.mean(team_crops[team_id], axis=0)
                team_colors[team_id] = (int(avg[0]), int(avg[1]), int(avg[2]))
            else:
                team_colors[team_id] = defaults[team_id]

        return team_colors

    def fit_from_video(self, video_path: str, tracks, stride: int = 60):
        """
        Streaming alternative to fit() — reads sample frames directly from
        a video file so the full frame list never has to be in RAM.

        Identical logic to fit(), but uses VideoCapture instead of a frame list.
        Safe for videos of any length.
        """
        import cv2

        crops_by_track = defaultdict(list)
        cap = cv2.VideoCapture(video_path)
        fps_estimate = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
        skip_frames  = fps_estimate * 5   # skip first 5 s (kickoff closeups)

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip first few seconds; only sample every stride-th frame
            if frame_num >= skip_frames and frame_num % stride == 0:
                if frame_num < len(tracks['players']):
                    for track_id, player_data in tracks['players'][frame_num].items():
                        crop = self._extract_crop(frame, player_data['bbox'])
                        if crop is not None:
                            crops_by_track[track_id].append(crop)

            frame_num += 1

        cap.release()

        # If no crops from skip-zone approach, try without skip
        if not crops_by_track:
            cap2 = cv2.VideoCapture(video_path)
            frame_num = 0
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                if frame_num % stride == 0 and frame_num < len(tracks['players']):
                    for track_id, player_data in tracks['players'][frame_num].items():
                        crop = self._extract_crop(frame, player_data['bbox'])
                        if crop is not None:
                            crops_by_track[track_id].append(crop)
                frame_num += 1
            cap2.release()

        if not crops_by_track:
            print("  Warning: no crops collected for SigLIP — skipping team classification")
            return

        if self._model is None:
            # Fallback: K-means on mean BGR colors
            from sklearn.cluster import KMeans
            track_ids  = list(crops_by_track.keys())
            mean_colors = np.array([
                np.mean([c[:c.shape[0]//2].reshape(-1, 3).mean(axis=0) for c in crops_by_track[tid]], axis=0)
                for tid in track_ids
            ])
            n_clusters = min(2, len(track_ids))
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = km.fit_predict(mean_colors)
            for tid, label in zip(track_ids, labels):
                self._cluster_labels[tid] = int(label)
            self._fitted = True
            return

        # SigLIP path — same as fit()
        from sklearn.cluster import KMeans
        import umap

        track_ids    = []
        all_crops    = []
        crop_to_track = []

        for track_id, crop_list in crops_by_track.items():
            track_ids.append(track_id)
            sampled = crop_list[::max(1, len(crop_list) // 5)][:5]
            all_crops.extend(sampled)
            crop_to_track.extend([track_id] * len(sampled))

        print(f"  Extracting SigLIP embeddings for {len(all_crops)} crops ({len(track_ids)} tracks)...")
        embeddings = self._extract_embeddings(all_crops)

        track_embeddings: dict = {}
        for i, tid in enumerate(crop_to_track):
            track_embeddings.setdefault(tid, []).append(embeddings[i])

        final_track_ids   = list(track_embeddings.keys())
        final_embeddings  = np.array([np.mean(track_embeddings[tid], axis=0) for tid in final_track_ids])

        if len(final_track_ids) < 2:
            for tid in final_track_ids:
                self._cluster_labels[tid] = 0
            self._fitted = True
            return

        n_neighbors = min(15, len(final_track_ids) - 1)
        reducer  = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
        reduced  = reducer.fit_transform(final_embeddings)
        km       = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels   = km.fit_predict(reduced)

        for tid, label in zip(final_track_ids, labels):
            self._cluster_labels[tid] = int(label)

        self._fitted = True
        c0 = sum(1 for l in labels if l == 0)
        print(f"  SigLIP classifier fitted (streaming): {c0} tracks cluster-0, {len(labels)-c0} cluster-1")

    def derive_team_colors_from_video(self, video_path: str, tracks, stride: int = 60) -> dict:
        """
        Streaming alternative to derive_team_colors_from_tracks().
        Reads sample frames from video_path without loading all into RAM.
        """
        import cv2

        team_crops: dict = {1: [], 2: []}
        cap = cv2.VideoCapture(video_path)
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % stride == 0 and frame_num < len(tracks['players']):
                for track_id, player_data in tracks['players'][frame_num].items():
                    team_id = self.get_player_team(track_id)
                    crop = self._extract_crop(frame, player_data['bbox'])
                    if crop is not None:
                        top_half = crop[:crop.shape[0] // 2]
                        mean_bgr = top_half.reshape(-1, 3).mean(axis=0)
                        if team_id in team_crops:
                            team_crops[team_id].append(mean_bgr)
            frame_num += 1

        cap.release()

        defaults = {1: (200, 200, 200), 2: (50, 50, 200)}
        team_colors = {}
        for team_id in [1, 2]:
            if team_crops[team_id]:
                avg = np.mean(team_crops[team_id], axis=0)
                team_colors[team_id] = (int(avg[0]), int(avg[1]), int(avg[2]))
            else:
                team_colors[team_id] = defaults[team_id]

        return team_colors

    def save(self, path: str):
        """Cache fitted cluster labels to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'cluster_labels': self._cluster_labels,
                'fitted': self._fitted,
            }, f)

    def load(self, path: str):
        """Load cached cluster labels from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._cluster_labels = data['cluster_labels']
        self._fitted = data['fitted']
        print(f"  Loaded team assignments from cache ({len(self._cluster_labels)} tracks)")
