import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height, width = first_frame_grayscale.shape

        # Create adaptive mask based on frame size
        # Track features on the edges (left 5% and right 20% of frame)
        mask_features = np.zeros_like(first_frame_grayscale)
        left_edge = int(width * 0.05)
        right_start = int(width * 0.80)

        mask_features[:, 0:left_edge] = 1
        mask_features[:, right_start:width] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement_streaming(self, video_path, num_frames, stub_path=None):
        """
        Compute camera movement by reading frames one at a time from disk.
        Never holds more than 2 frames in RAM — safe for any video length.

        Args:
            video_path:  Path to the source video file.
            num_frames:  Expected number of frames (len of tracks['players']).
            stub_path:   Optional cache path; loads/saves pickle stub.
        """
        if stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        cap = cv2.VideoCapture(video_path)
        camera_movement = [[0, 0]] * num_frames

        ret, first = cap.read()
        if not ret:
            cap.release()
            return camera_movement

        old_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        if old_features is None or len(old_features) == 0:
            cap.release()
            return camera_movement

        frame_num = 1
        while frame_num < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None or status is None:
                old_gray = frame_gray
                frame_num += 1
                continue

            good_new = new_features[status == 1]
            good_old = old_features[status == 1]

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            for new_pt, old_pt in zip(good_new, good_old):
                d = measure_distance(new_pt.ravel(), old_pt.ravel())
                if d > max_distance:
                    max_distance = d
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_pt.ravel(), new_pt.ravel()
                    )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                if old_features is None or len(old_features) == 0:
                    old_features = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
                    if old_features is None:
                        break

            old_gray = frame_gray
            frame_num += 1

        cap.release()

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        # If no features found, return no camera movement
        if old_features is None or len(old_features) == 0:
            print("Warning: No features detected for camera movement tracking. Assuming static camera.")
            if stub_path is not None:
                with open(stub_path,'wb') as f:
                    pickle.dump(camera_movement,f)
            return camera_movement

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            # Check if optical flow was successful
            if new_features is None or status is None:
                camera_movement[frame_num] = [0, 0]
                continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            # Filter good features based on status
            if status is not None:
                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                for i, (new,old) in enumerate(zip(good_new, good_old)):
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()

                    distance = measure_distance(new_features_point,old_features_point)
                    if distance>max_distance:
                        max_distance = distance
                        camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
                # If no new features found, keep old ones
                if old_features is None or len(old_features) == 0:
                    old_features = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
                    if old_features is None:
                        break

            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames