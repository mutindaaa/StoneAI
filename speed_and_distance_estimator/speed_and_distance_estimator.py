import cv2
import sys
sys.path.append('../')
from utils import measure_distance ,get_foot_position

MAX_REALISTIC_SPEED_KMH = 38.0  # Discard windows above this — tracking noise, not real speed

class SpeedAndDistance_Estimator():
    def __init__(self, frame_rate=24, unit_system='metric'):
        self.frame_window=5
        self.frame_rate=frame_rate  # Use actual video FPS for accurate calculations
        self.unit_system = unit_system  # 'metric' (km/h, m) or 'imperial' (mph, yd)
    
    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    if time_elapsed <= 0:
                        continue
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    # Discard windows with impossible speed — tracking/detection noise
                    if speed_km_per_hour > MAX_REALISTIC_SPEED_KMH:
                        continue

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def _format_speed_dist(self, speed_kmh, dist_m):
        """Convert speed/distance to display units based on unit_system."""
        if self.unit_system == 'imperial':
            return f"{speed_kmh * 0.62137:.1f}mph", f"{dist_m * 1.09361:.1f}yd"
        return f"{speed_kmh:.1f}km/h", f"{dist_m:.1f}m"

    def annotate_frame(self, frame, frame_num, tracks):
        """Add speed/distance text to a single frame — streaming-friendly."""
        for object, object_tracks in tracks.items():
            if object in ("ball", "referees"):
                continue
            if frame_num >= len(object_tracks):
                continue
            for _, track_info in object_tracks[frame_num].items():
                speed    = track_info.get("speed")
                distance = track_info.get("distance")
                if speed is None or distance is None:
                    continue
                bbox     = track_info["bbox"]
                position = list(get_foot_position(bbox))
                position[1] += 35
                position = tuple(map(int, position))
                spd_txt, dist_txt = self._format_speed_dist(speed, distance)
                cv2.putText(frame, spd_txt, position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
                cv2.putText(frame, spd_txt, position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                cv2.putText(frame, dist_txt, (position[0], position[1]+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
                cv2.putText(frame, dist_txt, (position[0], position[1]+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        return frame

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed')
                        distance = track_info.get('distance')
                        if speed is None or distance is None:
                            continue
                        bbox = track_info['bbox']
                        position = list(get_foot_position(bbox))
                        position[1] += 35
                        position = tuple(map(int, position))
                        spd_txt, dist_txt = self._format_speed_dist(speed, distance)
                        cv2.putText(frame, spd_txt, position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
                        cv2.putText(frame, spd_txt, position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                        cv2.putText(frame, dist_txt, (position[0], position[1]+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
                        cv2.putText(frame, dist_txt, (position[0], position[1]+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            output_frames.append(frame)

        return output_frames