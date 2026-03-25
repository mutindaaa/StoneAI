"""
Player Statistics Analyzer

Extracts and aggregates player statistics for scouting and analysis
"""

import json
import csv
import numpy as np
from collections import defaultdict
from pathlib import Path

class PlayerStatsAnalyzer:
    def __init__(self, tracks, video_fps=25, video_duration_sec=None):
        """
        Initialize analyzer with tracking data

        Args:
            tracks: Dictionary with 'players', 'ball', 'referees' tracking data
            video_fps: Video frames per second
            video_duration_sec: Total video duration in seconds
        """
        self.tracks = tracks
        self.fps = video_fps
        self.duration_sec = video_duration_sec or (len(tracks['players']) / video_fps)

    def get_all_player_ids(self):
        """Get list of all unique player IDs across all frames"""
        player_ids = set()
        for frame_tracks in self.tracks['players']:
            player_ids.update(frame_tracks.keys())
        return sorted(player_ids)

    def get_player_stats(self, player_id):
        """
        Calculate comprehensive stats for a single player

        Returns dict with:
            - player_id
            - total_distance_m
            - avg_speed_kmh
            - max_speed_kmh
            - time_on_screen_sec
            - possession_count
            - possession_time_sec
            - team
            - positions (list of x,y coordinates)
        """
        stats = {
            'player_id': player_id,
            'frames_detected': 0,
            'total_distance_m': 0.0,
            'speeds_kmh': [],
            'positions': [],
            'possession_frames': 0,
            'team': None,
            'team_color': None
        }

        for frame_num, frame_tracks in enumerate(self.tracks['players']):
            if player_id in frame_tracks:
                stats['frames_detected'] += 1
                player_data = frame_tracks[player_id]

                # Distance
                if 'distance' in player_data:
                    stats['total_distance_m'] += player_data['distance']

                # Speed
                if 'speed' in player_data and player_data['speed'] is not None:
                    stats['speeds_kmh'].append(player_data['speed'])

                # Position
                if 'position_transformed' in player_data and player_data['position_transformed'] is not None:
                    stats['positions'].append(player_data['position_transformed'])

                # Possession
                if player_data.get('has_ball', False):
                    stats['possession_frames'] += 1

                # Team
                if 'team' in player_data and stats['team'] is None:
                    stats['team'] = player_data['team']
                if 'team_color' in player_data and stats['team_color'] is None:
                    stats['team_color'] = player_data['team_color']

        # Calculate derived stats
        stats['time_on_screen_sec'] = stats['frames_detected'] / self.fps
        stats['avg_speed_kmh'] = np.mean(stats['speeds_kmh']) if stats['speeds_kmh'] else 0.0
        stats['max_speed_kmh'] = max(stats['speeds_kmh']) if stats['speeds_kmh'] else 0.0
        stats['possession_time_sec'] = stats['possession_frames'] / self.fps
        stats['possession_percentage'] = (stats['possession_frames'] / stats['frames_detected'] * 100) if stats['frames_detected'] > 0 else 0.0

        # Calculate movement stats
        if stats['positions']:
            stats['avg_position'] = np.mean(stats['positions'], axis=0).tolist()
            stats['position_std'] = np.std(stats['positions'], axis=0).tolist()

        # Remove raw data from output (keep it smaller)
        del stats['speeds_kmh']
        del stats['frames_detected']
        del stats['possession_frames']

        return stats

    def get_all_stats(self):
        """Get stats for all players"""
        all_player_ids = self.get_all_player_ids()
        return {
            player_id: self.get_player_stats(player_id)
            for player_id in all_player_ids
        }

    def get_team_stats(self, team_id):
        """Get aggregated stats for a team"""
        all_stats = self.get_all_stats()
        team_players = [
            stats for stats in all_stats.values()
            if stats.get('team') == team_id
        ]

        if not team_players:
            return None

        return {
            'team_id': team_id,
            'player_count': len(team_players),
            'total_distance_m': sum(p['total_distance_m'] for p in team_players),
            'avg_team_speed_kmh': np.mean([p['avg_speed_kmh'] for p in team_players]),
            'total_possession_time_sec': sum(p['possession_time_sec'] for p in team_players),
            'players': [p['player_id'] for p in team_players]
        }

    def export_to_json(self, output_path):
        """Export all stats to JSON file"""
        all_stats = self.get_all_stats()

        # Convert numpy int64 keys to regular ints for JSON serialization
        all_stats_clean = {int(k): v for k, v in all_stats.items()}

        output = {
            'video_metadata': {
                'fps': int(self.fps),
                'duration_sec': float(self.duration_sec),
                'total_frames': int(len(self.tracks['players']))
            },
            'player_stats': all_stats_clean,
            'team_stats': {
                1: self.get_team_stats(1),
                2: self.get_team_stats(2)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Player stats exported to: {output_path}")
        return output_path

    def export_to_csv(self, output_path):
        """Export player stats to CSV file"""
        all_stats = self.get_all_stats()

        # Define CSV columns
        fieldnames = [
            'player_id', 'team', 'time_on_screen_sec',
            'total_distance_m', 'avg_speed_kmh', 'max_speed_kmh',
            'possession_time_sec', 'possession_percentage'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for player_id, stats in sorted(all_stats.items()):
                row = {k: stats.get(k, '') for k in fieldnames}
                # Round floats for readability
                for key in row:
                    if isinstance(row[key], float):
                        row[key] = round(row[key], 2)
                writer.writerow(row)

        print(f"Player stats exported to: {output_path}")
        return output_path

    def generate_summary_report(self):
        """Generate a text summary report"""
        all_stats = self.get_all_stats()
        team1_stats = self.get_team_stats(1)
        team2_stats = self.get_team_stats(2)

        report = []
        report.append("=" * 60)
        report.append("PLAYER STATISTICS SUMMARY")
        report.append("=" * 60)
        report.append(f"Video Duration: {self.duration_sec:.1f} seconds ({self.duration_sec/60:.1f} minutes)")
        report.append(f"Total Players Tracked: {len(all_stats)}")
        report.append("")

        if team1_stats:
            report.append("TEAM 1:")
            report.append(f"  Players: {team1_stats['player_count']}")
            report.append(f"  Total Distance: {team1_stats['total_distance_m']:.1f}m")
            report.append(f"  Avg Speed: {team1_stats['avg_team_speed_kmh']:.1f} km/h")
            report.append(f"  Possession Time: {team1_stats['total_possession_time_sec']:.1f}s")
            report.append("")

        if team2_stats:
            report.append("TEAM 2:")
            report.append(f"  Players: {team2_stats['player_count']}")
            report.append(f"  Total Distance: {team2_stats['total_distance_m']:.1f}m")
            report.append(f"  Avg Speed: {team2_stats['avg_team_speed_kmh']:.1f} km/h")
            report.append(f"  Possession Time: {team2_stats['total_possession_time_sec']:.1f}s")
            report.append("")

        report.append("TOP 5 PLAYERS BY DISTANCE:")
        sorted_by_distance = sorted(all_stats.items(),
                                    key=lambda x: x[1]['total_distance_m'],
                                    reverse=True)[:5]
        for i, (player_id, stats) in enumerate(sorted_by_distance, 1):
            report.append(f"  {i}. Player #{player_id}: {stats['total_distance_m']:.1f}m "
                         f"(Team {stats['team']}, Avg {stats['avg_speed_kmh']:.1f} km/h)")

        report.append("")
        report.append("TOP 5 PLAYERS BY MAX SPEED:")
        sorted_by_speed = sorted(all_stats.items(),
                                key=lambda x: x[1]['max_speed_kmh'],
                                reverse=True)[:5]
        for i, (player_id, stats) in enumerate(sorted_by_speed, 1):
            report.append(f"  {i}. Player #{player_id}: {stats['max_speed_kmh']:.1f} km/h "
                         f"(Team {stats['team']})")

        report.append("=" * 60)

        return "\n".join(report)
