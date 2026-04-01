"""
Stone AI — Data Analytics Layer

Provides structured event data ingestion, VAEP/xT player valuation, and
rich pitch visualization. Works with both StatsBomb open data and
video-derived events from the Stone AI video pipeline.

Usage:
    from analytics import load_statsbomb_match, run_video_analytics
    from analytics.visualizer import shot_map, pass_network
"""

from analytics.statsbomb_loader import list_open_competitions, list_open_matches, load_match
from analytics.video_bridge import load_video_events

__all__ = [
    "load_match",
    "list_open_competitions",
    "list_open_matches",
    "load_video_events",
]
