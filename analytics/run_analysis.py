"""
Stone AI Analytics — CLI Runner

Loads event data (StatsBomb open data or video-derived events),
computes xT player values, and saves visualizations + CSV exports.

Usage:
    # StatsBomb open data (requires: pip install kloppy statsbombpy)
    python analytics/run_analysis.py --source statsbomb --match_id 3788741

    # Video-derived events from the Stone AI pipeline
    python analytics/run_analysis.py --source video \\
        --events output_videos/chicago_u19_vs_scwave_sep14_events.json

    # Specify output directory
    python analytics/run_analysis.py --source statsbomb --match_id 3788741 \\
        --output output_videos/analytics/

Outputs saved to <output_dir>/<match_id>/:
    shot_map.png
    pass_network_home.png
    pass_network_away.png
    xt_top_players.csv
    actions.csv

xT model:
    On first run the xT grid is fitted on 50 La Liga 2015/16 matches and
    cached to analytics/xt_grid_cache.npy (~1–2 min). Subsequent runs load
    the cache instantly.  Delete the .npy file to force a re-fit.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from analytics.config import DEFAULT_OUTPUT_DIR, TOP_PLAYERS_N

# Force UTF-8 stdout/stderr on Windows so non-ASCII player names don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path so `analytics.*` imports work when
# this script is run directly (python analytics/run_analysis.py) or via
# the main_v3.py subprocess call.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_statsbomb(match_id: int, output_dir: str) -> None:
    """
    Full analytics pipeline on a StatsBomb open data match.

    Args:
        match_id:    StatsBomb match ID (e.g. 3788741).
        output_dir:  Root output directory; results go into <output_dir>/<match_id>/.
    """
    from analytics.spadl_pipeline import compute_xt, load_or_fit_xt_grid, top_players_by_xt
    from analytics.statsbomb_loader import load_match
    from analytics.visualizer import pass_network, save_figure, shot_map, xt_bar_chart

    logger.info("[StatsBomb] Loading match %d...", match_id)
    try:
        result = load_match(match_id)
    except (ValueError, KeyError, RuntimeError) as e:
        logger.error("Failed to load match %d: %s", match_id, e)
        sys.exit(1)

    events_df = result["events"]
    logger.info("Loaded %d events", len(events_df))

    # Team names (already stringified by statsbomb_loader)
    teams = (
        [t for t in events_df["team"].dropna().unique().tolist() if t]
        if "team" in events_df.columns
        else []
    )
    home_team = teams[0] if teams else "home"
    away_team = teams[1] if len(teams) > 1 else "away"
    logger.info("Teams: %s vs %s", home_team, away_team)

    # --- Build actions DataFrame for xT scoring ---
    # (bypasses pandera-dependent SPADL conversion; uses kloppy event coords directly)
    x_col, y_col = "coordinates_x", "coordinates_y"
    ex_col, ey_col = "end_coordinates_x", "end_coordinates_y"

    move_events = events_df[
        events_df["event_type"].str.lower().isin(["pass", "carry", "dribble"])
    ].copy()

    actions_df = None
    xt_df = None

    if not move_events.empty and x_col in move_events.columns:
        actions_df = move_events[
            [x_col, y_col, ex_col, ey_col, "player", "team", "event_type"]
        ].copy()
        actions_df = actions_df.rename(
            columns={
                x_col: "start_x",
                y_col: "start_y",
                ex_col: "end_x",
                ey_col: "end_y",
                "player": "player_id",
                "team": "team_id",
                "event_type": "type_name",
            }
        )
        # kloppy normalizes coordinates to [0, 1] — scale to metres (105×68)
        actions_df["start_x"] = actions_df["start_x"] * 105.0
        actions_df["start_y"] = actions_df["start_y"] * 68.0
        actions_df["end_x"] = actions_df["end_x"].fillna(actions_df["start_x"]) * 105.0
        actions_df["end_y"] = actions_df["end_y"].fillna(actions_df["start_y"]) * 68.0
        actions_df["result_name"] = "success"

        # Validate coordinate ranges before xT scoring
        logger.info(
            "Actions coordinate ranges — start_x: [%.2f, %.2f]  start_y: [%.2f, %.2f]",
            float(actions_df["start_x"].min()),
            float(actions_df["start_x"].max()),
            float(actions_df["start_y"].min()),
            float(actions_df["start_y"].max()),
        )
        logger.info(
            "  end_x: [%.2f, %.2f]  end_y: [%.2f, %.2f]",
            float(actions_df["end_x"].min()),
            float(actions_df["end_x"].max()),
            float(actions_df["end_y"].min()),
            float(actions_df["end_y"].max()),
        )
        logger.info(
            "  type_name values: %s",
            sorted(actions_df["type_name"].unique().tolist()),
        )

        # Load / fit the corpus-derived xT grid once, then score
        logger.info("[xT] Loading/fitting xT model...")
        xt_grid = load_or_fit_xt_grid()

        logger.info("[xT] Scoring %d actions...", len(actions_df))
        actions_df = compute_xt(actions_df, grid=xt_grid)

        xt_df = top_players_by_xt(actions_df, n=TOP_PLAYERS_N)
        if not xt_df.empty:
            top = xt_df.iloc[0]["player_id"]
            logger.info(
                "Top player by xT: %s  (total_xt=%.4f)", top, float(xt_df.iloc[0]["total_xt"])
            )
        else:
            logger.warning("xT table is empty — no pass/carry/dribble actions were scored")
    else:
        logger.warning("No pass/carry/dribble events found in match %d", match_id)

    # --- Outputs ---
    out = Path(output_dir) / str(match_id)
    out.mkdir(parents=True, exist_ok=True)

    # Shot map
    logger.info("[Viz] Rendering shot map...")
    try:
        fig, _ = shot_map(events_df, title=f"Shot Map — match {match_id}")
        save_figure(fig, str(out / "shot_map.png"))
    except (ValueError, KeyError, RuntimeError) as e:
        logger.warning("Shot map failed: %s", e)

    # Pass networks
    for team in [home_team, away_team]:
        logger.info("[Viz] Pass network — %s...", team)
        try:
            fig, _ = pass_network(events_df, team=team)
            fname = f"pass_network_{team.replace(' ', '_')}.png"
            save_figure(fig, str(out / fname))
        except (ValueError, KeyError, RuntimeError) as e:
            logger.warning("Pass network failed for %s: %s", team, e)

    # xT bar chart + CSV
    if xt_df is not None and not xt_df.empty:
        logger.info("[Viz] xT bar chart...")
        fig, _ = xt_bar_chart(xt_df)
        save_figure(fig, str(out / "xt_top_players.png"))
        xt_df.to_csv(out / "xt_top_players.csv", index=False)
        logger.info("Saved xt_top_players.csv (%d players)", len(xt_df))

    # Actions CSV
    if actions_df is not None:
        actions_df.to_csv(out / "actions.csv", index=False)
        logger.info("Saved actions.csv (%d rows)", len(actions_df))

    logger.info("All outputs saved to: %s/", out)


def run_video(events_path: str, output_dir: str) -> None:
    """
    Analytics pipeline on video-derived events JSON.

    Args:
        events_path:  Path to *_events.json produced by main_v3.py.
        output_dir:   Root output directory; results go into <output_dir>/<stem>/.
    """
    from analytics.spadl_pipeline import compute_xt, load_or_fit_xt_grid, top_players_by_xt
    from analytics.video_bridge import infer_frame_dimensions, load_video_events
    from analytics.visualizer import save_figure, shot_map, xt_bar_chart

    logger.info("[Video] Loading events from %s...", events_path)
    frame_w, frame_h = infer_frame_dimensions(events_path)
    logger.info("Inferred frame dimensions: %dx%d", frame_w, frame_h)

    actions_df = load_video_events(events_path, frame_w=frame_w, frame_h=frame_h)
    if actions_df.empty:
        logger.warning("No mappable events found. Exiting.")
        return

    logger.info("%d SPADL actions loaded", len(actions_df))

    # Validate coordinate ranges
    logger.info(
        "Coordinate ranges — start_x: [%.2f, %.2f]  start_y: [%.2f, %.2f]",
        float(actions_df["start_x"].min()),
        float(actions_df["start_x"].max()),
        float(actions_df["start_y"].min()),
        float(actions_df["start_y"].max()),
    )

    logger.info("[xT] Loading/fitting xT model...")
    xt_grid = load_or_fit_xt_grid()

    logger.info("[xT] Scoring %d actions...", len(actions_df))
    actions_df = compute_xt(actions_df, grid=xt_grid)
    xt_df = top_players_by_xt(actions_df, n=TOP_PLAYERS_N)

    # --- Outputs ---
    match_stem = Path(events_path).stem.replace("_events", "")
    out = Path(output_dir) / match_stem
    out.mkdir(parents=True, exist_ok=True)

    # Shot map
    shots = actions_df[actions_df["type_name"] == "shot"].rename(
        columns={
            "start_x": "coordinates_x",
            "start_y": "coordinates_y",
            "type_name": "event_type",
            "result_name": "result",
        }
    )
    if not shots.empty:
        logger.info("[Viz] Shot map...")
        fig, _ = shot_map(shots, title=f"Shot Map — {match_stem}")
        save_figure(fig, str(out / "shot_map.png"))

    # xT bar chart + CSV
    if not xt_df.empty:
        logger.info("[Viz] xT bar chart...")
        fig, _ = xt_bar_chart(xt_df)
        save_figure(fig, str(out / "xt_top_players.png"))
        xt_df.to_csv(out / "xt_top_players.csv", index=False)
        logger.info("Saved xt_top_players.csv (%d players)", len(xt_df))

    # Actions CSV
    actions_df.to_csv(out / "actions.csv", index=False)
    logger.info("Saved actions.csv (%d rows)", len(actions_df))

    logger.info("All outputs saved to: %s/", out)


def main() -> None:
    """
    CLI entry point — parse arguments and dispatch to run_statsbomb or run_video.
    """
    parser = argparse.ArgumentParser(
        description="Stone AI Analytics — run data analytics on a match",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["statsbomb", "video"],
        required=True,
        help="Data source: 'statsbomb' for open data, 'video' for pipeline events JSON",
    )
    parser.add_argument(
        "--match_id",
        type=int,
        default=None,
        help="StatsBomb match ID (required for --source statsbomb)",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Path to events JSON (required for --source video)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for plots and CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if args.source == "statsbomb":
        if not args.match_id:
            print("Error: --match_id is required for --source statsbomb")
            sys.exit(1)
        run_statsbomb(args.match_id, args.output)

    elif args.source == "video":
        if not args.events:
            print("Error: --events is required for --source video")
            sys.exit(1)
        if not os.path.exists(args.events):
            print(f"Error: events file not found: {args.events}")
            sys.exit(1)
        run_video(args.events, args.output)


if __name__ == "__main__":
    main()
