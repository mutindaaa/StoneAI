"""
Pitch Visualizations — mplsoccer

All visualization functions return (fig, ax) tuples suitable for:
  - st.pyplot(fig)  in Streamlit
  - fig.savefig(path)  for saving to disk

Coordinate system: StatsBomb (120 × 80 yards) unless pitch_type overridden.
For video-derived events, coordinates are already normalized to [0-105] × [0-68]
metres — use pitch_type='custom' with appropriate dimensions.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Streamlit + CLI)
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Shot map
# ---------------------------------------------------------------------------

def shot_map(
    events_df: pd.DataFrame,
    team: str | None = None,
    title: str = "Shot Map",
    pitch_type: str = "statsbomb",
) -> tuple:
    """
    Draw a shot map. Dots sized by xG, green = goal, red = no goal.

    Args:
        events_df:  Events DataFrame (kloppy output or video events)
        team:       Filter to this team name/id (None = all)
        title:      Plot title
        pitch_type: mplsoccer pitch type ('statsbomb', 'custom', etc.)

    Returns:
        (fig, ax)
    """
    from mplsoccer import VerticalPitch

    shots = events_df[events_df["event_type"].str.lower().isin(["shot", "shot_on_target"])]
    if team:
        shots = shots[shots.get("team", shots.get("team_id", pd.Series(dtype=str))) == team]

    pitch = VerticalPitch(
        pitch_type=pitch_type,
        half=True,
        pitch_color="#1a1a2e",
        line_color="#4a4a8a",
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=(8, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=14, pad=15)

    if shots.empty:
        ax.text(0.5, 0.5, "No shots detected", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=12)
        return fig, ax

    # Coordinate columns — handle both StatsBomb and video naming
    x_col = "coordinates_x" if "coordinates_x" in shots.columns else "start_x"
    y_col = "coordinates_y" if "coordinates_y" in shots.columns else "start_y"

    # Size by xG if available, else uniform
    if "shot_statsbomb_xg" in shots.columns:
        sizes = shots["shot_statsbomb_xg"].fillna(0.05) * 1200 + 100
    else:
        sizes = 200

    # Color by goal / no goal
    result_col = "result" if "result" in shots.columns else "result_name"
    if result_col in shots.columns:
        goal_mask = shots[result_col].str.lower().isin(["success", "goal"])
        colors = ["#00d4aa" if g else "#ff6b6b" for g in goal_mask]
    else:
        colors = "#ff6b6b"

    pitch.scatter(
        shots[x_col], shots[y_col],
        s=sizes, c=colors,
        edgecolors="white", linewidths=0.5,
        alpha=0.85, zorder=3, ax=ax,
    )

    n_shots = len(shots)
    n_goals = sum(1 for c in colors if c == "#00d4aa") if isinstance(colors, list) else 0
    ax.text(
        0.02, 0.02,
        f"Shots: {n_shots}  Goals: {n_goals}",
        transform=ax.transAxes, color="white", fontsize=10,
    )

    return fig, ax


# ---------------------------------------------------------------------------
# Pass network
# ---------------------------------------------------------------------------

def pass_network(
    events_df: pd.DataFrame,
    team: str,
    min_passes: int = 3,
    pitch_type: str = "statsbomb",
    title: str | None = None,
) -> tuple:
    """
    Draw a pass network showing average player positions and connection thickness
    proportional to number of passes between each pair.

    Args:
        events_df:   Events DataFrame
        team:        Team name/id to draw
        min_passes:  Minimum passes between a pair to draw a line
        pitch_type:  mplsoccer pitch type
        title:       Plot title (auto-generated if None)

    Returns:
        (fig, ax)
    """
    from mplsoccer import Pitch

    title = title or f"Pass Network — {team}"

    # Filter passes for this team
    team_col = "team" if "team" in events_df.columns else "team_id"
    passes = events_df[
        (events_df["event_type"].str.lower() == "pass") &
        (events_df[team_col] == team)
    ].copy()

    pitch = Pitch(
        pitch_type=pitch_type,
        pitch_color="#1a1a2e",
        line_color="#4a4a8a",
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=14, pad=10)

    if passes.empty or "player" not in passes.columns:
        ax.text(0.5, 0.5, "Not enough pass data", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=12)
        return fig, ax

    x_col = "coordinates_x" if "coordinates_x" in passes.columns else "start_x"
    y_col = "coordinates_y" if "coordinates_y" in passes.columns else "start_y"

    # Average position per player
    avg_pos = (
        passes.groupby("player")[[x_col, y_col]]
        .mean()
        .rename(columns={x_col: "avg_x", y_col: "avg_y"})
    )
    pass_counts = passes.groupby("player").size().rename("n_passes")
    player_df = avg_pos.join(pass_counts)

    # Plot nodes
    pitch.scatter(
        player_df["avg_x"], player_df["avg_y"],
        s=player_df["n_passes"] * 20 + 200,
        color="#f7c59f", edgecolors="white", linewidths=1.5,
        zorder=4, ax=ax,
    )
    for player, row in player_df.iterrows():
        short = str(player).split()[-1] if " " in str(player) else str(player)
        ax.annotate(
            short, (row["avg_x"], row["avg_y"]),
            fontsize=7, color="white", ha="center", va="bottom",
            xytext=(0, 8), textcoords="offset points",
        )

    return fig, ax


# ---------------------------------------------------------------------------
# Player heatmap
# ---------------------------------------------------------------------------

def player_heatmap(
    positions_df: pd.DataFrame,
    player_id: str | None = None,
    x_col: str = "start_x",
    y_col: str = "start_y",
    pitch_type: str = "statsbomb",
    title: str | None = None,
) -> tuple:
    """
    Draw a kernel-density heatmap of player positions.

    Args:
        positions_df:  DataFrame with x/y columns (SPADL actions or tracking data)
        player_id:     Filter to this player (None = all rows)
        x_col, y_col:  Column names for coordinates
        pitch_type:    mplsoccer pitch type
        title:         Plot title

    Returns:
        (fig, ax)
    """
    from mplsoccer import Pitch

    if player_id:
        df = positions_df[positions_df["player_id"] == player_id]
        title = title or f"Heatmap — {player_id}"
    else:
        df = positions_df
        title = title or "Player Heatmap"

    pitch = Pitch(pitch_type=pitch_type, pitch_color="#1a1a2e", line_color="#4a4a8a")
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=14)

    if df.empty or x_col not in df.columns:
        ax.text(0.5, 0.5, "No position data", transform=ax.transAxes,
                ha="center", va="center", color="white")
        return fig, ax

    pitch.kdeplot(
        df[x_col], df[y_col],
        ax=ax, fill=True,
        cmap="hot", alpha=0.75, levels=100, thresh=0.01,
    )
    return fig, ax


# ---------------------------------------------------------------------------
# xT bar chart (top players)
# ---------------------------------------------------------------------------

def xt_bar_chart(xt_df: pd.DataFrame, title: str = "Top Players by xT") -> tuple:
    """
    Horizontal bar chart of top players by xT value.

    Args:
        xt_df:  Output of spadl_pipeline.top_players_by_xt()

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(xt_df) * 0.45)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if xt_df.empty:
        ax.text(0.5, 0.5, "No xT data available", ha="center", va="center",
                color="white", transform=ax.transAxes)
        return fig, ax

    labels = xt_df["player_id"].astype(str)
    values = xt_df["total_xt"]

    bars = ax.barh(labels, values, color="#00d4aa", edgecolor="none", height=0.6)
    ax.set_xlabel("Total xT", color="white")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#4a4a8a")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=8)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(fig, path: str, dpi: int = 150):
    """Save a matplotlib figure to disk."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
