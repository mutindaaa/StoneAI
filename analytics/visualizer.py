"""
Pitch Visualizations — mplsoccer

All visualization functions return (fig, ax) tuples suitable for:
  - st.pyplot(fig)  in Streamlit
  - fig.savefig(path)  for saving to disk

Coordinate system:
  kloppy normalizes all coordinates to [0, 1].  Every function here
  scales to StatsBomb units (x * 120, y * 80) before plotting on a
  pitch_type='statsbomb' canvas.  For video-derived events whose
  coordinates are already in metres (0-105 × 0-68), pass
  pitch_type='custom' and scale accordingly.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Streamlit + CLI
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_str_lower(series: pd.Series) -> pd.Series:
    """Stringify a column (handles kloppy enum objects) then lowercase."""
    return series.astype(str).str.lower()


def _scale_statsbomb(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Return (x, y) Series scaled from kloppy [0, 1] to StatsBomb units
    (120 × 80).  Clips to valid range.
    """
    x = df[x_col].clip(0, 1) * 120.0
    y = df[y_col].clip(0, 1) * 80.0
    return x, y


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
    Draw a shot map on a half-pitch.

    Dots are sized by xG (default size 100 when xG is missing),
    coloured green for goals and red for all other outcomes.

    Coordinates are expected in kloppy [0, 1] range and are scaled to
    StatsBomb 120 × 80 units before plotting.

    Args:
        events_df:  kloppy events DataFrame (or video events with renamed cols)
        team:       Filter to this team name (None = all teams)
        title:      Plot title
        pitch_type: mplsoccer pitch type (default 'statsbomb')

    Returns:
        (fig, ax)
    """
    from mplsoccer import VerticalPitch

    # Normalise event_type to lowercase strings — kloppy returns 'SHOT' (uppercase)
    et = _to_str_lower(events_df["event_type"])
    shots = events_df[et.isin(["shot", "shot_on_target"])].copy()

    if team:
        team_col = "team" if "team" in shots.columns else "team_id"
        shots = shots[shots[team_col] == team]

    pitch = VerticalPitch(
        pitch_type=pitch_type,
        half=True,
        pitch_color="#1a1a2e",
        line_color="#4a4a8a",
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=14, pad=15)

    if shots.empty:
        ax.text(0.5, 0.5, "No shots detected", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=12)
        return fig, ax

    # Coordinate columns — handle kloppy naming and video-bridge naming
    x_col = "coordinates_x" if "coordinates_x" in shots.columns else "start_x"
    y_col = "coordinates_y" if "coordinates_y" in shots.columns else "start_y"

    # Scale from kloppy [0, 1] → StatsBomb 120 × 80
    x, y = _scale_statsbomb(shots, x_col, y_col)

    # Size by xG if available, else uniform default of 100
    if "shot_statsbomb_xg" in shots.columns:
        sizes = shots["shot_statsbomb_xg"].fillna(0.05) * 1200 + 100
    else:
        sizes = 100

    # Color by outcome — stringify result first (kloppy may return enum objects)
    result_col = "result" if "result" in shots.columns else "result_name"
    if result_col in shots.columns:
        results = _to_str_lower(shots[result_col])
        goal_mask = results.isin(["goal", "success"])
        colors = ["#00d4aa" if g else "#ff6b6b" for g in goal_mask]
    else:
        goal_mask = pd.Series([False] * len(shots), index=shots.index)
        colors = "#ff6b6b"

    pitch.scatter(
        x, y,
        s=sizes, c=colors,
        edgecolors="white", linewidths=0.5,
        alpha=0.85, zorder=3, ax=ax,
    )

    n_shots = len(shots)
    n_goals = int(goal_mask.sum()) if isinstance(goal_mask, pd.Series) else 0
    ax.text(
        0.02, 0.02,
        f"Shots: {n_shots}  Goals: {n_goals}",
        transform=ax.transAxes, color="white", fontsize=10,
    )

    return fig, ax


# ---------------------------------------------------------------------------
# Pass network
# ---------------------------------------------------------------------------

def pass_network(events_df: pd.DataFrame, team: str, title: str = "") -> tuple:
    """Draw pass network for a team using average player positions."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    # DejaVu Sans ships with matplotlib and supports full Unicode (Turkish, etc.)
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    # Use a standard horizontal pitch with statsbomb coords (120x80)
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#0d1117',
        line_color='#6b7280',
        pad_top=15,
        pad_bottom=15,
        pad_left=15,
        pad_right=15,
    )
    fig, ax = pitch.draw(figsize=(14, 9))

    # Filter to this team's events
    team_events = events_df[
        events_df["team"].astype(str) == str(team)
    ].copy()

    passes = team_events[
        team_events["event_type"].astype(str).str.upper() == "PASS"
    ].copy()

    if passes.empty:
        ax.set_title(title or f"Pass Network — {team}", color="white", fontsize=14)
        return fig, ax

    # Scale coordinates from kloppy [0,1] to StatsBomb 120x80
    passes["x"] = passes["coordinates_x"].fillna(0.5) * 120
    passes["y"] = passes["coordinates_y"].fillna(0.5) * 80

    # Use ALL team events for average position so the full tactical shape is
    # captured — using only passes causes players to cluster where they pass
    # rather than where they actually operate on the pitch.
    team_events["x"] = team_events["coordinates_x"].fillna(0.5) * 120
    team_events["y"] = team_events["coordinates_y"].fillna(0.5) * 80

    avg_pos = (
        team_events.dropna(subset=["x", "y"])
        .groupby("player")[["x", "y"]]
        .mean()
        .reset_index()
        .rename(columns={"x": "avg_x", "y": "avg_y"})
    )

    # Count passes per player for node sizing; left-join so all positioned
    # players appear even if they had very few passes (e.g. late substitutes)
    pass_counts = passes.groupby("player").size().reset_index(name="n_passes")
    avg_pos = avg_pos.merge(pass_counts, on="player", how="left").fillna({"n_passes": 0})
    avg_pos = avg_pos[avg_pos["n_passes"] >= 1]  # only players who actually passed

    # Infer receiver as the next event's player for the same team
    passes = passes.reset_index(drop=True)
    passes["receiver"] = passes["player"].shift(-1)
    passes.loc[passes.index[-1], "receiver"] = None

    # Count passes between pairs
    pairs = (
        passes.dropna(subset=["receiver"])
        .groupby(["player", "receiver"])
        .size()
        .reset_index(name="n")
    )
    pairs = pairs[pairs["n"] >= 2]

    # Draw connection lines
    pos_dict = avg_pos.set_index("player")[["avg_x", "avg_y"]].to_dict("index")

    for _, row in pairs.iterrows():
        p1, p2 = row["player"], row["receiver"]
        if p1 not in pos_dict or p2 not in pos_dict:
            continue
        x1, y1 = pos_dict[p1]["avg_x"], pos_dict[p1]["avg_y"]
        x2, y2 = pos_dict[p2]["avg_x"], pos_dict[p2]["avg_y"]
        lw = min(row["n"] * 0.4, 6)
        pitch.lines(x1, y1, x2, y2, lw=lw, color="white", alpha=0.3, ax=ax)

    # Draw player nodes
    node_sizes = (avg_pos["n_passes"] / avg_pos["n_passes"].max() * 800 + 200).values
    pitch.scatter(
        avg_pos["avg_x"].values,
        avg_pos["avg_y"].values,
        s=node_sizes,
        color="#f5c07a",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        ax=ax,
    )

    # Label every player
    for _, row in avg_pos.iterrows():
        last_name = str(row["player"]).split()[-1]
        ax.annotate(
            last_name,
            xy=(row["avg_x"], row["avg_y"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color="white",
            zorder=6,
        )

    ax.set_title(title or f"Pass Network — {team}", color="white", fontsize=14, pad=15)
    fig.patch.set_facecolor('#0d1117')

    # Debug: print coordinate ranges so we can verify
    print(f"  [{team}] avg_x range: {avg_pos['avg_x'].min():.1f} – {avg_pos['avg_x'].max():.1f}")
    print(f"  [{team}] avg_y range: {avg_pos['avg_y'].min():.1f} – {avg_pos['avg_y'].max():.1f}")
    print(f"  [{team}] players: {avg_pos['player'].tolist()}")

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
    Horizontal bar chart of top players by total xT value.

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
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color="white", fontsize=8,
        )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(fig, path: str, dpi: int = 150):
    """Save a matplotlib figure to disk and close it."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
