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

def pass_network(
    events_df: pd.DataFrame,
    team: str,
    min_passes: int = 3,
    pitch_type: str = "statsbomb",
    title: str | None = None,
) -> tuple:
    """
    Draw a pass network on a half-pitch (VerticalPitch, half=True).

    Node size is proportional to total passes by that player.
    Line width is proportional to pass count between each pair.
    Only pairs with >= min_passes passes are connected.

    Pass receivers are inferred from the sequential event order: for each
    successful pass, the next event from the same team provides the receiver.

    Coordinates are expected in kloppy [0, 1] range and are scaled to
    StatsBomb 120 × 80 units before computing average positions.

    Args:
        events_df:   Full kloppy events DataFrame
        team:        Team name to draw
        min_passes:  Minimum passes between a pair to draw a connection
        pitch_type:  mplsoccer pitch type (default 'statsbomb')
        title:       Plot title (auto-generated if None)

    Returns:
        (fig, ax)
    """
    from mplsoccer import VerticalPitch

    title = title or f"Pass Network — {team}"
    team_col = "team" if "team" in events_df.columns else "team_id"
    x_col = "coordinates_x" if "coordinates_x" in events_df.columns else "start_x"
    y_col = "coordinates_y" if "coordinates_y" in events_df.columns else "start_y"

    pitch = VerticalPitch(
        pitch_type="statsbomb",
        pitch_length=120,
        pitch_width=80,
        half=True,
        pitch_color="#0d1117",
        line_color="#6b7280",
        pad_top=10,
        pad_bottom=10,
        pad_left=10,
        pad_right=10,
    )
    fig, ax = pitch.draw(figsize=(12, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_title(title, color="white", fontsize=14, pad=10)

    # Flip so defensive end is at top, attacking at bottom
    ax.invert_yaxis()

    # ── All team events → positions + pass counts ─────────────────────────────
    # Use every team event (not just passes) for stable average-position
    # computation so even low-volume passers get placed correctly.
    team_events = events_df[
        (events_df[team_col] == team) & events_df["player"].notna()
    ].copy()

    if team_events.empty or "player" not in team_events.columns:
        ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=12)
        return fig, ax

    # kloppy normalizes all coordinates to [0, 1]; scale to StatsBomb 120×80
    team_events["x_sb"] = team_events[x_col].clip(0, 1) * 120.0
    team_events["y_sb"] = team_events[y_col].clip(0, 1) * 80.0

    # Average position from all events with valid coordinates
    avg_pos = (
        team_events.dropna(subset=["x_sb", "y_sb"])
        .groupby("player")[["x_sb", "y_sb"]]
        .mean()
        .rename(columns={"x_sb": "avg_x", "y_sb": "avg_y"})
    )

    # Pass counts per player
    et_all = _to_str_lower(team_events["event_type"])
    pass_counts = (
        team_events[et_all == "pass"]
        .groupby("player")
        .size()
        .rename("n_passes")
    )

    # Node threshold: ≥2 passes (lower than connection threshold of 3)
    player_df = avg_pos.join(pass_counts).fillna({"n_passes": 0})
    player_df = player_df[player_df["n_passes"] >= 2].dropna(subset=["avg_x", "avg_y"])

    if player_df.empty:
        ax.text(0.5, 0.5, "Could not compute player positions",
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=12)
        return fig, ax

    # ── Infer pass connections from sequential events ─────────────────────────
    df_team_seq = (
        events_df[events_df[team_col] == team]
        .sort_values(["period_id", "timestamp"])
        .reset_index(drop=True)
    )
    df_team_seq["receiver"] = df_team_seq["player"].shift(-1)

    pass_pairs = df_team_seq[
        (_to_str_lower(df_team_seq["event_type"]) == "pass") &
        (df_team_seq["player"].astype(str) != df_team_seq["receiver"].astype(str)) &
        df_team_seq["receiver"].notna()
    ].copy()

    connections = (
        pass_pairs
        .groupby([pass_pairs["player"].astype(str), pass_pairs["receiver"].astype(str)])
        .size()
        .reset_index(name="n")
    )
    connections.columns = ["passer", "receiver", "n"]
    # Connection threshold stays at min_passes (default 3)
    connections = connections[connections["n"] >= min_passes]

    # Debug: print coordinate ranges so caller can verify spread
    print(f"[pass_network] {team}: avg_x range [{player_df['avg_x'].min():.1f}, {player_df['avg_x'].max():.1f}]  "
          f"avg_y range [{player_df['avg_y'].min():.1f}, {player_df['avg_y'].max():.1f}]  "
          f"nodes={len(player_df)}")

    # ── Draw connection lines ─────────────────────────────────────────────────
    # VerticalPitch axis swap: first arg = y (horizontal), second = x (vertical)
    max_n = connections["n"].max() if not connections.empty else 1
    player_index = player_df.index.astype(str)

    for _, row in connections.iterrows():
        if row["passer"] in player_index and row["receiver"] in player_index:
            px = player_df.loc[player_df.index.astype(str) == row["passer"]]
            rx = player_df.loc[player_df.index.astype(str) == row["receiver"]]
            if px.empty or rx.empty:
                continue
            lw = 1.5 + (row["n"] / max_n) * 8.0
            pitch.lines(
                px["avg_y"].iloc[0], px["avg_x"].iloc[0],
                rx["avg_y"].iloc[0], rx["avg_x"].iloc[0],
                lw=lw, color="white", alpha=0.4, zorder=2, ax=ax,
            )

    # ── Draw player nodes ────────────────────────────────────────────────────
    # VerticalPitch: first positional arg = y (horizontal), second = x (vertical)
    pitch.scatter(
        player_df["avg_y"], player_df["avg_x"],
        s=player_df["n_passes"] * 20 + 200,
        color="#f7c59f", edgecolors="white", linewidths=1.5,
        zorder=4, ax=ax,
    )

    # Labels: every node, last name only, offset (2, 2) so label clears the dot
    for player, row in player_df.iterrows():
        parts = str(player).split()
        short = parts[-1] if len(parts) > 1 else str(player)
        ax.annotate(
            short, (row["avg_y"], row["avg_x"]),
            fontsize=8, color="white", ha="left", va="bottom",
            xytext=(2, 2), textcoords="offset points",
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
