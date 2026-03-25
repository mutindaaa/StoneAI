"""
SPADL Pipeline — xT player valuation

Computes Expected Threat (xT) values for SPADL-style actions.

xT model is fitted on a large corpus of StatsBomb open data matches using
Karun Singh's iterative algorithm:

    xT(z) = P(shot|z) * P(goal|shot,z)
           + P(move|z) * sum_z'[T(z'|z) * xT(z')]

The fitted 12×16 grid is cached to analytics/xt_grid_cache.npy on first run,
then loaded from disk on subsequent calls (~5 ms vs ~2 min for fitting).

socceraction's built-in xT/VAEP pipeline is NOT used here because it requires
pandera<0.14 which conflicts with the project's current environment.

References:
    - xT:   https://karun.in/blog/expected-threat.html
    - VAEP: https://github.com/ML-KULeuven/socceraction
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default cache location — lives alongside this module
_DEFAULT_CACHE = Path(__file__).parent / "xt_grid_cache.npy"

# Pitch dimensions used throughout (SPADL standard, metres)
_PITCH_LEN = 105.0
_PITCH_W   = 68.0

# Grid dimensions
_L = 16   # columns (own-goal → opp-goal)
_W = 12   # rows    (left → right touchline)


# ---------------------------------------------------------------------------
# xT corpus building
# ---------------------------------------------------------------------------

def build_training_corpus(
    competition_id: int = 11,
    season_id: int = 27,
    max_matches: int = 50,
) -> pd.DataFrame:
    """
    Load StatsBomb open data matches via kloppy and return a combined actions
    DataFrame suitable for fitting an xT grid.

    Default corpus: La Liga 2015/2016 (competition_id=11, season_id=27),
    which has 380 matches — the largest freely available StatsBomb dataset.

    Args:
        competition_id:  StatsBomb competition ID  (default 11 = La Liga)
        season_id:       StatsBomb season ID       (default 27 = 2015/16)
        max_matches:     Cap on number of matches to load  (default 50)

    Returns:
        DataFrame with columns: start_x, start_y, end_x, end_y,
                                type_name, result_name
    """
    from statsbombpy import sb
    from kloppy import statsbomb

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].tolist()[:max_matches]
    logger.info(
        "Building xT corpus — comp=%d season=%d, loading %d/%d matches...",
        competition_id, season_id, len(match_ids), len(matches),
    )

    parts = []
    loaded = 0
    for match_id in match_ids:
        try:
            ds = statsbomb.load_open_data(match_id=match_id)
            df = ds.to_df(
                "event_type",
                "result",
                "coordinates_x",
                "coordinates_y",
                "end_coordinates_x",
                "end_coordinates_y",
            )
            # kloppy normalizes coordinates to [0, 1]; scale to metres.
            cx = df["coordinates_x"].fillna(0.0)
            cy = df["coordinates_y"].fillna(0.0)
            df["start_x"] = cx * _PITCH_LEN
            df["start_y"] = cy * _PITCH_W
            df["end_x"] = df["end_coordinates_x"].fillna(cx) * _PITCH_LEN
            df["end_y"] = df["end_coordinates_y"].fillna(cy) * _PITCH_W
            df["type_name"]   = df["event_type"].astype(str).str.lower()
            df["result_name"] = df["result"].fillna("fail").astype(str).str.lower()
            parts.append(
                df[["start_x", "start_y", "end_x", "end_y", "type_name", "result_name"]]
            )
            loaded += 1
        except Exception as exc:
            logger.warning("Skipped match %s: %s", match_id, exc)

    if not parts:
        raise RuntimeError("No matches could be loaded for xT corpus.")

    corpus = pd.concat(parts, ignore_index=True)
    logger.info(
        "xT corpus built: %d actions from %d matches (comp=%d, season=%d)",
        len(corpus), loaded, competition_id, season_id,
    )
    return corpus


# ---------------------------------------------------------------------------
# xT grid fitting (Karun Singh algorithm, vectorized)
# ---------------------------------------------------------------------------

def fit_xt_grid(
    corpus_df: pd.DataFrame,
    l: int = _L,
    w: int = _W,
    pitch_len: float = _PITCH_LEN,
    pitch_w: float = _PITCH_W,
    iterations: int = 50,
) -> np.ndarray:
    """
    Fit a w×l xT grid from a corpus of actions using Karun Singh's algorithm.

    Args:
        corpus_df:  DataFrame with start_x/y, end_x/y, type_name, result_name
        l:          Grid columns (pitch length direction)  default 16
        w:          Grid rows   (pitch width direction)    default 12
        iterations: Max iterations for the value-iteration solve

    Returns:
        np.ndarray of shape (w, l) with xT values per zone
    """
    # --- Map coordinates to grid cell indices (vectorized) ---
    def _cell_col(x):
        return np.clip((x / pitch_len * l).astype(int), 0, l - 1)

    def _cell_row(y):
        return np.clip((y / pitch_w * w).astype(int), 0, w - 1)

    sc = _cell_col(corpus_df["start_x"].to_numpy(dtype=float))
    sr = _cell_row(corpus_df["start_y"].to_numpy(dtype=float))
    ec = _cell_col(corpus_df["end_x"].to_numpy(dtype=float))
    er = _cell_row(corpus_df["end_y"].to_numpy(dtype=float))
    types   = corpus_df["type_name"].to_numpy(dtype=str)
    results = corpus_df["result_name"].to_numpy(dtype=str)

    n_actions  = np.zeros((w, l), dtype=float)
    n_shots    = np.zeros((w, l), dtype=float)
    n_goals    = np.zeros((w, l), dtype=float)
    n_moves    = np.zeros((w, l), dtype=float)
    move_mat   = np.zeros((w * l, w * l), dtype=float)   # transition counts

    # Count actions per zone and build transition matrix
    np.add.at(n_actions, (sr, sc), 1)

    shot_mask = types == "shot"
    np.add.at(n_shots, (sr[shot_mask], sc[shot_mask]), 1)
    goal_mask = shot_mask & np.isin(results, ["goal", "success"])
    np.add.at(n_goals, (sr[goal_mask], sc[goal_mask]), 1)

    move_mask = np.isin(types, ["pass", "carry", "dribble"])
    s_flat = sr[move_mask] * l + sc[move_mask]
    e_flat = er[move_mask] * l + ec[move_mask]
    np.add.at(n_moves, (sr[move_mask], sc[move_mask]), 1)
    np.add.at(move_mat, (s_flat, e_flat), 1)

    # Compute per-zone probabilities
    p_shot = np.divide(n_shots, n_actions,  out=np.zeros_like(n_shots),  where=n_actions > 0)
    p_goal = np.divide(n_goals, n_shots,    out=np.zeros_like(n_goals),  where=n_shots  > 0)
    p_move = np.divide(n_moves, n_actions,  out=np.zeros_like(n_moves),  where=n_actions > 0)

    # Row-normalise transition matrix → T[i,j] = P(end in j | start in i, move)
    row_sums = move_mat.sum(axis=1, keepdims=True)
    T = np.divide(move_mat, row_sums, out=np.zeros_like(move_mat), where=row_sums > 0)

    # Value iteration: xT[z] = P(shot|z)*P(goal|z) + P(move|z) * T @ xT
    p_sg_flat = (p_shot * p_goal).flatten()
    p_move_flat = p_move.flatten()
    xt = np.zeros(w * l)
    for i in range(iterations):
        xt_new = p_sg_flat + p_move_flat * (T @ xt)
        delta = float(np.max(np.abs(xt_new - xt)))
        xt = xt_new
        if delta < 1e-6:
            logger.debug("xT converged after %d iterations (delta=%.2e)", i + 1, delta)
            break

    grid = xt.reshape(w, l)
    logger.info(
        "xT grid fitted: min=%.4f  max=%.4f  mean=%.4f",
        float(grid.min()), float(grid.max()), float(grid.mean()),
    )
    return grid


# ---------------------------------------------------------------------------
# Grid cache management
# ---------------------------------------------------------------------------

def load_or_fit_xt_grid(
    cache_path: str | Path | None = None,
    competition_id: int = 11,
    season_id: int = 27,
    max_matches: int = 50,
) -> np.ndarray:
    """
    Return the xT grid, loading from disk cache if available, else fitting
    a new model from StatsBomb open data and saving to cache.

    On first run this downloads and processes up to `max_matches` matches
    (roughly 1–2 minutes for 50 La Liga 2015/16 matches). Subsequent calls
    load the cached .npy file in under 10 ms.

    Args:
        cache_path:     Path to .npy cache file  (default: analytics/xt_grid_cache.npy)
        competition_id: StatsBomb competition ID for corpus
        season_id:      StatsBomb season ID for corpus
        max_matches:    Maximum matches to load for fitting

    Returns:
        np.ndarray of shape (12, 16) with xT values per zone
    """
    cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
    if cache_path.exists():
        grid = np.load(str(cache_path))
        logger.info("Loaded cached xT grid from %s", cache_path)
        return grid

    logger.info(
        "No cached xT grid found at %s — fitting from StatsBomb open data "
        "(comp=%d, season=%d, max_matches=%d). This may take 1–2 minutes...",
        cache_path, competition_id, season_id, max_matches,
    )
    corpus = build_training_corpus(
        competition_id=competition_id,
        season_id=season_id,
        max_matches=max_matches,
    )
    grid = fit_xt_grid(corpus)
    np.save(str(cache_path), grid)
    logger.info("Saved fitted xT grid to %s", cache_path)
    return grid


# ---------------------------------------------------------------------------
# xT scoring
# ---------------------------------------------------------------------------

def compute_xt(
    actions_df: pd.DataFrame,
    grid: np.ndarray | None = None,
    l: int = _L,
    w: int = _W,
) -> pd.DataFrame:
    """
    Compute Expected Threat (xT) value for each action in actions_df.

    xT = xT(end_zone) - xT(start_zone) for pass/carry/dribble actions;
    0.0 for all other action types.

    On first call with grid=None, loads or fits the corpus-derived xT grid
    (see load_or_fit_xt_grid). The grid is cached to disk automatically.

    Args:
        actions_df:  DataFrame with start_x, start_y, end_x, end_y, type_name
        grid:        Optional pre-loaded (w, l) xT grid. If None, the default
                     cached/fitted grid is used.
        l:           Grid columns (default 16)
        w:           Grid rows   (default 12)

    Returns:
        actions_df copy with column "xt_value" added
    """
    if grid is None:
        grid = load_or_fit_xt_grid()

    df = actions_df.copy()

    # Ensure type_name is lowercase for matching
    if "type_name" in df.columns:
        df["type_name"] = df["type_name"].astype(str).str.lower()

    # Validate and log coordinate ranges
    for col in ("start_x", "start_y"):
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                logger.debug(
                    "Coordinate %s: min=%.2f  max=%.2f  nulls=%d",
                    col, float(vals.min()), float(vals.max()),
                    int(df[col].isna().sum()),
                )

    # Vectorized zone lookup
    def _col_idx(x):
        return np.clip((x / _PITCH_LEN * l).astype(int), 0, l - 1)

    def _row_idx(y):
        return np.clip((y / _PITCH_W * w).astype(int), 0, w - 1)

    sx = df["start_x"].fillna(0.0).to_numpy(dtype=float)
    sy = df["start_y"].fillna(0.0).to_numpy(dtype=float)
    ex = df["end_x"].fillna(df["start_x"].fillna(0.0)).to_numpy(dtype=float)
    ey = df["end_y"].fillna(df["start_y"].fillna(0.0)).to_numpy(dtype=float)

    v_start = grid[_row_idx(sy), _col_idx(sx)]
    v_end   = grid[_row_idx(ey), _col_idx(ex)]

    move_mask = df["type_name"].isin(["pass", "carry", "dribble"]).to_numpy()
    xt_raw = np.where(move_mask, np.maximum(0.0, v_end - v_start), 0.0)

    df["xt_value"] = np.round(xt_raw, 5)

    n_nonzero = int((df["xt_value"] > 0).sum())
    n_moves   = int(move_mask.sum())
    logger.info(
        "xT scored: %d/%d move actions have positive xT  "
        "(min=%.5f  max=%.5f  mean=%.5f)",
        n_nonzero, n_moves,
        float(df["xt_value"].min()),
        float(df["xt_value"].max()),
        float(df["xt_value"].mean()),
    )
    return df


# ---------------------------------------------------------------------------
# Player rankings
# ---------------------------------------------------------------------------

def top_players_by_xt(actions_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Rank players by total xT value added across all their actions.

    Returns:
        DataFrame with player_id, team_id, total_xt, actions_count, xt_per_action
    """
    if "xt_value" not in actions_df.columns:
        return pd.DataFrame(
            columns=["player_id", "team_id", "total_xt", "actions_count", "xt_per_action"]
        )

    grp = (
        actions_df.groupby(["player_id", "team_id"])["xt_value"]
        .agg(total_xt="sum", actions_count="count")
        .reset_index()
    )
    grp["xt_per_action"] = (grp["total_xt"] / grp["actions_count"]).round(4)
    grp["total_xt"] = grp["total_xt"].round(4)
    return grp.sort_values("total_xt", ascending=False).head(n).reset_index(drop=True)


def player_action_summary(actions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player action count breakdown by type and result.

    Returns:
        DataFrame with player_id, team_id, and one column per action type
    """
    if actions_df.empty:
        return pd.DataFrame()

    return (
        actions_df.groupby(["player_id", "team_id", "type_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Legacy SPADL conversion helpers (require socceraction — currently broken
# due to pandera version conflict; kept for future use)
# ---------------------------------------------------------------------------

def statsbomb_to_spadl(events_df: pd.DataFrame, home_team_id: str) -> pd.DataFrame:
    """Convert StatsBomb kloppy events to full SPADL format via socceraction."""
    import socceraction.spadl.statsbomb as spadl_sb
    return spadl_sb.convert_to_actions(events_df, home_team_id=home_team_id)


def video_events_to_spadl(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Add type_id / result_id integer columns expected by some socceraction models."""
    import socceraction.spadl.config as spadl_config

    type_name_to_id   = {t.name: t.action_type_id for t in spadl_config.actiontypes}
    result_name_to_id = {r.name: r.result_id       for r in spadl_config.results}

    df = actions_df.copy()
    df["type_id"]     = df["type_name"].map(type_name_to_id).fillna(0).astype(int)
    df["result_id"]   = df["result_name"].map(result_name_to_id).fillna(1).astype(int)
    df["bodypart_id"] = 0
    df["bodypart_name"] = "foot"
    return df
