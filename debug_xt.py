"""
Debug script — diagnose xT = 0 issues in the StatsBomb analytics pipeline.

Run from project root:
    python debug_xt.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force UTF-8 output on Windows (avoids cp1252 crash on non-ASCII player names)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from analytics.statsbomb_loader import load_match
from analytics.spadl_pipeline import compute_xt, load_or_fit_xt_grid

# ── Load match ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CHECK 1 — RAW EVENT TYPES")
print("="*60)
result = load_match(3788741)
df = result["events"]
print(f"Total events: {len(df)}")
print("Unique event_type values:")
for v in sorted(df["event_type"].dropna().unique()):
    n = (df["event_type"] == v).sum()
    print(f"  {v!r:<40}  n={n}")

# ── Coordinate ranges ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CHECK 2 — COORDINATE RANGES")
print("="*60)
coord_cols = [c for c in ["coordinates_x","coordinates_y","end_coordinates_x","end_coordinates_y"] if c in df.columns]
print(df[coord_cols].describe().to_string())
print(f"\nNull counts:")
print(df[coord_cols].isna().sum().to_string())

# ── xT grid ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CHECK 3 — XT GRID")
print("="*60)
grid = load_or_fit_xt_grid()
print(f"Shape: {grid.shape}")
print(f"min={grid.min():.5f}  max={grid.max():.5f}  mean={grid.mean():.5f}")
print(f"Non-zero cells: {(grid > 0).sum()} / {grid.size}")
print("\nGrid (rounded to 4dp):")
print(np.round(grid, 4))

# ── Build actions_df exactly as run_analysis.py does ─────────────────────────
print("\n" + "="*60)
print("CHECK 4 — ACTIONS BUILD + TYPE_NAME CASING")
print("="*60)
x_col, y_col   = "coordinates_x",     "coordinates_y"
ex_col, ey_col = "end_coordinates_x", "end_coordinates_y"

move_events = df[df["event_type"].str.lower().isin(["pass", "carry", "dribble"])].copy()
print(f"Move events after .str.lower() filter: {len(move_events)}")
print(f"Unique event_type in move_events (BEFORE rename): {move_events['event_type'].unique().tolist()}")

actions_df = move_events[[x_col, y_col, ex_col, ey_col, "player", "team", "event_type"]].copy()
actions_df = actions_df.rename(columns={
    x_col: "start_x",   y_col: "start_y",
    ex_col: "end_x",    ey_col: "end_y",
    "player": "player_id", "team": "team_id",
    "event_type": "type_name",
})
actions_df["start_x"] = actions_df["start_x"] * 105.0
actions_df["start_y"] = actions_df["start_y"] * 68.0
actions_df["end_x"]   = actions_df["end_x"].fillna(actions_df["start_x"]) * 105.0
actions_df["end_y"]   = actions_df["end_y"].fillna(actions_df["start_y"]) * 68.0
actions_df["result_name"] = "success"

print(f"\ntype_name values AFTER rename (before compute_xt lowercases them):")
print(f"  {actions_df['type_name'].unique().tolist()}")
print(f"\nCoordinate ranges (post-scale to metres):")
print(f"  start_x: [{actions_df['start_x'].min():.2f}, {actions_df['start_x'].max():.2f}]")
print(f"  start_y: [{actions_df['start_y'].min():.2f}, {actions_df['start_y'].max():.2f}]")
print(f"  end_x:   [{actions_df['end_x'].min():.2f},  {actions_df['end_x'].max():.2f}]")
print(f"  end_y:   [{actions_df['end_y'].min():.2f},  {actions_df['end_y'].max():.2f}]")

# ── Run compute_xt and inspect output ─────────────────────────────────────────
print("\n" + "="*60)
print("CHECK 5 — COMPUTE_XT OUTPUT")
print("="*60)
scored = compute_xt(actions_df, grid=grid)
n_nonzero = (scored["xt_value"] > 0).sum()
print(f"Actions scored:  {len(scored)}")
print(f"Non-zero xT:     {n_nonzero} / {len(scored)}")
print(f"xT range:        min={scored['xt_value'].min():.5f}  max={scored['xt_value'].max():.5f}  mean={scored['xt_value'].mean():.5f}")

print(f"\ntype_name values INSIDE scored df:")
print(f"  {scored['type_name'].unique().tolist()}")

move_mask = scored["type_name"].isin(["pass", "carry", "dribble"])
print(f"\nRows matching move mask (pass/carry/dribble): {move_mask.sum()}")

print("\nSample rows (first 5 pass/carry/dribble):")
sample = scored[move_mask].head(5)[["player_id", "type_name", "start_x", "start_y", "end_x", "end_y", "xt_value"]]
print(sample.to_string())

# ── Manual zone lookup for one row ────────────────────────────────────────────
print("\n" + "="*60)
print("CHECK 6 — MANUAL ZONE LOOKUP FOR ONE FORWARD PASS")
print("="*60)
passes = scored[(scored["type_name"] == "pass") & (scored["end_x"] > scored["start_x"])].head(1)
if not passes.empty:
    row = passes.iloc[0]
    sx, sy = row["start_x"], row["start_y"]
    ex, ey = row["end_x"],   row["end_y"]
    l, w = 16, 12
    sc = int(np.clip(sx / 105.0 * l, 0, l-1))
    sr = int(np.clip(sy / 68.0  * w, 0, w-1))
    ec = int(np.clip(ex / 105.0 * l, 0, l-1))
    er = int(np.clip(ey / 68.0  * w, 0, w-1))
    v_start = grid[sr, sc]
    v_end   = grid[er, ec]
    print(f"Player:     {row['player_id']}")
    print(f"start:      ({sx:.2f}, {sy:.2f})  → zone ({sr}, {sc})  xT={v_start:.5f}")
    print(f"end:        ({ex:.2f}, {ey:.2f})  → zone ({er}, {ec})  xT={v_end:.5f}")
    print(f"delta xT:   {max(0, v_end - v_start):.5f}  (stored: {row['xt_value']:.5f})")
else:
    print("No forward passes found.")
