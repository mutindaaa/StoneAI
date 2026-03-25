"""
Position-Based Jersey Number Assignment

When no roster is provided (zero-config / auto mode), assigns traditional 1-11
jersey numbers based on each player's average field position across the match.

Traditional soccer numbering:
  1  - GK  (deepest in own half)
  2  - RB  (right side, defensive third)
  3  - LB  (left side, defensive third)
  4  - CB  (center, defensive third)
  5  - CB  (center, defensive third)
  6  - CDM (center, lower midfield)
  7  - RW  (right, midfield/attack)
  8  - CM  (center midfield)
  9  - ST  (center, attacking third)
  10 - AM  (center, attacking-mid)
  11 - LW  (left, midfield/attack)

This is applied per-team after SigLIP assigns team 1/2 to each tracking ID.
The field is divided into depth layers (Y axis) and horizontal zones (X axis).
"""

from typing import Dict, List, Any, Tuple
import numpy as np




def assign_position_numbers(
    player_tracks: List[Dict],
    player_id_map: Dict[int, str],
    match_config: Dict[str, Any],
) -> Dict[str, int]:
    """
    Assign traditional jersey numbers (1-11) to players based on average field position.

    Only runs when the team's roster in match_config is empty (zero-config mode).

    Args:
        player_tracks: Per-frame player tracking data (with position_transformed)
        player_id_map: tracking_id -> player_id mapping
        match_config: Match configuration (used to check if rosters are empty)

    Returns:
        Dict mapping player_id -> assigned jersey number
    """
    result: Dict[str, int] = {}

    # Process all teams — even if a roster is defined, position-based numbers
    # are used as a fallback display label when OCR jersey detection finds nothing.
    # The caller (main_v3.py) only writes the number when jersey_number is None,
    # so real OCR results are never overwritten.
    teams_to_process = {}
    for team_key in ['team_home', 'team_away']:
        team = match_config.get(team_key, {})
        team_id = team.get('id', team_key)
        teams_to_process[team_id] = []

    # Collect average Y position and average X position for each player_id
    avg_positions: Dict[str, Tuple[float, float]] = {}  # player_id -> (avg_x, avg_y)
    pos_counts: Dict[str, int] = {}

    for frame_players in player_tracks:
        for track_id, pdata in frame_players.items():
            player_id = player_id_map.get(track_id)
            if player_id is None:
                continue
            pos = pdata.get('position_transformed')
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if player_id not in avg_positions:
                avg_positions[player_id] = (0.0, 0.0)
                pos_counts[player_id] = 0
            sx, sy = avg_positions[player_id]
            n = pos_counts[player_id]
            avg_positions[player_id] = (sx + x, sy + y)
            pos_counts[player_id] = n + 1

    # Compute actual averages
    for pid in list(avg_positions.keys()):
        n = pos_counts[pid]
        if n > 0:
            sx, sy = avg_positions[pid]
            avg_positions[pid] = (sx / n, sy / n)

    # Group players by team
    team_players: Dict[str, List[str]] = {tid: [] for tid in teams_to_process}
    for pid in avg_positions:
        for team_id in teams_to_process:
            if pid.startswith(f"{team_id}_"):
                team_players[team_id].append(pid)
                break

    for team_id, players in team_players.items():
        if not players:
            continue

        positions = [avg_positions[pid] for pid in players]
        ys = [p[1] for p in positions]
        xs = [p[0] for p in positions]

        if not ys:
            continue

        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        y_range = max(y_max - y_min, 1.0)
        x_range = max(x_max - x_min, 1.0)

        # Sort by Y (deepest = smallest Y = defender side, highest = attacker)
        players_sorted = sorted(players, key=lambda p: avg_positions[p][1])

        n = len(players_sorted)
        jersey_assignments = _assign_1_to_11(players_sorted, avg_positions, n)

        for pid, jersey in jersey_assignments.items():
            result[pid] = jersey

    return result


def _assign_1_to_11(
    players_sorted_by_y: List[str],
    avg_positions: Dict[str, Tuple[float, float]],
    n: int,
) -> Dict[str, int]:
    """
    Assign jersey 1-11 to a sorted list of players using depth+lateral grouping.

    Layout (Y increases away from own goal):
    - Deepest 1 player: GK (jersey 1)
    - Next 3-4 players: defenders (2, 3, 4, 5) sorted by X
    - Middle 3-4 players: midfielders (6, 7, 8) or (6, 7, 8) sorted by X
    - Furthest 1-3 players: forwards (9, 10, 11) sorted by X

    For fewer than 11 players, assigns the appropriate jerseys for the depth tier.
    """
    result: Dict[str, int] = {}

    if n == 0:
        return result

    # Divide into 4 depth tiers
    gk_count     = 1
    def_count    = min(4, max(1, round(n * 0.36)))  # ~4 of 11
    mid_count    = min(4, max(1, round(n * 0.36)))  # ~4 of 11
    fwd_count    = max(1, n - gk_count - def_count - mid_count)

    # Adjust so totals match n
    while gk_count + def_count + mid_count + fwd_count > n:
        if fwd_count > 1:
            fwd_count -= 1
        elif mid_count > 1:
            mid_count -= 1
        elif def_count > 1:
            def_count -= 1
    while gk_count + def_count + mid_count + fwd_count < n:
        mid_count += 1

    tiers = [
        (gk_count,  [1]),
        (def_count, [3, 4, 5, 2]),   # sorted left→right: LB, CB, CB, RB
        (mid_count, [7, 6, 8, 10]),  # sorted left→right: LW/RM, CDM, CM, RW
        (fwd_count, [11, 9, 10]),    # sorted left→right: LW, ST, RW
    ]

    idx = 0
    for tier_count, jersey_list in tiers:
        tier_players = players_sorted_by_y[idx: idx + tier_count]
        # Sort within tier by X (left to right)
        tier_players_sorted = sorted(tier_players, key=lambda p: avg_positions[p][0])
        for i, pid in enumerate(tier_players_sorted):
            jersey = jersey_list[i] if i < len(jersey_list) else jersey_list[-1]
            result[pid] = jersey
        idx += tier_count

    return result
