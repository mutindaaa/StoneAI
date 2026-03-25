"""
StatsBomb Open Data Loader

Loads StatsBomb event data via kloppy, normalizes coordinates to a
standard model, and returns DataFrames ready for SPADL conversion.

StatsBomb open data covers 40+ competitions including:
- La Liga (full Messi era at Barcelona)
- FIFA World Cups (men's and women's)
- NWSL, Champions League finals

No API key needed for open data.
"""

import pandas as pd


def load_match(match_id: int) -> dict:
    """
    Load a single StatsBomb match via kloppy.

    Args:
        match_id: StatsBomb match ID (e.g. 3788741)

    Returns:
        dict with keys:
            - dataset:   kloppy EventDataset (normalized)
            - events:    pandas DataFrame with standardized columns
            - match_id:  the match_id passed in
    """
    from kloppy import statsbomb

    dataset = statsbomb.load_open_data(match_id=match_id)

    events_df = dataset.to_df(
        "event_id",
        "event_type",
        "result",
        "player",
        "team",
        "coordinates_x",
        "coordinates_y",
        "end_coordinates_x",
        "end_coordinates_y",
        "period_id",
        "timestamp",
    )

    # kloppy returns Player/Team model objects — stringify them so pandas
    # groupby/sort/merge operations work correctly downstream.
    if "player" in events_df.columns:
        events_df["player"] = events_df["player"].apply(
            lambda p: p.name if hasattr(p, "name") and p is not None else str(p) if p is not None else None
        )
    if "team" in events_df.columns:
        events_df["team"] = events_df["team"].apply(
            lambda t: t.name if hasattr(t, "name") and t is not None else str(t) if t is not None else None
        )

    return {
        "dataset": dataset,
        "events": events_df,
        "match_id": match_id,
    }


def list_open_competitions() -> pd.DataFrame:
    """
    Return all competitions available in StatsBomb open data.

    Returns:
        DataFrame with columns: competition_id, competition_name,
        season_id, season_name, competition_gender
    """
    from statsbombpy import sb
    return sb.competitions()


def list_open_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Return all matches for a competition/season from StatsBomb open data.

    Args:
        competition_id: StatsBomb competition ID (e.g. 11 = La Liga)
        season_id:      StatsBomb season ID (e.g. 1 = 2005/06)

    Returns:
        DataFrame with match_id, home_team, away_team, match_date, score
    """
    from statsbombpy import sb
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    return matches


def get_match_metadata(match_id: int) -> dict:
    """
    Return basic metadata for a StatsBomb match without loading full events.

    Returns:
        dict with home_team, away_team, score, competition, season, date
    """
    from statsbombpy import sb

    # Find the match in open data competitions
    comps = list_open_competitions()
    for _, row in comps.iterrows():
        try:
            matches = sb.matches(
                competition_id=int(row["competition_id"]),
                season_id=int(row["season_id"]),
            )
            match_row = matches[matches["match_id"] == match_id]
            if not match_row.empty:
                m = match_row.iloc[0]
                return {
                    "match_id": match_id,
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "home_score": m.get("home_score", 0),
                    "away_score": m.get("away_score", 0),
                    "competition": row["competition_name"],
                    "season": row["season_name"],
                    "date": str(m.get("match_date", "")),
                }
        except Exception:
            continue

    return {"match_id": match_id, "home_team": "Unknown", "away_team": "Unknown"}
