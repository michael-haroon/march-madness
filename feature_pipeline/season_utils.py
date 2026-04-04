"""
season_utils.py
---------------
Shared helpers for mapping a calendar date to a Kaggle Season + DayNum.

Usage:
    from feature_pipeline.season_utils import build_season_table, get_season_and_daynum

    table = build_season_table("data/kaggle/MSeasons.csv")
    season, daynum = get_season_and_daynum(date(2022, 1, 15), table)
"""

from datetime import datetime, date

import pandas as pd


def build_season_table(mseasons_path: str) -> list[tuple[int, date]]:
    """
    Return list of (season, dayzero_date) sorted by dayzero descending.

    Args:
        mseasons_path: path to MSeasons.csv

    Returns:
        List of (Season, DayZero as date) sorted newest-first.
    """
    df = pd.read_csv(mseasons_path)
    rows = []
    for _, r in df.iterrows():
        dz = datetime.strptime(r["DayZero"], "%m/%d/%Y").date()
        rows.append((int(r["Season"]), dz))
    return sorted(rows, key=lambda x: x[1], reverse=True)


def get_season_and_daynum(
    file_date: date,
    season_table: list[tuple[int, date]],
    max_daynum: int = 210,
) -> tuple[int, int] | tuple[None, None]:
    """
    Given a calendar date, find the Kaggle Season whose DayZero gives
    0 <= DayNum <= max_daynum.

    Args:
        file_date:    the calendar date to convert
        season_table: output of build_season_table()
        max_daynum:   upper bound for valid DayNum (default 210 covers full season)

    Returns:
        (Season, DayNum) or (None, None) if no season matches.
    """
    for season, dayzero in season_table:
        dn = (file_date - dayzero).days
        if 0 <= dn <= max_daynum:
            return season, dn
    return None, None
