"""
data.py
-------
Load pre-computed data, resolve team names, build prediction feature vectors.
"""

import warnings
import numpy as np
import pandas as pd

from strategy.config import (
    SURVIVING_FEATURES, DATA_DIR, GAME_PAIRS_PATH, TEAM_FEATURES_PATH,
)


def load_game_pairs(path: str = GAME_PAIRS_PATH) -> pd.DataFrame:
    """Load pre-built game pairs, subset to surviving features + metadata."""
    df = pd.read_csv(path)
    meta_cols = ["Season", "TeamA", "TeamB", "team_a_wins", "round_num", "DayNum"]
    # Deduplicate: TeamB may appear in both meta_cols and SURVIVING_FEATURES
    all_cols = list(dict.fromkeys(
        [c for c in meta_cols if c in df.columns] +
        [c for c in SURVIVING_FEATURES if c in df.columns]
    ))
    df = df[all_cols].copy()

    # Median-impute NaN in feature columns (BPI, NET-based features have era gaps)
    for col in SURVIVING_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def load_team_features(path: str = TEAM_FEATURES_PATH) -> pd.DataFrame:
    """Load per-team season features for prediction."""
    return pd.read_csv(path)


def resolve_bracket_teams(bracket: dict, data_dir: str = DATA_DIR) -> dict:
    """
    Resolve bracket team names to Kaggle TeamIDs.
    Returns {team_name: TeamID} for all 4 teams.
    """
    from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id
    from feature_pipeline.config import TEAM_NAME_MAP

    kaggle_dir = f"{data_dir}/kaggle"
    lookup = build_id_lookup(kaggle_dir)

    result = {}
    for matchup in bracket.values():
        for name in matchup:
            tid = resolve_team_id(name, lookup, TEAM_NAME_MAP)
            if tid is None:
                warnings.warn(f"Could not resolve team: {name}")
            result[name] = int(tid) if tid is not None else None
    return result


def load_path_features(team_ids: list, season: int,
                       data_dir: str = DATA_DIR) -> dict:
    """Load actual tournament path features for Final Four teams (through E8)."""
    from feature_pipeline.game_model import load_actual_path_features
    return load_actual_path_features(data_dir, season, team_ids)


def build_matchup_features(team_df: pd.DataFrame,
                           tid_a: int, tid_b: int,
                           season: int,
                           path_features: dict = None) -> np.ndarray:
    """
    Build a single-row diff feature vector for team_a vs team_b.
    Canonical ordering: tid_a < tid_b (caller must handle flip).

    Returns array of shape (1, n_features) in SURVIVING_FEATURES order.
    """
    season_df = team_df[team_df["Season"] == season].set_index("TeamID")
    if tid_a not in season_df.index or tid_b not in season_df.index:
        return np.full((1, len(SURVIVING_FEATURES)), np.nan)

    fa = season_df.loc[tid_a]
    fb = season_df.loc[tid_b]

    diffs = []
    for feat in SURVIVING_FEATURES:
        # TeamB is the higher TeamID in the canonical ordering — not a diff feature
        if feat == "TeamB":
            diffs.append(float(max(tid_a, tid_b)))
            continue

        base = feat.replace("diff_", "", 1)

        # Check path feature override
        if path_features and base.startswith("path_"):
            va = path_features.get(tid_a, {}).get(base, np.nan)
            vb = path_features.get(tid_b, {}).get(base, np.nan)
        else:
            va = float(fa[base]) if base in fa.index else np.nan
            vb = float(fb[base]) if base in fb.index else np.nan

        try:
            diffs.append(float(va) - float(vb))
        except (TypeError, ValueError):
            diffs.append(np.nan)

    return np.array(diffs, dtype=float).reshape(1, -1)


def load_market_data(data_dir: str = DATA_DIR, year: int = 2026) -> dict:
    """
    Load Kalshi market implied probabilities for Final Four teams.
    Returns {team_name: implied_prob} normalized to sum to 1.
    """
    try:
        from feature_pipeline.market_features import (
            load_kalshi_trades, compute_market_features,
        )
        trades = load_kalshi_trades(data_dir)
        mkt = compute_market_features(trades)
        mkt_year = mkt[mkt["year"] == year].copy()

        if mkt_year.empty:
            warnings.warn(f"No market data for {year}")
            return {}

        # Return VWAP dict, normalized
        team_vwap = dict(zip(mkt_year["team"], mkt_year["mkt_vwap"]))
        total = sum(team_vwap.values())
        if total > 0:
            return {t: v / total for t, v in team_vwap.items()}
        return team_vwap
    except Exception as e:
        warnings.warn(f"Market data loading failed: {e}")
        return {}
