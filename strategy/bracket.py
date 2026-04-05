"""
bracket.py
----------
Closed-form 4-team single-elimination bracket probabilities.

For Final Four (A vs B, C vs D, winners meet in final):
  P(A champ) = P(A>B) * [P(C>D) * P(A>C) + P(D>C) * P(A>D)]

No Monte Carlo — exact, deterministic, zero noise.
"""

import numpy as np
import pandas as pd

from strategy.config import SURVIVING_FEATURES
from strategy.data import build_matchup_features


def compute_pairwise_probs(model, team_df: pd.DataFrame,
                           team_ids: list, season: int,
                           path_features: dict = None) -> dict:
    """
    Compute P(A beats B) for all ordered pairs among team_ids.
    Uses canonical ordering (min ID first) and flips as needed.

    Returns {(tid_a, tid_b): P(tid_a beats tid_b)} for all 12 ordered pairs.
    """
    probs = {}
    for i, ta in enumerate(team_ids):
        for j, tb in enumerate(team_ids):
            if ta == tb:
                continue

            # Canonical: lower ID first
            lo, hi = min(ta, tb), max(ta, tb)
            if (lo, hi) not in probs:
                X = build_matchup_features(team_df, lo, hi, season, path_features)
                X_df = pd.DataFrame(X, columns=SURVIVING_FEATURES)
                # Impute any NaN with 0 (neutral diff)
                X_df = X_df.fillna(0)
                p_lo_wins = float(model.predict_proba(X_df)[:, 1][0])
                probs[(lo, hi)] = p_lo_wins

            # Return in requested order
            if ta == lo:
                probs[(ta, tb)] = probs[(lo, hi)]
            else:
                probs[(ta, tb)] = 1.0 - probs[(lo, hi)]

    return probs


def compute_championship_probs(team_ids: list,
                               pairwise: dict) -> pd.DataFrame:
    """
    Closed-form bracket probabilities for 4-team single elimination.

    team_ids: [s1a, s1b, s2a, s2b]
        Semi 1: team_ids[0] vs team_ids[1]
        Semi 2: team_ids[2] vs team_ids[3]

    pairwise: {(tid_a, tid_b): P(a beats b)} from compute_pairwise_probs

    Returns DataFrame with columns: TeamID, p_win_semi, p_reach_final, p_champion
    """
    A, B, C, D = team_ids

    p_ab = pairwise[(A, B)]  # P(A beats B)
    p_cd = pairwise[(C, D)]  # P(C beats D)

    # All 4 possible finals matchups
    p_ac = pairwise[(A, C)]
    p_ad = pairwise[(A, D)]
    p_bc = pairwise[(B, C)]
    p_bd = pairwise[(B, D)]

    # Championship probabilities (law of total probability)
    champ_A = p_ab * (p_cd * p_ac + (1 - p_cd) * p_ad)
    champ_B = (1 - p_ab) * (p_cd * p_bc + (1 - p_cd) * p_bd)
    champ_C = p_cd * (p_ab * (1 - p_ac) + (1 - p_ab) * (1 - p_bc))
    champ_D = (1 - p_cd) * (p_ab * (1 - p_ad) + (1 - p_ab) * (1 - p_bd))

    # Semi win probabilities
    p_semi = {A: p_ab, B: 1 - p_ab, C: p_cd, D: 1 - p_cd}

    # Reach final = win semi (same thing for 4-team bracket)
    p_final = p_semi.copy()

    records = []
    for tid, p_champ in [(A, champ_A), (B, champ_B), (C, champ_C), (D, champ_D)]:
        records.append({
            "TeamID": tid,
            "p_win_semi": p_semi[tid],
            "p_reach_final": p_final[tid],
            "p_champion": p_champ,
        })

    result = pd.DataFrame(records)

    # Sanity check: probabilities should sum to 1
    total = result["p_champion"].sum()
    if abs(total - 1.0) > 0.001:
        print(f"  Warning: championship probs sum to {total:.4f}, expected 1.0")

    return result
