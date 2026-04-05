"""
market.py
---------
Market integration: blend model with Kalshi, compute edges, Kelly sizing.
"""

import numpy as np
import pandas as pd


def blend(model_probs: dict, market_probs: dict,
          weight: float = 0.30) -> dict:
    """
    Linear blend of model and market probabilities.
    model_probs: {team_name: prob}
    market_probs: {team_name: prob}
    weight: market weight (0 = pure model, 1 = pure market)

    Returns {team_name: blended_prob} normalized to sum to 1.
    """
    blended = {}
    for team in model_probs:
        mp = model_probs[team]
        mkt = market_probs.get(team, mp)  # fallback to model if no market
        blended[team] = (1 - weight) * mp + weight * mkt

    # Normalize
    total = sum(blended.values())
    if total > 0:
        blended = {t: p / total for t, p in blended.items()}

    return blended


def compute_edges(model_probs: dict, market_probs: dict) -> dict:
    """Per-team edge: model - market. Positive = model says undervalued."""
    return {t: model_probs[t] - market_probs.get(t, model_probs[t])
            for t in model_probs}


def kelly_fraction(model_prob: float, market_prob: float,
                   fraction: float = 0.5) -> float:
    """
    Half-Kelly fraction for a binary bet.
    model_prob: our estimated true probability of winning
    market_prob: implied probability from market price
    fraction: Kelly fraction (0.5 = half-Kelly, conservative)

    Returns fraction of bankroll to bet (0 if no edge, capped at 0.25).
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0

    # Odds: if market says 30%, buying YES at $0.30 pays $1.00
    # b = (1 / market_prob) - 1 = net odds
    b = (1.0 / market_prob) - 1.0
    q = 1.0 - model_prob

    # Kelly: f = (p*b - q) / b
    f = (model_prob * b - q) / b

    # Apply fractional Kelly and cap
    f = max(0.0, f * fraction)
    return min(f, 0.25)


def trade_recommendations(model_probs: dict, market_probs: dict,
                          bankroll: float = 1000.0,
                          kelly_frac: float = 0.5) -> pd.DataFrame:
    """
    Generate trade recommendations for each team.

    Returns DataFrame with: team, model_prob, market_prob, edge,
        direction, kelly_pct, bet_amount, expected_value
    """
    rows = []
    for team in sorted(model_probs.keys()):
        mp = model_probs[team]
        mkt = market_probs.get(team, mp)
        edge = mp - mkt

        if edge > 0:
            direction = "BUY YES"
            kf = kelly_fraction(mp, mkt, kelly_frac)
        elif edge < -0.02:  # only sell if meaningful negative edge
            direction = "BUY NO"
            # For NO bet: our prob of NO winning = 1-mp, market NO price = 1-mkt
            kf = kelly_fraction(1 - mp, 1 - mkt, kelly_frac)
        else:
            direction = "HOLD"
            kf = 0.0

        bet = bankroll * kf
        # EV = bet * (payout * true_prob - cost)
        if direction == "BUY YES" and mkt > 0:
            ev = bet * ((1.0 / mkt - 1) * mp - (1 - mp))
        elif direction == "BUY NO" and mkt < 1:
            ev = bet * ((1.0 / (1 - mkt) - 1) * (1 - mp) - mp)
        else:
            ev = 0.0

        rows.append({
            "team": team,
            "model_prob": round(mp, 4),
            "market_prob": round(mkt, 4),
            "edge": round(edge, 4),
            "direction": direction,
            "kelly_pct": round(kf * 100, 1),
            "bet_amount": round(bet, 2),
            "expected_value": round(ev, 2),
        })

    return pd.DataFrame(rows)
