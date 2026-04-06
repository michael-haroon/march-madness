"""
run.py
------
Entry point for the strategy module.

Usage:
    python -m strategy.run [--market-weight 0.3] [--model lgbm|logreg|both]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.config import (
    SURVIVING_FEATURES, BRACKET_2026, DEFAULT_MARKET_WEIGHT,
    DATA_DIR, OUTPUT_DIR,
)
from strategy.data import (
    load_game_pairs, load_team_features, resolve_bracket_teams,
    load_path_features, load_market_data,
)
from strategy.model import train_and_evaluate, calibrate, compare_models
from strategy.bracket import compute_pairwise_probs, compute_championship_probs
from strategy.market import blend, compute_edges, trade_recommendations


def main(market_weight: float = DEFAULT_MARKET_WEIGHT,
         model_type: str = "both"):

    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("Step 1: Loading data...")
    pairs_df = load_game_pairs()
    team_df = load_team_features()

    avail = [f for f in SURVIVING_FEATURES if f in pairs_df.columns]
    missing = [f for f in SURVIVING_FEATURES if f not in pairs_df.columns]
    if missing:
        print(f"  Warning: {len(missing)} features not in game_pairs: {missing}")
    print(f"  {len(pairs_df)} games, {len(avail)} surviving features")

    # ── Step 2: Train models ─────────────────────────────────────────────
    print("\nStep 2: Training models on surviving features...")

    results = {}
    if model_type in ("lgbm", "both"):
        print("  Training LightGBM...")
        results["lgbm"] = train_and_evaluate(pairs_df, avail, "lgbm")
    if model_type in ("logreg", "both"):
        print("  Training LogReg...")
        results["logreg"] = train_and_evaluate(pairs_df, avail, "logreg")

    best_name = compare_models(results)
    best = results[best_name]

    # Save CV results
    for name, res in results.items():
        res["cv_results"].to_csv(output / f"cv_{name}.csv", index=False)

    # Calibrate best model
    X_all = pairs_df[avail]
    y_all = pairs_df["team_a_wins"].astype(int)
    cal_model = calibrate(best["model"], X_all, y_all)

    # ── Step 3: Resolve 2026 bracket ─────────────────────────────────────
    print("\nStep 3: Resolving 2026 bracket...")
    name_to_id = resolve_bracket_teams(BRACKET_2026)
    id_to_name = {v: k for k, v in name_to_id.items()}

    s1a_name, s1b_name = BRACKET_2026["semi1"]
    s2a_name, s2b_name = BRACKET_2026["semi2"]
    s1a, s1b = name_to_id[s1a_name], name_to_id[s1b_name]
    s2a, s2b = name_to_id[s2a_name], name_to_id[s2b_name]
    team_ids = [s1a, s1b, s2a, s2b]

    print(f"  Semi 1: {s1a_name} ({s1a}) vs {s1b_name} ({s1b})")
    print(f"  Semi 2: {s2a_name} ({s2a}) vs {s2b_name} ({s2b})")

    # ── Step 4: Load path features + compute pairwise probs ──────────────
    print("\nStep 4: Computing pairwise probabilities...")

    # Try loading actual path features (through E8)
    path_features = None
    try:
        path_features = load_path_features(team_ids, 2026)
        has_path = any(
            pf.get("path_games_played", 0) > 0
            for pf in path_features.values()
        )
        if has_path:
            print("  Using actual E8 path features")
        else:
            print("  No path data available (2026 tournament results not in Kaggle)")
            path_features = None
    except Exception as e:
        print(f"  Path features unavailable: {e}")

    pairwise = compute_pairwise_probs(
        cal_model, team_df, team_ids, 2026, path_features
    )

    print("\n  Pairwise probabilities:")
    for (ta, tb), p in sorted(pairwise.items()):
        if ta < tb:
            na = id_to_name.get(ta, str(ta))
            nb = id_to_name.get(tb, str(tb))
            print(f"    {na} vs {nb}: {na} {p:.1%} | {nb} {1-p:.1%}")

    # ── Step 5: Closed-form bracket probabilities ────────────────────────
    print("\nStep 5: Championship probabilities (closed-form)...")
    bracket_df = compute_championship_probs(team_ids, pairwise)
    bracket_df["team_name"] = bracket_df["TeamID"].map(id_to_name)

    # Get seeds
    season_2026 = team_df[team_df["Season"] == 2026].set_index("TeamID")
    bracket_df["seed"] = bracket_df["TeamID"].apply(
        lambda t: int(season_2026.loc[t, "seed_num"])
        if t in season_2026.index else "?"
    )

    print("\n  " + "-" * 60)
    print(f"  {'Team':<18} {'Seed':>4} {'Win Semi':>10} {'Champion':>10}")
    print("  " + "-" * 60)
    for _, row in bracket_df.sort_values("p_champion", ascending=False).iterrows():
        print(f"  {row['team_name']:<18} {row['seed']:>4} "
              f"{row['p_win_semi']:>10.1%} {row['p_champion']:>10.1%}")
    print("  " + "-" * 60)

    # ── Step 6: Market blend ─────────────────────────────────────────────
    if market_weight>0:
        print(f"\nStep 6: Blending with Kalshi market (weight={market_weight})...")

        # Build model probs dict keyed by team name
        model_probs = {}
        for _, row in bracket_df.iterrows():
            model_probs[row["team_name"]] = row["p_champion"]

        # Load market data
        mkt_all = load_market_data(DATA_DIR, 2026)
        # Filter to just our 4 teams, normalize
        ff_names = set(name_to_id.keys())

        # Handle UConn -> Connecticut mapping
        from feature_pipeline.config import TEAM_NAME_MAP
        reverse_map = {v: k for k, v in TEAM_NAME_MAP.items()}

        market_probs = {}
        for name in ff_names:
            # Try exact name and common aliases
            candidates = [name] + [k for k, v in TEAM_NAME_MAP.items() if v == name]
            for c in candidates:
                if c in mkt_all:
                    market_probs[name] = mkt_all[c]
                    break

        # Normalize market probs to sum to 1 across FF teams
        mkt_total = sum(market_probs.values()) if market_probs else 0
        if mkt_total > 0:
            market_probs = {t: p / mkt_total for t, p in market_probs.items()}

        if market_probs:
            blended = blend(model_probs, market_probs, market_weight)
            edges = compute_edges(model_probs, market_probs)
            recs = trade_recommendations(model_probs, market_probs)

            print("\n  " + "-" * 75)
            print(f"  {'Team':<18} {'Model':>8} {'Market':>8} {'Blended':>8} {'Edge':>8}")
            print("  " + "-" * 75)
            for team in sorted(blended, key=lambda t: -blended[t]):
                mp = model_probs.get(team, 0)
                mk = market_probs.get(team, 0)
                bl = blended[team]
                ed = edges.get(team, 0)
                print(f"  {team:<18} {mp:>8.1%} {mk:>8.1%} {bl:>8.1%} {ed:>+8.1%}")
            print("  " + "-" * 75)

        print("\n  Trade recommendations (half-Kelly, $1000 bankroll):")
        for _, row in recs.iterrows():
            if row["direction"] != "HOLD":
                print(f"    {row['team']}: {row['direction']} "
                      f"${row['bet_amount']:.0f} ({row['kelly_pct']:.1f}% Kelly) "
                      f"EV=${row['expected_value']:.2f}")

        recs.to_csv(output / "trade_recommendations.csv", index=False)
    else:
        print(f"  Market weight is {market_weight} — skipping blend")
        print("Change it in strategy/config.py")

    # ── Save outputs ─────────────────────────────────────────────────────
    bracket_df.to_csv(output / "predictions_2026.csv", index=False)
    print(f"\nOutputs saved to {output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="March Madness strategy")
    parser.add_argument("--market-weight", type=float, default=DEFAULT_MARKET_WEIGHT)
    parser.add_argument("--model", default="both", choices=["lgbm", "logreg", "both"])
    args = parser.parse_args()
    main(market_weight=args.market_weight, model_type=args.model)
