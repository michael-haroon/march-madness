"""
run_v2.py
---------
Pairwise game-level pipeline for NCAA tournament prediction (Phase 1A + 1B).

Orchestrates: team features → enrichment → game pairs (with path) → model → predictions

Usage:
    python -m feature_pipeline.run_v2 [--data-dir data/] [--output-dir output_v2/]
    python -m feature_pipeline.run_v2 --skip-enrich   # Kaggle-only features
    python -m feature_pipeline.run_v2 --skip-path     # Phase 1A only (no path features)
"""

import argparse
import warnings
from pathlib import Path

from feature_pipeline.game_model import (
    build_team_season_features,
    enrich_with_existing_features,
    build_game_pairs,
    train_game_model,
    predict_final_four,
    blend_with_market,
    load_actual_path_features,
)
from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id
from feature_pipeline.config import BRACKET_2026, TEAM_NAME_MAP


def main(data_dir: str = "data", output_dir: str = "output_v2",
         skip_enrich: bool = False, skip_path: bool = False,
         n_sims: int = 50000):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    kaggle_dir = f"{data_dir}/kaggle"

    # ── Step 1: Build team-season features from Kaggle game data ──────────────
    print("Step 1: Building team-season features from game data...")
    team_df = build_team_season_features(data_dir, min_season=2003)
    team_df.to_csv(output / "team_season_features.csv", index=False)
    print(f"  {len(team_df)} team-seasons built, {team_df.shape[1]} columns")

    # ── Market features — loaded once, reused in Step 6 ───────────────────────
    mkt_features = None
    print("Loading market features (used for blend in Step 6)...")
    try:
        from feature_pipeline.market_features import load_kalshi_trades, compute_market_features
        _trades = load_kalshi_trades(data_dir)
        mkt_features = compute_market_features(_trades)
        print(f"  Market features loaded: {len(mkt_features)} (year, team) pairs")
    except Exception as e:
        print(f"  Market load failed: {e}")

    # ── Step 2: Enrich with team sheets, awards, rankings ─────────────────────
    if not skip_enrich:
        print("Step 2: Enriching with team sheets, awards, rankings...")
        try:
            from feature_pipeline.data_loader import load_all
            from feature_pipeline.feature_engineering import build_features

            existing_df = load_all(
                data_dir,
                include_kaggle=False,   # Kaggle game stats already in team_df from Step 1
                include_team_stats=False,
                include_market=False,   # Market handled separately above
                verbose=False,
            )
            existing_df = build_features(
                existing_df, run_pca=False, run_reconcile=False, verbose=False
            )
            team_df = enrich_with_existing_features(team_df, existing_df)
            print(f"  {team_df.shape[1]} features per team-season after enrichment")
        except Exception as e:
            import traceback
            warnings.warn(f"Enrichment failed: {e}\n{traceback.format_exc()}")
    else:
        print("Step 2: Skipped (--skip-enrich)")

    # ── Step 3: Build tournament game pairs ───────────────────────────────────
    print("Step 3: Building pairwise tournament game features...")
    tourney_compact_path = f"{kaggle_dir}/MNCAATourneyCompactResults.csv"
    tourney_detailed_path = f"{kaggle_dir}/MNCAATourneyDetailedResults.csv"

    pairs_df = build_game_pairs(
        team_df,
        tourney_compact_path,
        min_season=2003,
        max_season=2025,
        include_path=not skip_path,
        tourney_detailed_path=tourney_detailed_path,
    )
    pairs_df.to_csv(output / "game_pairs.csv", index=False)
    label_balance = pairs_df["team_a_wins"].mean()
    path_cols = [c for c in pairs_df.columns if "path_" in c]
    print(f"  {len(pairs_df)} tournament games (label balance: {label_balance:.3f}), "
          f"{len(path_cols)} path diff columns")

    # ── Step 4: Train pairwise model ──────────────────────────────────────────
    print("Step 4: Training pairwise game model...")
    results = train_game_model(pairs_df)
    results["cv_results"].to_csv(output / "cv_results.csv", index=False)
    results["oof_preds"].to_csv(output / "oof_predictions.csv", index=False)
    results["feature_importance"].to_csv(output / "feature_importance.csv", index=False)

    mean_auc = results["cv_results"]["auc"].mean()
    mean_ll = results["cv_results"]["log_loss"].mean()
    mean_acc = results["cv_results"]["accuracy"].mean()
    print(f"  CV AUC: {mean_auc:.4f}  Log-Loss: {mean_ll:.4f}  Accuracy: {mean_acc:.4f}")

    print("\nTop 10 features by importance:")
    for _, row in results["feature_importance"].head(10).iterrows():
        print(f"  {row['feature']:<42} {row['importance']:.1f}")

    # ── Step 5: Predict 2026 Final Four ───────────────────────────────────────
    print("\nStep 5: Predicting 2026 Final Four...")
    lookup = build_id_lookup(kaggle_dir)

    semi1_names = BRACKET_2026["semi1"]
    semi2_names = BRACKET_2026["semi2"]
    all_ff_names = list(semi1_names) + list(semi2_names)

    ff_ids = []
    for name in all_ff_names:
        tid = resolve_team_id(name, lookup, TEAM_NAME_MAP)
        if tid is None:
            print(f"  WARNING: Could not resolve '{name}' to TeamID")
        ff_ids.append(tid)

    if any(tid is None for tid in ff_ids):
        print("  WARNING: Some teams could not be resolved.")
        ff_ids = [tid if tid is not None else -(i + 1) for i, tid in enumerate(ff_ids)]

    team_df_2026 = team_df[team_df["Season"] == 2026]
    if len(team_df_2026) == 0:
        print("  WARNING: No 2026 season data. Using 2025 as proxy.")
        team_df_2026 = team_df[team_df["Season"] == 2025].copy()
        team_df_2026["Season"] = 2026

    # Load actual E8 path features for 2026 FF teams (real tournament results)
    actual_path_feats = None
    if not skip_path:
        try:
            actual_path_feats = load_actual_path_features(data_dir, season=2026,
                                                          team_ids=[t for t in ff_ids if t])
            print(f"  Loaded actual path features for "
                  f"{sum(1 for v in actual_path_feats.values() if v['path_games_played'] > 0)} "
                  f"teams (games through E8)")
        except Exception as e:
            print(f"  Path feature load failed: {e}; proceeding without path features")
            actual_path_feats = None

    preds = predict_final_four(
        results["model"],
        team_df_2026,
        ff_ids,
        results["feature_cols"],
        n_sims=n_sims,
        random_seed=42,
        include_path=(not skip_path and actual_path_feats is not None),
        actual_path_features=actual_path_feats,
    )

    # ── Step 6: Blend with market (use cached features — no re-load) ──────────
    print("Step 6: Blending with Kalshi market...")
    if mkt_features is not None:
        mkt_2026 = mkt_features[mkt_features["year"] == 2026].copy()
        if len(mkt_2026) == 0:
            mkt_2026 = mkt_features[mkt_features["year"] == mkt_features["year"].max()].copy()
            mkt_2026["year"] = 2026
        preds = blend_with_market(preds, mkt_2026, market_weight=0.3)
        print("  Market blend successful")
    else:
        print("  Market data unavailable; using model probs directly")
        preds["blended_prob"] = preds["p_champion"]
        preds["mkt_implied"] = float("nan")
        preds["model_edge"] = float("nan")

    preds.to_csv(output / "2026_predictions.csv", index=False)

    print("\n" + "=" * 57)
    print("  2026 Championship Predictions")
    print("=" * 57)
    display_cols = ["team_name", "seed", "p_win_semi", "p_champion", "blended_prob", "model_edge"]
    display_cols = [c for c in display_cols if c in preds.columns]
    print(preds[display_cols].to_string(index=False))
    print("=" * 57)
    print(f"\nOutputs written to {output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise game-level NCAA tournament model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output_v2")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip team-sheet enrichment (Kaggle-only features)")
    parser.add_argument("--skip-path", action="store_true",
                        help="Skip tournament path features (Phase 1A only)")
    parser.add_argument("--n-sims", type=int, default=50000)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.skip_enrich, args.skip_path, args.n_sims)
