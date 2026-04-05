"""
run_v2.py
---------
Feature analysis pipeline for NCAA tournament prediction.

Orchestrates: team features → enrichment → game pairs → de Prado feature analysis
(PCA → ONC clustering → MDI/MDA/SFI)

Predictive modeling and trading strategy live in /strategy (separate module).

Usage:
    python -m feature_pipeline.run_v2 [--data-dir data/] [--output-dir output_v2/]
    python -m feature_pipeline.run_v2 --skip-enrich   # Kaggle-only features
    python -m feature_pipeline.run_v2 --skip-path     # No tournament path features
"""

import argparse
import warnings
from pathlib import Path

from feature_pipeline.game_model import (
    build_team_season_features,
    enrich_with_existing_features,
    build_game_pairs,
)
from feature_pipeline.feature_importance import run_all_importance, synthetic_validation


def main(data_dir: str = "data", output_dir: str = "output_v2",
         skip_enrich: bool = False, skip_path: bool = False):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    kaggle_dir = f"{data_dir}/kaggle"

    # ── Step 1: Build team-season features from Kaggle game data ──────────────
    print("Step 1: Building team-season features from game data...")
    team_df = build_team_season_features(data_dir, min_season=2003)
    team_df.to_csv(output / "team_season_features.csv", index=False)
    print(f"  {len(team_df)} team-seasons built, {team_df.shape[1]} columns")

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
                include_market=False,
                verbose=False,
            )
            # run_pca=True: compress ts_* features into principal components
            existing_df = build_features(
                existing_df, run_pca=True, run_reconcile=False, verbose=False
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

    # ── Step 4: de Prado feature analysis (PCA already applied above) ─────────
    # Flow: ONC clustering (internal) → MDI → MDA → SFI
    # Target = pairwise game win. Folds = seasons (non-overlapping, no purge needed).
    print("\nStep 4: de Prado feature importance analysis...")

    # Synthetic validation — confirms MDI can recover a known signal before real run
    synth = synthetic_validation()
    print(f"  Synthetic MDI validation: {'PASS' if synth['mdi_pass'] else 'FAIL'}")

    # Build X, y, years from game pairs
    # Use all numeric diff columns (team_a - team_b feature differences) as X
    skip_cols = {"Season", "team_a_id", "team_b_id", "team_a_wins",
                 "team_a_seed", "team_b_seed", "DayNum", "Round"}
    feat_cols = [c for c in pairs_df.columns
                 if c not in skip_cols and pairs_df[c].dtype.kind in "fi"]
    X = pairs_df[feat_cols].copy()
    y = pairs_df["team_a_wins"].astype(int)
    years = pairs_df["Season"]

    print(f"  Running MDI/MDA/SFI on {len(X)} game pairs, {len(feat_cols)} features...")

    # run_all_importance handles ONC clustering internally for clustered MDI/MDA
    importance_results = run_all_importance(X, y, years, run_sfi=True)

    # Save summary (includes p-values)
    importance_results["summary"].to_csv(output / "feature_importance_catalog.csv")

    # Save per-method summaries
    importance_results["mdi"].to_csv(output / "importance_mdi.csv")
    importance_results["mda"].to_csv(output / "importance_mda.csv")
    if importance_results.get("sfi") is not None:
        importance_results["sfi"].to_csv(output / "importance_sfi.csv")

    # Save raw distributions
    importance_results["mdi_raw"].to_csv(output / "importance_mdi_raw.csv", index=True)
    importance_results["mda_raw"].to_csv(output / "importance_mda_raw.csv", index=True)
    if importance_results.get("sfi_raw") is not None:
        importance_results["sfi_raw"].to_csv(output / "importance_sfi_raw.csv", index=True)

    # Save filter report (de Prado criteria: tiers, pass/fail, composite rank)
    filtered_dir = output / "filtered"
    filtered_dir.mkdir(exist_ok=True)
    importance_results["filter_report"].to_csv(filtered_dir / "feature_report.csv")
    survivors = importance_results["survivors"]
    with open(filtered_dir / "feature_list.txt", "w") as f:
        f.write("\n".join(survivors))
    print(f"\n  {len(survivors)} surviving features → {filtered_dir}/feature_list.txt")

    # Save PCA cross-check results
    importance_results["pca_info"].to_csv(output / "pca_cross_check.csv")
    import json
    with open(output / "kendall_tau.json", "w") as f:
        json.dump(importance_results["tau_results"], f, indent=2)

    print("\nTop 15 features by SFI mean:")
    sfi = importance_results.get("sfi")
    if sfi is not None:
        top = sfi.nlargest(15, "mean")
        for feat, row in top.iterrows():
            print(f"  {feat:<42} SFI={row['mean']:.4f}")
    else:
        print("  SFI not available")

    print(f"\nOutputs written to {output}/")
    print("  feature_importance_catalog.csv     ← summary: MDI/MDA/SFI + p-values")
    print("  importance_mdi_raw/mda_raw/sfi_raw ← per-tree/fold distributions")
    print("  filtered/feature_report.csv        ← de Prado filter: tiers + pass/fail")
    print("  filtered/feature_list.txt          ← STRONG + MODERATE survivors")
    print("  pca_cross_check.csv                ← PCA ranks + weighted loadings")
    print("  kendall_tau.json                   ← weighted Kendall's tau vs MDI/MDA/SFI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCAA tournament feature analysis pipeline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output_v2/features")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip team-sheet enrichment (Kaggle-only features)")
    parser.add_argument("--skip-path", action="store_true",
                        help="Skip tournament path features")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.skip_enrich, args.skip_path)
