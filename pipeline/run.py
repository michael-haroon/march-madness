"""
run.py
------
Main entry point.  Ties together:
  1. Data loading
  2. Feature engineering (de Prado + basketball features)
  3. Synthetic validation (does MDI work on known data?)
  4. Feature importance analysis (MDI, MDA, SFI, CFI)
  5. Pairwise model training + purged CV evaluation
  6. 2026 win probability prediction

Run from project root:
    python pipeline/run.py

Outputs saved to  outputs/
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# Allow running as either `python pipeline/run.py` or `python -m pipeline.run`
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.data_loader        import load_all
from pipeline.feature_engineering import (
    build_features, build_pairwise_frame,
    ALL_FEATURES, TIER1_FEATURES, KAGGLE_FEATURES, TOURNEY_PATH_FEATURES,
    CORE_RANK_FEATURES, DEPRADO_FEATURES, RECORD_FEATURES, SOS_FEATURES,
    TEAM_STATS_PC_FEATURES, MASSEY_FEATURES, AWARD_FEATURES,
)
from pipeline.feature_importance  import (
    run_all_importance, synthetic_validation, build_rf, feat_imp_sfi,
    PurgedYearKFold,
)
from pipeline.model               import (
    build_pairwise, train_pairwise_model,
    pairwise_to_win_prob, build_meta_labels, build_market_meta_labels,
    predict_current_year,
)

OUTPUT_DIR = "outputs"
DATA_DIR   = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Load data")
    print("="*60)
    df = load_all(DATA_DIR,
                  include_kaggle=True,
                  include_team_stats=True,
                  include_market=True)
    if df.empty:
        print("No data loaded — check your data/ directory.")
        return
    df.to_csv(os.path.join(OUTPUT_DIR, "master_frame_raw.csv"), index=False)

    # ── 2. Feature engineering ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Feature engineering")
    print("="*60)
    df = build_features(df, run_pca=True, run_reconcile=True, run_redundancy_audit=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "master_frame_features.csv"), index=False)

    # ── 3. Synthetic validation ───────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Synthetic validation (de Prado framework check)")
    print("="*60)
    synth = synthetic_validation(n_samples=600, n_informative=3,
                                  n_redundant=4, n_noise=3)
    synth["mdi"].to_csv(os.path.join(OUTPUT_DIR, "synthetic_mdi.csv"))
    print(f"Synthetic MDI pass: {synth['mdi_pass']}")

    # ── 3b. SFI on Tier 1 labels (~1,400 tournament team rows) ───────────
    print("\n" + "="*60)
    print("STEP 3b: SFI on Tier 1 tournament data (expanded sample)")
    print("="*60)
    # Use all seeded tournament teams (not just Final Four)
    tier1_df = df[df["seed_num"].notna()].copy()
    print(f"  Tier 1 frame: {len(tier1_df)} rows across {tier1_df['year'].nunique()} years")

    # Tier 1 uses only regular-season features (no tourney path, no team-stats raw)
    tier1_feature_candidates = (
        TIER1_FEATURES +
        [f"ts_pc{i}" for i in range(1, 8)] +   # PCA components
        MASSEY_FEATURES
    )
    avail_tier1 = [
        c for c in tier1_feature_candidates
        if c in tier1_df.columns and tier1_df[c].notna().sum() > len(tier1_df) * 0.2
    ]
    print(f"  Using {len(avail_tier1)} Tier 1 features")

    sfi_tier1_results = None
    if len(tier1_df) >= 50 and len(avail_tier1) >= 3:
        # Use "made_ff" as Tier 1 label (~4 positives per year = 4/68 ≈ 6% positive rate)
        X_t1  = tier1_df[avail_tier1].fillna(tier1_df[avail_tier1].median())
        y_t1  = tier1_df["made_ff"].fillna(0).astype(int)
        yr_t1 = tier1_df["year"]

        if y_t1.sum() >= 10:
            print(f"  Running SFI on {len(X_t1)} rows, {y_t1.sum()} positives (made_ff=1)...")
            sfi_tier1_results = feat_imp_sfi(
                build_rf(n_estimators=300), X_t1, y_t1, yr_t1
            )
            sfi_tier1_results.to_csv(
                os.path.join(OUTPUT_DIR, "sfi_tier1_results.csv")
            )
            # Filter to features with positive SFI (above random baseline)
            baseline = -np.log(2)  # log-loss baseline (50/50 prediction)
            sfi_survivors = sfi_tier1_results[
                sfi_tier1_results["mean"] > baseline
            ].index.tolist()
            print(f"  SFI survivors (above baseline): {len(sfi_survivors)} features")
            pd.Series(sfi_survivors).to_csv(
                os.path.join(OUTPUT_DIR, "sfi_tier1_filtered.csv"), index=False
            )
        else:
            print("  Not enough positives for Tier 1 SFI — skipping")
            sfi_survivors = avail_tier1
    else:
        print("  Insufficient Tier 1 data — skipping SFI")
        sfi_survivors = avail_tier1

    # ── 4. Feature importance on Final Four frame ─────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Feature importance  (Final Four teams only)")
    print("="*60)
    ff = df[df["in_final_four"] == 1].copy()
    print(f"  Final Four frame: {len(ff)} rows across {ff['year'].nunique()} years")

    # Tier 2 features: Tier 1 survivors + tournament path + PCA components
    tier2_candidates = (
        sfi_survivors +
        [c for c in TOURNEY_PATH_FEATURES if c in ff.columns] +
        [f"ts_pc{i}" for i in range(1, 8) if f"ts_pc{i}" in ff.columns]
    )
    # Also include any important-looking feature not captured by SFI filter
    all_tier2 = list(dict.fromkeys(tier2_candidates + ALL_FEATURES))  # dedup, preserve order

    avail_features = [
        c for c in all_tier2
        if c in ff.columns and ff[c].notna().sum() > len(ff) * 0.3  # at least 30% non-null
    ]
    print(f"  Using {len(avail_features)} features: {avail_features}")

    if len(ff) >= 8 and len(avail_features) >= 2:
        X_ff = ff[avail_features].fillna(ff[avail_features].median())
        y_ff = ff["champion_flag"]
        years_ff  = ff["year"]
        weights_ff = ff["sample_weight"] if "sample_weight" in ff.columns else None

        importance_results = run_all_importance(
            X_ff, y_ff, years_ff,
            sample_weight=weights_ff,
            run_sfi=True,
        )

        importance_results["summary"].to_csv(
            os.path.join(OUTPUT_DIR, "feature_importance_summary.csv")
        )
        importance_results["mdi"].to_csv(os.path.join(OUTPUT_DIR, "fi_mdi.csv"))
        importance_results["mda"].to_csv(os.path.join(OUTPUT_DIR, "fi_mda.csv"))
        if importance_results["sfi"] is not None:
            importance_results["sfi"].to_csv(os.path.join(OUTPUT_DIR, "fi_sfi.csv"))
        importance_results["cfi_mda"].to_csv(os.path.join(OUTPUT_DIR, "fi_cfi_mda.csv"))
    else:
        print("  Not enough Final Four data for importance analysis yet.")
        importance_results = {}

    # ── 5. Pairwise model ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Pairwise Bradley-Terry model")
    print("="*60)

    # Fill NaN before pairwise construction so diff features don't get dropped
    rank_features = [c for c in avail_features if c in ff.columns]
    ff_filled = ff.copy()
    for col in rank_features:
        if ff_filled[col].isna().any():
            ff_filled[col] = ff_filled[col].fillna(ff_filled[col].median())

    pair_df = build_pairwise(ff_filled, rank_features)
    print(f"  Pairwise frame: {len(pair_df)} matchup pairs")

    diff_cols = [c for c in pair_df.columns if c.startswith("diff_")]
    pair_clean = pair_df.dropna(subset=diff_cols + ["a_wins"])
    print(f"  Usable pairs (no NaN): {len(pair_clean)}")

    if len(pair_clean) >= 20:
        model_results = train_pairwise_model(pair_df, rank_features)
        ff = ff_filled  # use filled version for downstream steps
        model_results["cv_results"].to_csv(
            os.path.join(OUTPUT_DIR, "pairwise_cv_results.csv"), index=False
        )
    else:
        print("  Not enough pairwise data yet. Skipping model training.")
        model_results = None

    # ── 6. Meta-labels (with market primary model where available) ────────
    print("\n" + "="*60)
    print("STEP 6: Meta-labels")
    print("="*60)
    ff_meta = build_market_meta_labels(ff)
    meta_acc = ff_meta.groupby("year")["meta_label"].max().mean()
    print(f"  Primary model accuracy (market/consensus): {meta_acc:.1%}")
    save_cols = [c for c in ["year", "team", "finish", "consensus_rank",
                              "mkt_vwap", "mkt_implied_prob",
                              "primary_model", "primary_pick", "meta_label"]
                 if c in ff_meta.columns]
    ff_meta[save_cols].to_csv(
        os.path.join(OUTPUT_DIR, "meta_labels.csv"), index=False
    )

    # ── 7. 2026 prediction ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7: 2026 Final Four win probabilities")
    print("="*60)
    ff_2026 = df[(df["year"] == 2026) & (df["in_final_four"] == 1)].copy()

    # NOTE: 2026 Final Four teams from your MD: Illinois, UConn, Michigan, Arizona
    # If they're not yet in the data as final_four=1, you can override:
    FINAL_FOUR_2026 = ["Illinois", "Connecticut", "Michigan", "Arizona"]
    if ff_2026.empty:
        ff_2026 = df[(df["year"] == 2026) &
                     (df["team"].isin(FINAL_FOUR_2026))].copy()
        ff_2026["in_final_four"] = 1
        print(f"  Using overridden 2026 Final Four: {FINAL_FOUR_2026}")

    if not ff_2026.empty:
        ff_2026 = ff_2026.copy()
        # Fill NaN in feature cols so pairwise doesn't drop all pairs
        for col in rank_features:
            if col in ff_2026.columns and ff_2026[col].isna().any():
                ff_2026[col] = ff_2026[col].fillna(ff_2026[col].median())

        if model_results is not None:
            pred_2026 = predict_current_year(
                model_results["model"], ff_2026, rank_features
            )
        else:
            # No trained model — rank by composite of available signals
            # Weighted blend: market implied prob + inverse consensus rank + scoring margin
            ff_2026["_mkt"]  = ff_2026.get("mkt_vwap",       pd.Series(0.25, index=ff_2026.index)).fillna(0.25)
            ff_2026["_rank"] = ff_2026.get("consensus_rank",  pd.Series(10.0, index=ff_2026.index)).fillna(10.0)
            ff_2026["_mar"]  = ff_2026.get("kg_scoring_margin", pd.Series(0.0, index=ff_2026.index)).fillna(0.0)
            # Normalise each signal to [0,1] then blend: 50% market, 30% rank, 20% margin
            def _norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
            ff_2026["win_prob"] = (
                0.50 * _norm(ff_2026["_mkt"]) +
                0.30 * _norm(-ff_2026["_rank"]) +  # lower rank = better
                0.20 * _norm(ff_2026["_mar"])
            )
            total = ff_2026["win_prob"].sum()
            ff_2026["win_prob"] = ff_2026["win_prob"] / total if total > 0 else 0.25
            pred_2026 = predict_current_year(None, ff_2026, rank_features)

        print("\n  2026 Win Probabilities:")
        print(pred_2026.to_string())
        pred_2026.to_csv(os.path.join(OUTPUT_DIR, "2026_predictions.csv"))

    # ── 8. Feature discovery catalog ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8: Feature discovery catalog")
    print("="*60)

    def _feature_source(col):
        if col.startswith("kg_"):        return "kaggle"
        if col.startswith("ts_"):        return "team_stats"
        if col.startswith("massey_"):    return "massey"
        if col.startswith("mkt_"):       return "market"
        if col.endswith("_yoy"):         return "deprado_fracdiff"
        if col in ("win_entropy", "cusum_peak"): return "deprado_entropy"
        if col in CORE_RANK_FEATURES:    return "team_sheet_rank"
        if col in SOS_FEATURES:          return "team_sheet_sos"
        if col in RECORD_FEATURES:       return "team_sheet_record"
        if col in AWARD_FEATURES:        return "yearlys"
        return "engineered"

    catalog_rows = []
    for col in avail_features:
        row = {
            "feature":      col,
            "source":       _feature_source(col),
            "coverage_ff":  f"{ff[col].notna().sum()}/{len(ff)}",
            "coverage_t1":  f"{tier1_df[col].notna().sum()}/{len(tier1_df)}" if col in tier1_df.columns else "N/A",
        }
        # Attach importance ranks if available
        if sfi_tier1_results is not None and col in sfi_tier1_results.index:
            row["sfi_tier1_mean"] = round(sfi_tier1_results.loc[col, "mean"], 4)
        if importance_results and col in importance_results.get("summary", pd.DataFrame()).index:
            summ = importance_results["summary"]
            row["mdi_rank"]   = summ.loc[col, "rank_MDI"]  if "rank_MDI"  in summ.columns else np.nan
            row["mda_rank"]   = summ.loc[col, "rank_MDA"]  if "rank_MDA"  in summ.columns else np.nan
            row["avg_rank"]   = summ.loc[col, "avg_rank"]  if "avg_rank"  in summ.columns else np.nan
        catalog_rows.append(row)

    catalog = pd.DataFrame(catalog_rows)
    if "avg_rank" in catalog.columns:
        catalog = catalog.sort_values("avg_rank")
    catalog.to_csv(os.path.join(OUTPUT_DIR, "feature_discovery_report.csv"), index=False)
    print(f"  Saved feature catalog: {len(catalog)} features")
    if not catalog.empty:
        print(catalog[["feature", "source", "coverage_ff"]].head(15).to_string(index=False))

    print("\n" + "="*60)
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
