"""
model.py
--------
Two modelling approaches for the Final Four prediction problem:

1. PAIRWISE (Bradley-Terry style)
   Each observation = one matchup between two Final Four teams.
   Features = metric DIFFERENCES (team_A - team_B).
   Target   = did team_A finish better?
   Model    = LightGBM binary classifier.
   Predict  = P(A beats B) for all 6 pairs → aggregate to win probability.

2. RANK-4 (ordinal / softmax)
   Each observation = one Final Four team.
   Target = ordinal finish rank (1=champion … 4=semis loss).
   Model  = LightGBM with ordinal objective or multi-class.

Why pairwise first:
- Turns 4 observations/year into 6 → better sample efficiency
- Every pair is a natural binary problem
- Aggregation to win probability is principled (row normalisation)
- De Prado: meta-labelling fits naturally on top (is the model's favourite the
  actual winner?)
"""

import warnings
import numpy as np
import pandas as pd
from itertools import permutations
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from feature_pipeline.config import LGBM_PARAMS
from feature_pipeline.feature_importance import PurgedYearKFold


# ─────────────────────────────────────────────────────────────────────────────
#  Pairwise frame utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_pairwise(df_ff: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Given the Final Four sub-frame (4 rows per year), build one row per
    ordered pair (A, B).  Features = A_metric - B_metric.
    Target = 1 if A finished better (lower finish_rank).

    We include BOTH orderings (A→B and B→A) so the model sees balanced data.
    """
    from itertools import permutations
    rows = []
    for year, grp in df_ff.groupby("year"):
        teams = grp.set_index("team")
        for a, b in permutations(teams.index, 2):
            row = {"year": year, "team_a": a, "team_b": b}
            for col in feature_cols:
                if col in teams.columns:
                    va = teams.loc[a, col]
                    vb = teams.loc[b, col]
                    row[f"diff_{col}"] = (
                        float(va) - float(vb)
                        if pd.notna(va) and pd.notna(vb) else np.nan
                    )
                else:
                    # Always emit the column so the model sees the right feature count.
                    # Missing columns get NaN here; fillna(0) in pairwise_to_win_prob
                    # treats a missing feature as a zero difference (no advantage either way).
                    row[f"diff_{col}"] = np.nan
            ra = teams.loc[a, "finish_rank"] if "finish_rank" in teams.columns else np.nan
            rb = teams.loc[b, "finish_rank"] if "finish_rank" in teams.columns else np.nan
            if pd.notna(ra) and pd.notna(rb):
                row["a_wins"] = int(float(ra) < float(rb))
            else:
                row["a_wins"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_to_win_prob(model, df_ff: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    For each team in a Final Four group, estimate win probability by:
    1. Running all 12 ordered pairwise comparisons (4 teams × 3 opponents × 2 orders)
    2. P(team wins) = average P(team_A beats all others) over all 3 matchups
    3. Normalise so probabilities in each year sum to 1.

    Returns df_ff with new column 'win_prob'.
    """
    years = df_ff['year'].unique().tolist()
    print(f"\n[pairwise_predict] years={years}  teams={df_ff['team'].tolist()}")
    print(f"[pairwise_predict] feature_cols: {len(feature_cols)}")

    # Audit which features are present / missing / all-NaN in df_ff
    missing_from_df   = [c for c in feature_cols if c not in df_ff.columns]
    present           = [c for c in feature_cols if c in df_ff.columns]
    all_nan_in_df     = [c for c in present if df_ff[c].isna().all()]
    partially_nan     = [c for c in present if df_ff[c].isna().any() and not df_ff[c].isna().all()]

    if missing_from_df:
        print(f"[pairwise_predict] MISSING from df_ff ({len(missing_from_df)}): {missing_from_df}")
    if all_nan_in_df:
        print(f"[pairwise_predict] ALL-NaN in df_ff ({len(all_nan_in_df)}): {all_nan_in_df}")
    if partially_nan:
        print(f"[pairwise_predict] Partially NaN ({len(partially_nan)}): {partially_nan}")
    if not missing_from_df and not all_nan_in_df:
        print(f"[pairwise_predict] All {len(feature_cols)} features present and non-NaN ✓")

    # Always use ALL feature_cols — build_pairwise emits NaN diff for missing/all-NaN cols,
    # and fillna(0) below treats them as zero difference (no advantage either way).
    # This guarantees the model always sees the exact feature count it was trained on.
    pair_df   = build_pairwise(df_ff, feature_cols)
    diff_cols = [f"diff_{c}" for c in feature_cols]  # canonical order matches training

    print(f"[pairwise_predict] pair_df: {pair_df.shape}  diff_cols expected={len(diff_cols)}  found={sum(c in pair_df.columns for c in diff_cols)}")

    nan_counts = pair_df[diff_cols].isna().sum()
    nonzero_nan = nan_counts[nan_counts > 0]
    if not nonzero_nan.empty:
        print(f"[pairwise_predict] diff cols with NaN (will be filled to 0):\n{nonzero_nan.to_string()}")

    X_pair = pair_df[diff_cols].fillna(0)
    print(f"[pairwise_predict] X_pair shape: {X_pair.shape}  (model expects {model.n_features_in_} features)")

    probs = model.predict_proba(X_pair)[:, 1]
    pair_df["p_a_wins"] = probs
    print(f"[pairwise_predict] probs range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Aggregate: for each (year, team_a), average P(A beats each opponent)
    team_prob = (
        pair_df.groupby(["year", "team_a"])["p_a_wins"]
        .mean()
        .reset_index()
        .rename(columns={"team_a": "team", "p_a_wins": "raw_prob"})
    )

    # Normalise within each year so probabilities sum to 1
    team_prob["win_prob"] = team_prob.groupby("year")["raw_prob"].transform(
        lambda x: x / x.sum()
    )
    print(f"[pairwise_predict] win_prob:\n{team_prob[['year','team','win_prob']].to_string(index=False)}")

    df_ff = df_ff.merge(team_prob[["year", "team", "win_prob"]],
                        on=["year", "team"], how="left")
    return df_ff


# ─────────────────────────────────────────────────────────────────────────────
#  LightGBM pairwise model with purged year-CV
# ─────────────────────────────────────────────────────────────────────────────

def train_pairwise_model(pair_df: pd.DataFrame,
                         feature_cols: list,
                         params: dict = None,
                         verbose: bool = True) -> dict:
    """
    Train a LightGBM binary classifier on the pairwise frame.
    Uses leave-one-year-out CV.

    Returns:
        model      – fitted on ALL data (for future prediction)
        cv_results – per-fold metrics
        oof_preds  – out-of-fold predictions (for calibration / meta-labelling)
    """
    if params is None:
        params = LGBM_PARAMS

    diff_cols = [c for c in pair_df.columns if c.startswith("diff_")]
    pair_clean = pair_df.dropna(subset=diff_cols + ["a_wins"]).copy()
    print(f"[train_pairwise] {len(diff_cols)} diff features, {len(pair_clean)} training pairs")
    print(f"[train_pairwise] features: {diff_cols}")
    X = pair_clean[diff_cols].fillna(0).values
    y = pair_clean["a_wins"].astype(int).values
    years = pair_clean["year"].values

    unique_years = sorted(np.unique(years))
    oof_probs = np.full(len(y), np.nan)
    cv_results = []

    for test_year in unique_years:
        tr_idx = np.where(years != test_year)[0]
        te_idx = np.where(years == test_year)[0]
        if len(tr_idx) < 10 or len(te_idx) == 0:
            continue

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[tr_idx], y[tr_idx],
                eval_set=[(X[te_idx], y[te_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)])
        probs = clf.predict_proba(X[te_idx])[:, 1]
        oof_probs[te_idx] = probs

        try:
            auc  = roc_auc_score(y[te_idx], probs)
            ll   = log_loss(y[te_idx], probs)
            brier = brier_score_loss(y[te_idx], probs)
        except Exception:
            auc = ll = brier = np.nan

        cv_results.append({
            "test_year": test_year,
            "n_train":   len(tr_idx),
            "n_test":    len(te_idx),
            "auc":       auc,
            "log_loss":  ll,
            "brier":     brier,
        })
        if verbose:
            print(f"  Year {test_year}: AUC={auc:.3f}  LogLoss={ll:.3f}  Brier={brier:.3f}")

    cv_df = pd.DataFrame(cv_results)
    if verbose and not cv_df.empty:
        print(f"\nMean AUC:      {cv_df['auc'].mean():.3f} ± {cv_df['auc'].std():.3f}")
        print(f"Mean Log-Loss: {cv_df['log_loss'].mean():.3f}")
        print(f"Mean Brier:    {cv_df['brier'].mean():.3f}")

    # Fit final model on ALL data
    final_clf = lgb.LGBMClassifier(**params)
    final_clf.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    return {
        "model":       final_clf,
        "cv_results":  cv_df,
        "oof_probs":   oof_probs,
        "feature_names": diff_cols,
        "X":           X,
        "y":           y,
        "years":       years,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Meta-labelling layer  (de Prado AFML Ch.3 / MLAM Ch.5)
# ─────────────────────────────────────────────────────────────────────────────

def build_meta_labels(df_ff: pd.DataFrame) -> pd.DataFrame:
    """
    Primary model: the predictive rankings (consensus_rank / pred_consensus).
    Meta-label: did the primary model's top pick actually win?

    For each year:
    - Find the team with the best (lowest) consensus_rank = primary prediction
    - Label that team 1 if it won, 0 if it lost
    - This becomes the target for the meta-model

    The meta-model asks: "under what conditions does the favourite actually win?"
    Features: recent form, entropy, CUSUM, awards, SOS gap between 1st and 2nd.
    """
    df = df_ff.copy()
    df["meta_label"]   = np.nan
    df["primary_pick"] = None   # object dtype

    for year, grp in df.groupby("year"):
        if "consensus_rank" not in grp.columns:
            continue
        # Team the primary model would pick
        best_idx = grp["consensus_rank"].idxmin()
        if pd.isna(best_idx):
            continue
        predicted_winner = grp.loc[best_idx, "team"]
        actual_winner    = grp[grp["champion_flag"] == 1]["team"].values

        if len(actual_winner) == 0:
            continue
        # Meta-label: 1 = primary model was right, 0 = it was wrong
        correct = int(predicted_winner in actual_winner)
        df.loc[df["year"] == year, "primary_pick"]  = predicted_winner
        df.loc[df["year"] == year, "meta_label"]    = correct

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Market-as-primary-model meta-label  (de Prado AFML Ch.3)
# ─────────────────────────────────────────────────────────────────────────────

def build_market_meta_labels(df_ff: pd.DataFrame) -> pd.DataFrame:
    """
    For years with Kalshi market data (2025–2026): use mkt_vwap as primary model.
    For years without market data: fall back to consensus_rank as primary model.

    Meta-label = 1 if the primary model's favourite actually won, 0 if not.

    The meta-model then answers: "under what conditions does the market/ranking
    favourite actually win the championship?"
    """
    df = df_ff.copy()
    df["meta_label"]    = np.nan
    df["primary_pick"]  = None   # object dtype to accept team name strings
    df["primary_model"] = None

    for year, grp in df.groupby("year"):
        # Choose primary model: market if available, else consensus rank
        has_market = (
            "mkt_vwap" in grp.columns and
            grp["mkt_vwap"].notna().sum() >= 2
        )

        if has_market:
            # Market: highest VWAP = highest implied probability = primary pick
            best_idx = grp["mkt_vwap"].idxmax()
            primary_model_name = "kalshi_vwap"
        elif "consensus_rank" in grp.columns:
            best_idx = grp["consensus_rank"].idxmin()
            primary_model_name = "consensus_rank"
        else:
            continue

        if pd.isna(best_idx):
            continue

        predicted_winner = grp.loc[best_idx, "team"]
        actual_winners   = grp[grp["champion_flag"] == 1]["team"].values

        correct = int(len(actual_winners) > 0 and predicted_winner in actual_winners)
        df.loc[grp.index, "primary_pick"]   = predicted_winner
        df.loc[grp.index, "meta_label"]     = correct
        df.loc[grp.index, "primary_model"]  = primary_model_name

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Predict current year  (2026)
# ─────────────────────────────────────────────────────────────────────────────

def predict_current_year(model,
                         df_ff_2026: pd.DataFrame,
                         feature_cols: list) -> pd.DataFrame:
    """
    Given the four 2026 Final Four teams and their features,
    output win probabilities.
    """
    df_2026 = df_ff_2026.copy()
    if model is not None:
        df_2026 = pairwise_to_win_prob(model, df_2026, feature_cols)
    # If win_prob already set (fallback path from run.py), skip pairwise

    display_cols = ["team", "win_prob", "consensus_rank", "pred_consensus",
                    "overall_win_pct", "q1_win_pct", "resume_score",
                    "mkt_vwap", "kg_scoring_margin", "kg_tourney_avg_margin"]
    present_cols = [c for c in display_cols if c in df_2026.columns]
    result = (
        df_2026[present_cols]
        .sort_values("win_prob", ascending=False)
        .reset_index(drop=True)
    )
    result.index += 1
    result.index.name = "predicted_rank"
    return result
