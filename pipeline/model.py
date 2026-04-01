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
from pipeline.config import LGBM_PARAMS
from pipeline.feature_importance import PurgedYearKFold


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
    pair_df  = build_pairwise(df_ff, feature_cols)
    diff_cols = [c for c in pair_df.columns if c.startswith("diff_")]

    print(f"[DEBUG pairwise_to_win_prob] df_ff shape: {df_ff.shape}, years: {df_ff['year'].unique().tolist()}")
    print(f"[DEBUG pairwise_to_win_prob] feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"[DEBUG pairwise_to_win_prob] pair_df shape before dropna: {pair_df.shape}, diff_cols: {len(diff_cols)}")
    if diff_cols:
        nan_counts = pair_df[diff_cols].isna().sum()
        print(f"[DEBUG pairwise_to_win_prob] NaN counts per diff_col:\n{nan_counts[nan_counts > 0]}")

    # Drop rows where diff features are NaN; a_wins can be NaN for prediction years
    pair_df = pair_df.dropna(subset=diff_cols)
    print(f"[DEBUG pairwise_to_win_prob] pair_df shape after dropna: {pair_df.shape}")
    if pair_df.empty:
        print("[DEBUG pairwise_to_win_prob] pair_df is EMPTY after dropna — returning NaN win_prob")
        df_ff["win_prob"] = np.nan
        return df_ff

    X_pair = pair_df[diff_cols].fillna(0)
    probs  = model.predict_proba(X_pair)[:, 1]
    pair_df["p_a_wins"] = probs
    print(f"[DEBUG pairwise_to_win_prob] predict_proba done, probs range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Aggregate: for each (year, team_a), average P(A beats each opponent)
    team_prob = (
        pair_df.groupby(["year", "team_a"])["p_a_wins"]
        .mean()
        .reset_index()
        .rename(columns={"team_a": "team", "p_a_wins": "raw_prob"})
    )
    print(f"[DEBUG pairwise_to_win_prob] team_prob:\n{team_prob}")

    # Normalise within each year so probabilities sum to 1
    team_prob["win_prob"] = team_prob.groupby("year")["raw_prob"].transform(
        lambda x: x / x.sum()
    )

    print(f"[DEBUG pairwise_to_win_prob] merging on year+team. df_ff years/teams: {list(zip(df_ff['year'], df_ff['team']))}")
    print(f"[DEBUG pairwise_to_win_prob] team_prob years/teams: {list(zip(team_prob['year'], team_prob['team']))}")
    df_ff = df_ff.merge(team_prob[["year", "team", "win_prob"]],
                        on=["year", "team"], how="left")
    print(f"[DEBUG pairwise_to_win_prob] after merge win_prob:\n{df_ff[['year','team','win_prob']]}")
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
