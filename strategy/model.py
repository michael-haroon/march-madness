"""
model.py
--------
Train pairwise classifiers on surviving features with leave-one-year-out CV.
LightGBM and logistic regression, with Platt calibration.
"""

import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

from strategy.config import SURVIVING_FEATURES, LGBM_PARAMS, LOGREG_PARAMS


SKIP_SEASON = 2020


def train_and_evaluate(pairs_df: pd.DataFrame,
                       features: list = None,
                       model_type: str = "lgbm") -> dict:
    """
    Train a pairwise classifier with leave-one-year-out CV.

    Args:
        pairs_df: game pairs with feature columns and 'team_a_wins', 'Season'
        features: list of feature column names (default: SURVIVING_FEATURES)
        model_type: 'lgbm' or 'logreg'

    Returns dict with: model, cv_results, oof_preds, feature_cols
    """
    if features is None:
        features = [f for f in SURVIVING_FEATURES if f in pairs_df.columns]

    X = pairs_df[features].copy()
    y = pairs_df["team_a_wins"].astype(int)
    seasons = sorted(pairs_df["Season"].unique())

    cv_rows = []
    oof_rows = []

    for season in seasons:
        if season == SKIP_SEASON:
            continue
        train_mask = (pairs_df["Season"] != season) & (pairs_df["Season"] != SKIP_SEASON)
        val_mask = pairs_df["Season"] == season

        X_tr, X_val = X[train_mask], X[val_mask]
        y_tr, y_val = y[train_mask], y[val_mask]

        if len(X_val) == 0 or y_tr.nunique() < 2:
            continue

        mdl = _build_model(model_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(X_tr, y_tr)

        proba = mdl.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, proba) if y_val.nunique() > 1 else np.nan
        ll = log_loss(y_val, proba)
        brier = brier_score_loss(y_val, proba)
        acc = accuracy_score(y_val, (proba >= 0.5).astype(int))

        cv_rows.append({
            "season": season, "auc": auc, "log_loss": ll,
            "brier": brier, "accuracy": acc, "n_games": int(val_mask.sum()),
        })

        for i, idx in enumerate(pairs_df.index[val_mask]):
            oof_rows.append({
                "index": idx, "Season": season,
                "y_true": int(y_val.iloc[i]), "y_pred": float(proba[i]),
            })

    cv_results = pd.DataFrame(cv_rows)

    # Final model on all data
    final_model = _build_model(model_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X, y)

    oof_df = pd.DataFrame(oof_rows)

    return {
        "model": final_model,
        "cv_results": cv_results,
        "oof_preds": oof_df,
        "feature_cols": features,
        "model_type": model_type,
    }


def calibrate(model, X: pd.DataFrame, y: pd.Series,
              method: str = "sigmoid") -> CalibratedClassifierCV:
    """
    Platt calibration (sigmoid) on the full training set.
    Returns a CalibratedClassifierCV wrapper.
    """
    cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    cal.fit(X, y)
    return cal


def get_matchup_prob(model, feature_vector: np.ndarray) -> float:
    """Return P(team_a_wins) for a single matchup feature vector."""
    prob = model.predict_proba(feature_vector)[:, 1]
    return float(prob[0])


def compare_models(results: dict) -> str:
    """
    Compare CV metrics across models, print summary, return name of best.
    results: {model_name: train_and_evaluate output dict}
    """
    print("\n=== Model Comparison (Leave-One-Year-Out CV) ===")
    print(f"{'Model':<12} {'AUC':>7} {'LogLoss':>9} {'Brier':>7} {'Acc':>7}")
    print("-" * 46)

    best_name = None
    best_ll = np.inf

    for name, res in results.items():
        cv = res["cv_results"]
        auc = cv["auc"].mean()
        ll = cv["log_loss"].mean()
        brier = cv["brier"].mean()
        acc = cv["accuracy"].mean()
        print(f"{name:<12} {auc:>7.4f} {ll:>9.4f} {brier:>7.4f} {acc:>7.4f}")

        if ll < best_ll:
            best_ll = ll
            best_name = name

    print(f"\nBest by log-loss: {best_name}")
    return best_name


def _build_model(model_type: str):
    if model_type == "lgbm":
        return LGBMClassifier(**LGBM_PARAMS)
    elif model_type == "logreg":
        return LogisticRegression(**LOGREG_PARAMS)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
