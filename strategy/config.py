"""
config.py
---------
Strategy-specific configuration. Self-contained — does not import from feature_pipeline.
"""

# ── Surviving features (de Prado MDI/MDA/SFI filtering) ─────────────────────
# Source: output_v2/features/filtered/feature_list.txt
# TeamB included — it passed all three de Prado tests. Kaggle TeamIDs were assigned
# by order of database entry, which correlates with historical program prestige.
SURVIVING_FEATURES = [
    "diff_massey_WLK",
    "diff_massey_POM",
    "diff_seed_num",
    "diff_massey_PMW",
    "diff_massey_MOR",
    "diff_massey_DOL",
    "diff_massey_RPI",
    "diff_consensus_rank",
    "diff_massey_SAG",
    "diff_massey_COL",
    "diff_kg_wins",
    "diff_kg_scoring_margin",
    "diff_kg_margin_last10_delta",
    "diff_kg_win_pct",
    "diff_kg_losses",
    "diff_massey_BPI",
    "diff_path_avg_opp_seed",
    "diff_path_best_opp_seed",
    "diff_kg_net_strong_margin",
    "TeamB",
]

# ── LightGBM params (tighter regularization than feature_pipeline) ──────────
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_samples": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "verbosity": -1,
}

# ── Logistic regression params ──────────────────────────────────────────────
LOGREG_PARAMS = {
    "C": 0.1,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}

# ── 2026 Final Four bracket ─────────────────────────────────────────────────
BRACKET_2026 = {
    "semi1": ("Arizona", "Michigan"),
    "semi2": ("Illinois", "Connecticut"),
}

# ── Market blend ────────────────────────────────────────────────────────────
DEFAULT_MARKET_WEIGHT = 0.10

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
GAME_PAIRS_PATH = "output_v2/features/game_pairs.csv"
TEAM_FEATURES_PATH = "output_v2/features/team_season_features.csv"
OUTPUT_DIR = "strategy/output"
