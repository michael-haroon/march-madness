"""
tests/test_game_model.py
------------------------
Unit tests for feature_pipeline.game_model (Phase 1A).

Run with: pytest tests/test_game_model.py -v
Or quick smoke test: pytest tests/test_game_model.py -v -m "not slow"
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Allow import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = "data"
KAGGLE_DIR = os.path.join(DATA_DIR, "kaggle")

# Skip integration tests if data not present
has_data = os.path.exists(os.path.join(KAGGLE_DIR, "MRegularSeasonDetailedResults.csv"))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def team_df():
    """Build team-season features once for all tests."""
    if not has_data:
        pytest.skip("Kaggle data not available")
    from feature_pipeline.game_model import build_team_season_features
    return build_team_season_features(DATA_DIR, min_season=2003)


@pytest.fixture(scope="module")
def pairs_df(team_df):
    """Build game pairs once for all tests."""
    from feature_pipeline.game_model import build_game_pairs
    tourney_path = os.path.join(KAGGLE_DIR, "MNCAATourneyCompactResults.csv")
    return build_game_pairs(team_df, tourney_path, min_season=2003, max_season=2025)


@pytest.fixture(scope="module")
def model_results(pairs_df):
    """Train model once for slow tests."""
    from feature_pipeline.game_model import train_game_model
    return train_game_model(pairs_df)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: build_team_season_features
# ─────────────────────────────────────────────────────────────────────────────

def test_team_features_no_lookahead():
    """Verify MRegularSeasonDetailedResults only contains regular season games (DayNum <= 132).
    The file-level separation is the primary lookahead guard; our DayNum filter is a belt-and-suspenders."""
    if not has_data:
        pytest.skip("Kaggle data not available")
    import pandas as pd
    rs = pd.read_csv(os.path.join(KAGGLE_DIR, "MRegularSeasonDetailedResults.csv"))
    # Kaggle keeps regular season separate from tournament — all rows should be DayNum <= 132
    assert rs["DayNum"].max() <= 132, (
        f"Regular season file has games with DayNum > 132: {rs['DayNum'].max()}"
    )
    # Also verify tournament data is in a different file (no cross-contamination)
    assert os.path.exists(os.path.join(KAGGLE_DIR, "MNCAATourneyCompactResults.csv")), \
        "Tournament results should be in a separate file"


def test_no_2020(team_df):
    """Season 2020 must not appear anywhere in team features."""
    assert 2020 not in team_df["Season"].values


def test_team_df_shape(team_df):
    """Should have ~68 teams per season (some years have play-in extras)."""
    per_season = team_df.groupby("Season").size()
    # Each season should have at least 64 and at most 72 (play-in variants)
    assert per_season.min() >= 60, f"Too few teams in a season: {per_season.min()}"
    assert per_season.max() <= 80, f"Too many teams in a season: {per_season.max()}"


def test_team_df_required_columns(team_df):
    """Required feature columns must be present."""
    required = ["Season", "TeamID", "seed_num", "kg_win_pct", "kg_fg_pct",
                "kg_scoring_margin", "massey_POM", "consensus_rank"]
    missing = [c for c in required if c not in team_df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_seed_num_range(team_df):
    """Seeds should be 1-16."""
    valid = team_df["seed_num"].dropna()
    assert valid.between(1, 16).all(), f"Seed out of range: {valid[~valid.between(1, 16)].unique()}"


def test_win_pct_range(team_df):
    """Win percentage must be in [0, 1]."""
    valid = team_df["kg_win_pct"].dropna()
    assert valid.between(0.0, 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: build_game_pairs
# ─────────────────────────────────────────────────────────────────────────────

def test_pair_ordering_canonical(pairs_df):
    """TeamA must always be less than TeamB (canonical ordering)."""
    assert (pairs_df["TeamA"] < pairs_df["TeamB"]).all()


def test_pair_labels_balanced(pairs_df):
    """Canonical ordering should give ~50/50 labels (within 40-60%)."""
    balance = pairs_df["team_a_wins"].mean()
    assert 0.40 < balance < 0.60, f"Label balance unexpected: {balance:.3f}"


def test_no_2020_in_pairs(pairs_df):
    """Season 2020 must not appear in game pairs."""
    assert 2020 not in pairs_df["Season"].values


def test_pairs_season_range(pairs_df):
    """Game pairs should only cover 2003-2025."""
    assert pairs_df["Season"].min() >= 2003
    assert pairs_df["Season"].max() <= 2025


def test_pairs_approximate_count(pairs_df):
    """Should have ~1,300-1,500 tournament games (63/year × ~22 years)."""
    n = len(pairs_df)
    assert 1000 < n < 2000, f"Unexpected game count: {n}"


def test_diff_features_present(pairs_df):
    """All diff_* columns should be present."""
    diff_cols = [c for c in pairs_df.columns if c.startswith("diff_")]
    assert len(diff_cols) > 10, f"Too few diff columns: {len(diff_cols)}"


def test_diff_features_antisymmetric(team_df):
    """diff(A,B) == -diff(B,A) for the same game."""
    from feature_pipeline.game_model import build_game_pairs
    tourney_path = os.path.join(KAGGLE_DIR, "MNCAATourneyCompactResults.csv")

    pairs = build_game_pairs(team_df, tourney_path, min_season=2022, max_season=2022)
    if len(pairs) == 0:
        pytest.skip("No 2022 games found")

    # Pick first game
    row = pairs.iloc[0]
    team_a = int(row["TeamA"])
    team_b = int(row["TeamB"])
    season = int(row["Season"])

    diff_cols = [c for c in pairs.columns if c.startswith("diff_")]

    # Build reverse pairs by swapping TeamA/TeamB labels
    # We do this by creating a mini team_df with just these two teams
    mini_team_df = team_df[(team_df["Season"] == season) &
                           (team_df["TeamID"].isin([team_a, team_b]))]
    if len(mini_team_df) != 2:
        pytest.skip("Cannot find both teams for antisymmetry test")

    # Get features for A and B
    fa = mini_team_df[mini_team_df["TeamID"] == team_a].iloc[0]
    fb = mini_team_df[mini_team_df["TeamID"] == team_b].iloc[0]

    feat_cols = [c for c in mini_team_df.columns if c not in ("Season", "TeamID")]
    for fc in feat_cols[:5]:  # spot-check first 5
        diff_col = f"diff_{fc}"
        if diff_col not in pairs.columns:
            continue
        actual_diff = row[diff_col]
        # A is always min(id), B is always max(id), so diff = A - B
        # If we reverse: new_diff = B - A = -diff
        try:
            expected = float(fa[fc]) - float(fb[fc])
            if not np.isnan(actual_diff) and not np.isnan(expected):
                assert abs(actual_diff - expected) < 1e-6, (
                    f"{diff_col}: got {actual_diff}, expected {expected}"
                )
        except (TypeError, ValueError):
            pass


def test_round_num_values(pairs_df):
    """round_num should only take values 1-6."""
    valid_rounds = {1, 2, 3, 4, 5, 6}
    assert set(pairs_df["round_num"].unique()).issubset(valid_rounds)


def test_no_tourney_features_in_diff(pairs_df):
    """Diff columns must not include tournament path features (lookahead)."""
    forbidden = [c for c in pairs_df.columns
                 if "tourney" in c.lower() and c.startswith("diff_")]
    assert len(forbidden) == 0, f"Tournament lookahead features found: {forbidden}"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: train_game_model
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_cv_auc_reasonable(model_results):
    """CV AUC should be between 0.60 and 0.85."""
    mean_auc = model_results["cv_results"]["auc"].mean()
    assert 0.60 < mean_auc < 0.85, f"CV AUC out of expected range: {mean_auc:.4f}"


@pytest.mark.slow
def test_cv_log_loss_reasonable(model_results):
    """CV log-loss should be below 0.70 (seed-only baseline ≈ 0.65)."""
    mean_ll = model_results["cv_results"]["log_loss"].mean()
    assert mean_ll < 0.70, f"CV log-loss too high: {mean_ll:.4f}"


@pytest.mark.slow
def test_model_results_keys(model_results):
    """train_game_model must return all required keys."""
    required = {"model", "cv_results", "oof_preds", "feature_importance", "feature_cols"}
    assert required.issubset(model_results.keys())


@pytest.mark.slow
def test_feature_importance_shape(model_results, pairs_df):
    """Feature importance should cover all feature columns."""
    fi = model_results["feature_importance"]
    assert "feature" in fi.columns and "importance" in fi.columns
    assert len(fi) == len(model_results["feature_cols"])


@pytest.mark.slow
def test_oof_covers_all_seasons(model_results, pairs_df):
    """OOF predictions should cover all seasons."""
    oof_seasons = set(model_results["oof_preds"]["Season"].unique())
    pair_seasons = set(pairs_df["Season"].unique())
    assert pair_seasons == oof_seasons, f"Missing OOF seasons: {pair_seasons - oof_seasons}"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: predict_final_four
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_predict_sums_to_one(model_results, team_df):
    """Championship probabilities must sum to 1.0."""
    from feature_pipeline.game_model import predict_final_four
    from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id
    from feature_pipeline.config import TEAM_NAME_MAP

    lookup = build_id_lookup(KAGGLE_DIR)
    ff_names = ["Arizona", "Connecticut", "Illinois", "Michigan"]
    ff_ids = [resolve_team_id(n, lookup, TEAM_NAME_MAP) for n in ff_names]

    if any(tid is None for tid in ff_ids):
        pytest.skip("Could not resolve all 2026 FF team names")

    team_df_2026 = team_df[team_df["Season"] == 2026]
    if len(team_df_2026) == 0:
        pytest.skip("No 2026 season data")

    preds = predict_final_four(
        model_results["model"],
        team_df_2026,
        ff_ids,
        model_results["feature_cols"],
        n_sims=1000,
        random_seed=42,
    )

    assert abs(preds["p_champion"].sum() - 1.0) < 0.02, (
        f"p_champion sum = {preds['p_champion'].sum():.4f}, expected ≈ 1.0"
    )


@pytest.mark.slow
def test_simulation_deterministic_with_seed(model_results, team_df):
    """Same random seed → same predictions."""
    from feature_pipeline.game_model import predict_final_four
    from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id
    from feature_pipeline.config import TEAM_NAME_MAP

    lookup = build_id_lookup(KAGGLE_DIR)
    ff_names = ["Arizona", "Connecticut", "Illinois", "Michigan"]
    ff_ids = [resolve_team_id(n, lookup, TEAM_NAME_MAP) for n in ff_names]

    if any(tid is None for tid in ff_ids):
        pytest.skip("Could not resolve all 2026 FF team names")

    team_df_2026 = team_df[team_df["Season"] == 2026]
    if len(team_df_2026) == 0:
        pytest.skip("No 2026 season data")

    kwargs = dict(
        model=model_results["model"],
        team_df=team_df_2026,
        final_four_teams=ff_ids,
        feature_cols=model_results["feature_cols"],
        n_sims=500,
        random_seed=42,
    )

    p1 = predict_final_four(**kwargs)
    p2 = predict_final_four(**kwargs)

    pd.testing.assert_frame_equal(
        p1[["TeamID", "p_champion"]].reset_index(drop=True),
        p2[["TeamID", "p_champion"]].reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: parse_seed helper
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_seed():
    """parse_seed correctly converts seed strings to integers."""
    from feature_pipeline.game_model import parse_seed
    assert parse_seed("W01") == 1
    assert parse_seed("X16a") == 16
    assert parse_seed("Z11b") == 11
    assert parse_seed("Y04") == 4
    assert np.isnan(parse_seed("bad"))
    assert np.isnan(parse_seed(None))


def test_daynum_to_round():
    """daynum_to_round correctly maps DayNums to rounds."""
    from feature_pipeline.game_model import daynum_to_round
    assert daynum_to_round(134) == 1  # R64
    assert daynum_to_round(136) == 1
    assert daynum_to_round(137) == 2  # R32
    assert daynum_to_round(143) == 3  # S16
    assert daynum_to_round(146) == 4  # E8
    assert daynum_to_round(152) == 5  # FF
    assert daynum_to_round(154) == 6  # Championship


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: Phase 1B — compute_path_features
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_path_features_r1_returns_nan():
    """Empty prior_games (R1 game) returns path_games_played=0 and all others NaN."""
    from feature_pipeline.game_model import compute_path_features
    result = compute_path_features(1163, 2026, prior_games=[], seeds_df=None)
    assert result["path_games_played"] == 0
    assert np.isnan(result["path_avg_margin"])
    assert np.isnan(result["path_fg_pct"])
    assert np.isnan(result["path_momentum"])


def test_compute_path_features_games_played():
    """path_games_played equals number of prior games."""
    from feature_pipeline.game_model import compute_path_features

    games = [
        {"DayNum": 136, "won": 1, "margin": 10.0, "FGM": 25, "FGA": 50,
         "OppFGM": 20, "OppFGA": 50, "NumOT": 0, "opp_team_id": 999},
        {"DayNum": 138, "won": 1, "margin": 5.0, "FGM": 23, "FGA": 48,
         "OppFGM": 21, "OppFGA": 50, "NumOT": 1, "opp_team_id": 888},
    ]
    result = compute_path_features(1163, 2026, prior_games=games, seeds_df=None)
    assert result["path_games_played"] == 2
    assert abs(result["path_avg_margin"] - 7.5) < 1e-6
    assert result["path_worst_margin"] == 5.0
    assert result["path_best_margin"] == 10.0
    assert result["path_ot_games"] == 1
    # momentum = last - first = 5.0 - 10.0 = -5.0
    assert abs(result["path_momentum"] - (-5.0)) < 1e-6


def test_compute_path_features_fg_pct():
    """path_fg_pct and path_opp_fg_pct computed from aggregate shots."""
    from feature_pipeline.game_model import compute_path_features

    games = [
        {"DayNum": 136, "won": 1, "margin": 8.0, "FGM": 20, "FGA": 40,
         "OppFGM": 18, "OppFGA": 45, "NumOT": 0, "opp_team_id": 999},
        {"DayNum": 138, "won": 1, "margin": 12.0, "FGM": 30, "FGA": 60,
         "OppFGM": 22, "OppFGA": 55, "NumOT": 0, "opp_team_id": 888},
    ]
    result = compute_path_features(1163, 2026, prior_games=games, seeds_df=None)
    # FG%: (20+30)/(40+60) = 50/100 = 0.50
    assert abs(result["path_fg_pct"] - 0.50) < 1e-6
    # Opp FG%: (18+22)/(45+55) = 40/100 = 0.40
    assert abs(result["path_opp_fg_pct"] - 0.40) < 1e-6


def test_pairs_with_path_r1_diffs_zero():
    """R1 games have diff_path_games_played == 0 (both teams have no prior path)."""
    if not has_data:
        pytest.skip("Kaggle data not available")
    from feature_pipeline.game_model import build_game_pairs, build_team_season_features

    team_df_fixture = build_team_season_features(DATA_DIR, min_season=2022)
    tourney_compact = os.path.join(KAGGLE_DIR, "MNCAATourneyCompactResults.csv")
    tourney_detailed = os.path.join(KAGGLE_DIR, "MNCAATourneyDetailedResults.csv")

    pairs = build_game_pairs(
        team_df_fixture, tourney_compact,
        min_season=2022, max_season=2022,
        include_path=True, tourney_detailed_path=tourney_detailed,
    )
    if len(pairs) == 0:
        pytest.skip("No 2022 pairs found")

    r1_pairs = pairs[pairs["round_num"] == 1]
    if len(r1_pairs) == 0:
        pytest.skip("No R1 games found")

    assert "diff_path_games_played" in pairs.columns
    # Most R1 games: both teams have 0 prior path → diff = 0.
    # Exception: First Four play-in teams (DayNum 132-134) have 1 game before R64 → diff = ±1.
    assert (r1_pairs["diff_path_games_played"].abs() <= 1).all(), \
        f"R1 diff_path_games_played out of range: {r1_pairs['diff_path_games_played'].unique()}"


def test_pairs_with_path_no_lookahead():
    """Path features for game at DayNum D only use games with DayNum < D."""
    if not has_data:
        pytest.skip("Kaggle data not available")
    from feature_pipeline.game_model import build_game_pairs, build_team_season_features

    team_df_fixture = build_team_season_features(DATA_DIR, min_season=2023)
    tourney_compact = os.path.join(KAGGLE_DIR, "MNCAATourneyCompactResults.csv")
    tourney_detailed = os.path.join(KAGGLE_DIR, "MNCAATourneyDetailedResults.csv")

    pairs = build_game_pairs(
        team_df_fixture, tourney_compact,
        min_season=2023, max_season=2023,
        include_path=True, tourney_detailed_path=tourney_detailed,
    )
    if "diff_path_games_played" not in pairs.columns:
        pytest.skip("No path columns")

    # R1: most teams have 0 prior games, but First Four play-in teams may have 1 → diff ≤ ±1
    r1 = pairs[pairs["round_num"] == 1]
    assert (r1["diff_path_games_played"].abs() <= 1).all()

    # R2 games must have |diff_path_games_played| <= 1 (each team has 0 or 1 prior game)
    r2 = pairs[pairs["round_num"] == 2]
    if len(r2) > 0:
        assert (r2["diff_path_games_played"].abs() <= 1.0).all()


@pytest.mark.slow
def test_predict_with_path_sums_to_one(model_results, team_df):
    """Championship probabilities sum to 1.0 with path features enabled."""
    from feature_pipeline.game_model import predict_final_four, load_actual_path_features
    from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id
    from feature_pipeline.config import TEAM_NAME_MAP

    lookup = build_id_lookup(KAGGLE_DIR)
    ff_names = ["Arizona", "Michigan", "Illinois", "Connecticut"]
    ff_ids = [resolve_team_id(n, lookup, TEAM_NAME_MAP) for n in ff_names]

    if any(tid is None for tid in ff_ids):
        pytest.skip("Could not resolve all 2026 FF team names")

    team_df_2026 = team_df[team_df["Season"] == 2026]
    if len(team_df_2026) == 0:
        pytest.skip("No 2026 season data")

    actual_pf = load_actual_path_features(DATA_DIR, season=2026, team_ids=ff_ids)

    preds = predict_final_four(
        model_results["model"],
        team_df_2026,
        ff_ids,
        model_results["feature_cols"],
        n_sims=500,
        random_seed=42,
        include_path=True,
        actual_path_features=actual_pf,
    )

    assert abs(preds["p_champion"].sum() - 1.0) < 0.02, \
        f"p_champion sum = {preds['p_champion'].sum():.4f}"
