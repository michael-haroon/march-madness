# SPEC: Pairwise Game-Level Model Refactor

**Status**: Draft  
**Author**: Claude (spec), Sonnet (implementation)  
**Date**: 2026-04-02  

---

## Motivation

The current pipeline predicts "who wins the championship" from ~84 Final Four rows (21 seasons x 4 teams). This is a tiny-sample regime where any ML model will overfit or be uninformative.

**The fix**: reframe the problem as "who wins THIS game?" using all NCAA tournament games (2003-2025, ~1,400 games). This gives 20x more training data while answering the same question — we just chain pairwise predictions through the Final Four bracket.

### What changes
| Aspect | Current | New |
|--------|---------|-----|
| Unit of observation | (year, team) | (year, game) = (TeamA vs TeamB) |
| Training rows | ~84 (Final Four only) | ~1,400 (all tourney games 2003-2025) |
| Label | `champion_flag` or `finish_rank` | `team_a_wins` (binary) |
| Feature format | Raw team metrics | **Difference** features (TeamA metric - TeamB metric) |
| Prediction target | Win probability among 4 teams | P(TeamA beats TeamB) for each FF matchup |
| CV scheme | Leave-one-year-out on 21 folds | Leave-one-year-out on 22 seasons |

### What stays the same
- All data loading (`data_loader.py`) — keep as-is
- Feature engineering (`feature_engineering.py`) — keep all team-level feature builders
- `name_resolver.py`, `market_features.py`, `config.py` — unchanged
- Feature importance methods (`feature_importance.py`) — unchanged, just called on new data

---

## Architecture Overview

```
                     EXISTING (keep)                          NEW
                 ┌─────────────────────┐          ┌─────────────────────────┐
                 │   data_loader.py    │          │                         │
                 │   load_all() → df   │─────────▶│  game_model.py          │
                 │   (year, team)      │          │                         │
                 └─────────────────────┘          │  1. build_team_features │
                 ┌─────────────────────┐          │     (year, TeamID) →    │
                 │feature_engineering.py│─────────▶│     feature vector      │
                 │   build_features()  │          │                         │
                 └─────────────────────┘          │  2. build_game_pairs    │
                                                  │     (year, game) →      │
                 ┌─────────────────────┐          │     diff features       │
                 │ Kaggle tourney CSV  │─────────▶│                         │
                 │ MNCAATourneyDetailed│          │  3. train / predict     │
                 │ Results.csv         │          │     LightGBM pairwise   │
                 └─────────────────────┘          │                         │
                                                  │  4. predict_final_four  │
                 ┌─────────────────────┐          │     Chain 3 games →     │
                 │ market_features.py  │─────────▶│     championship prob   │
                 │ (2025-2026 only)    │          │                         │
                 └─────────────────────┘          └─────────────────────────┘
                                                             │
                                                             ▼
                                                  ┌─────────────────────────┐
                                                  │  run_v2.py              │
                                                  │  Orchestrates new flow  │
                                                  └─────────────────────────┘
```

---

## File-by-File Implementation Plan

### File 1: `feature_pipeline/game_model.py` (NEW — ~400 lines)

This is the core new file. It builds game-level pairwise features and trains the model.

#### Function 1: `build_team_season_features(data_dir, min_season=2003) -> pd.DataFrame`

**Purpose**: Build one feature vector per (season, TeamID) for ALL tournament teams, using only pre-tournament data (DayNum <= 132).

**Input**: `data_dir` (str) — path to `data/` directory

**Output**: DataFrame indexed by `(Season, TeamID)` with columns:
```
# From Kaggle regular season box scores (DayNum <= 132)
kg_wins, kg_losses, kg_win_pct,
kg_fg_pct, kg_fg3_pct, kg_ft_pct, kg_efg_pct,
kg_off_reb_pg, kg_def_reb_pg, kg_ast_pg, kg_to_pg, kg_stl_pg, kg_blk_pg,
kg_scoring_margin, kg_opp_fg_pct,
kg_margin_last5, kg_margin_last10,

# From Massey ordinals (last available before DayNum 133)
massey_POM, massey_SAG, massey_RPI, massey_MOR, massey_WLK, massey_DOL, massey_COL,

# From seeds
seed_num,

# Derived
consensus_rank (mean of available Massey systems),
rank_spread (std of available Massey systems),
```

**Implementation notes**:
- Load `MRegularSeasonDetailedResults.csv`. Filter `DayNum <= 132`.
- For each (Season, TeamID), compute stats from BOTH the W* and L* columns (team appears as winner in some rows, loser in others).
- For `kg_margin_lastN`: sort by DayNum descending, take last N games, compute mean scoring margin.
- Load `MMasseyOrdinals.csv`. For each (Season, TeamID, SystemName), take the row with max `RankingDayNum <= 133`. Pivot so each system is a column.
- Load `MNCAATourneySeeds.csv`. Parse seed string (e.g., "W01" -> 1, "X16b" -> 16). Join on (Season, TeamID).
- Skip Season=2020 everywhere.
- Return shape: ~1,500 rows (68 teams/year x 22 seasons).

**Critical**: This function does NOT call `load_all()` or `build_features()`. It reads Kaggle CSVs directly for speed and clarity. The existing pipeline's team-level features (team sheets, SOS, quadrant records, etc.) are integrated in a later step (see `enrich_with_existing_features`).

#### Function 2: `enrich_with_existing_features(team_df, existing_df) -> pd.DataFrame`

**Purpose**: Merge the Kaggle-derived team features with features from the existing pipeline (team sheets, awards, Massey from team sheets, market data).

**Input**:
- `team_df`: output of `build_team_season_features` — indexed by (Season, TeamID)
- `existing_df`: output of `load_all() |> build_features()` — indexed by (year, team) with string team names

**Output**: `team_df` enriched with additional columns from `existing_df`

**Implementation notes**:
- Use `name_resolver.build_id_lookup()` to map `existing_df["team"]` -> `TeamID`.
- Join on `(year=Season, TeamID)`.
- Columns to pull from existing_df:
  - Team sheet ranks: `net_rank`, `kpi`, `sor`, `bpi` (2005+ only, NaN for 2003-2004)
  - SOS features: `net_sos`, `rpi_sos`, `net_nc_sos`
  - Record splits: `overall_win_pct`, `road_win_pct`, `nc_win_pct`, `conf_win_pct`
  - Quadrant: `q1_win_pct`, `resume_score`
  - De Prado: `win_entropy`, `cusum_peak`, `net_rank_yoy`, `kpi_yoy`, `sor_yoy`
  - Awards: `total_awards`, `has_player_of_year_award`
  - Market (2025-2026 only): `mkt_vwap`, `mkt_ofi`, `mkt_trade_count`, `mkt_volatility`
- Do NOT pull `kg_*` columns from existing_df (we already computed them fresh from game data).
- Do NOT pull tournament path features (`kg_tourney_*`) — those leak future info for non-FF games.

#### Function 3: `build_game_pairs(team_df, tourney_results_path, min_season=2003, max_season=2025) -> pd.DataFrame`

**Purpose**: Create one row per tournament game with difference features.

**Input**:
- `team_df`: output of `build_team_season_features` (or enriched version)
- `tourney_results_path`: path to `MNCAATourneyCompactResults.csv` (or Detailed)

**Output**: DataFrame with one row per game:
```
Season, DayNum, TeamA, TeamB,
diff_kg_win_pct, diff_kg_fg_pct, diff_kg_scoring_margin, ...,
diff_seed_num, diff_consensus_rank, ...,
seed_matchup (e.g., "1v16", "5v12"),
round_num (1=R64, 2=R32, 3=S16, 4=E8, 5=FF, 6=Championship),
team_a_wins (label: 1 if TeamA won, 0 if TeamB won)
```

**Implementation notes**:
- Load tournament results CSV. Each row has `WTeamID`, `LTeamID`.
- **Canonicalize ordering**: always set `TeamA = min(WTeamID, LTeamID)`, `TeamB = max(...)`. This removes winner/loser ordering bias. Set `team_a_wins = 1` if `TeamA == WTeamID`, else 0.
- For each feature column `col` in `team_df`: `diff_{col} = team_df[TeamA][col] - team_df[TeamB][col]`.
- Derive `round_num` from DayNum:
  ```python
  def daynum_to_round(daynum):
      if daynum <= 136: return 1    # First Four / R64
      elif daynum <= 138: return 2  # R32
      elif daynum <= 145: return 3  # Sweet 16
      elif daynum <= 148: return 4  # Elite 8
      elif daynum <= 152: return 5  # Final Four
      else: return 6                # Championship
  ```
- Add `seed_matchup` as a categorical: `f"{min(seedA,seedB)}v{max(seedA,seedB)}"`.
- Filter: `min_season <= Season <= max_season`, skip Season=2020.
- Expected output: ~1,400 rows (roughly 63 games/year x 22 years).

#### Function 4: `train_game_model(pairs_df, feature_cols=None, params=None) -> dict`

**Purpose**: Train LightGBM on pairwise game data with leave-one-year-out CV.

**Input**:
- `pairs_df`: output of `build_game_pairs`
- `feature_cols`: list of `diff_*` columns to use. If None, auto-detect all `diff_*` columns.
- `params`: LightGBM params dict. Default from `config.LGBM_PARAMS` but with these overrides:
  ```python
  {
      "n_estimators": 500,       # more data → can use more trees
      "learning_rate": 0.03,     # slower learning, more trees
      "max_depth": 5,            # slightly deeper OK with 1400 rows
      "min_child_samples": 10,
      "subsample": 0.8,
      "colsample_bytree": 0.7,
      "reg_alpha": 0.1,          # L1 regularization
      "reg_lambda": 1.0,         # L2 regularization
  }
  ```

**Output**: dict with keys:
- `model`: LGBMClassifier fitted on ALL training data (2003-2025)
- `cv_results`: DataFrame with per-year metrics: `season`, `auc`, `log_loss`, `brier`, `accuracy`
- `oof_preds`: DataFrame with out-of-fold predictions for every game
- `feature_importance`: DataFrame with `feature`, `importance` (gain-based)
- `feature_cols`: list of features used

**Implementation notes**:
- Label: `team_a_wins` (binary).
- CV: for each season S in [2003..2025]:
  - Train on all seasons != S
  - Predict on season S
  - Compute: AUC, log-loss, Brier score, accuracy
- After CV, retrain on ALL data for the final model.
- No sample weights needed — each game is equally important.
- Handle NaN features: use LightGBM's native NaN handling (do NOT impute). Set `use_missing=True` (default).
- Print summary: mean AUC, mean log-loss across folds.

#### Function 5: `predict_final_four(model, team_df, final_four_teams, feature_cols, n_sims=10000) -> pd.DataFrame`

**Purpose**: Predict championship probabilities for the 4 Final Four teams by simulating the bracket.

**Input**:
- `model`: trained LGBMClassifier
- `team_df`: team-level features for the prediction year
- `final_four_teams`: list of 4 TeamIDs
- `feature_cols`: feature columns (without `diff_` prefix)
- `n_sims`: number of Monte Carlo simulations

**Output**: DataFrame with columns:
```
TeamID, team_name, seed,
p_win_semi, p_win_final, p_champion,
mkt_vwap (if available),
model_edge (= p_champion - mkt_implied, if market data exists)
```

**Implementation notes**:
- The 2026 Final Four bracket:
  - Semi 1: Arizona vs Connecticut (based on seeds/bracket position)
  - Semi 2: Illinois vs Michigan
  - Final: winner of Semi 1 vs winner of Semi 2
- For each simulation:
  1. Compute `diff_*` features for Semi 1 matchup. Get `p_a_wins` from model.
  2. Draw Bernoulli(p_a_wins) to determine Semi 1 winner.
  3. Repeat for Semi 2.
  4. Compute `diff_*` features for the Final matchup (using the two winners).
  5. Draw Bernoulli to determine champion.
- After `n_sims` simulations, `p_champion[team] = count(team won) / n_sims`.
- Also record `p_win_semi` and `p_win_final` (conditional on reaching final).
- **Bracket seeding for 2026**: Look up seeds from `MNCAATourneySeeds.csv`. The bracket pairs are determined by region: the two national semifinal matchups come from paired regions. For 2026, hardcode:
  ```python
  BRACKET_2026 = {
      "semi1": (TeamID_Arizona, TeamID_Connecticut),
      "semi2": (TeamID_Illinois, TeamID_Michigan),
  }
  ```
  Use `name_resolver` to get TeamIDs. If bracket is unknown, enumerate all 3 possible pairings and average.

#### Function 6: `blend_with_market(model_probs, market_df, market_weight=0.3) -> pd.DataFrame`

**Purpose**: Blend model predictions with Kalshi market implied probabilities.

**Input**:
- `model_probs`: output of `predict_final_four` (has `p_champion` column)
- `market_df`: market features with `mkt_vwap` column
- `market_weight`: weight for market signal (default 0.3)

**Output**: Same DataFrame with added columns:
```
mkt_implied (normalized VWAP),
blended_prob (weighted average of model and market),
model_edge (model - market, positive = model more bullish)
```

**Implementation notes**:
- Normalize market VWAP to sum to 1 across 4 teams.
- `blended = (1 - market_weight) * model_prob + market_weight * mkt_implied`
- Renormalize blended to sum to 1.
- The `model_edge` column flags where the model disagrees with the market — this is the actionable signal.

---

### File 2: `feature_pipeline/run_v2.py` (NEW — ~200 lines)

New orchestration script. Does NOT replace `run.py` — runs alongside it.

```python
"""
Pairwise game-level pipeline for NCAA tournament prediction.

Usage:
    python -m feature_pipeline.run_v2 [--data-dir data/] [--output-dir output_v2/]
"""

import argparse
from pathlib import Path
from feature_pipeline.game_model import (
    build_team_season_features,
    enrich_with_existing_features,
    build_game_pairs,
    train_game_model,
    predict_final_four,
    blend_with_market,
)
from feature_pipeline.data_loader import load_all
from feature_pipeline.feature_engineering import build_features
from feature_pipeline.market_features import load_kalshi_trades, compute_market_features
from feature_pipeline.name_resolver import build_id_lookup


def main(data_dir="data", output_dir="output_v2"):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    # ── Step 1: Build team-season features from Kaggle game data ──
    print("Step 1: Building team-season features from game data...")
    team_df = build_team_season_features(data_dir, min_season=2003)
    team_df.to_csv(output / "team_season_features.csv", index=False)
    print(f"  {len(team_df)} team-seasons built")

    # ── Step 2: Enrich with existing pipeline features ──
    print("Step 2: Enriching with team sheets, awards, rankings...")
    existing_df = load_all(data_dir, include_kaggle=True, include_team_stats=False, include_market=True)
    existing_df = build_features(existing_df, run_pca=False, run_reconcile=False, verbose=False)
    team_df = enrich_with_existing_features(team_df, existing_df)
    print(f"  {team_df.shape[1]} features per team-season")

    # ── Step 3: Build tournament game pairs ──
    print("Step 3: Building pairwise tournament game features...")
    tourney_path = f"{data_dir}/kaggle/MNCAATourneyCompactResults.csv"
    pairs_df = build_game_pairs(team_df, tourney_path, min_season=2003, max_season=2025)
    pairs_df.to_csv(output / "game_pairs.csv", index=False)
    print(f"  {len(pairs_df)} tournament games")

    # ── Step 4: Train pairwise model ──
    print("Step 4: Training pairwise game model...")
    results = train_game_model(pairs_df)
    results["cv_results"].to_csv(output / "cv_results.csv", index=False)
    results["oof_preds"].to_csv(output / "oof_predictions.csv", index=False)
    results["feature_importance"].to_csv(output / "feature_importance.csv", index=False)

    mean_auc = results["cv_results"]["auc"].mean()
    mean_ll = results["cv_results"]["log_loss"].mean()
    print(f"  CV AUC: {mean_auc:.4f}, CV Log-Loss: {mean_ll:.4f}")

    # ── Step 5: Predict 2026 Final Four ──
    print("Step 5: Predicting 2026 Final Four...")
    ff_2026 = ["Arizona", "Connecticut", "Illinois", "Michigan"]
    lookup = build_id_lookup(f"{data_dir}/kaggle")
    # Resolve names to TeamIDs
    ff_ids = []
    for name in ff_2026:
        from feature_pipeline.data_loader import normalise_team
        norm = normalise_team(name)
        # name_resolver uses normalised lookup
        tid = None
        for k, v in lookup.items():
            if norm.lower() in k.lower() or k.lower() in norm.lower():
                tid = v
                break
        ff_ids.append(tid)

    team_df_2026 = team_df[team_df["Season"] == 2026]
    preds = predict_final_four(
        results["model"], team_df_2026, ff_ids,
        results["feature_cols"], n_sims=50000
    )

    # ── Step 6: Blend with market ──
    print("Step 6: Blending with Kalshi market...")
    try:
        trades = load_kalshi_trades(data_dir)
        mkt = compute_market_features(trades)
        mkt_2026 = mkt[mkt["year"] == 2026]
        preds = blend_with_market(preds, mkt_2026, market_weight=0.3)
    except Exception as e:
        print(f"  Market blend skipped: {e}")

    preds.to_csv(output / "2026_predictions.csv", index=False)
    print("\n=== 2026 Championship Predictions ===")
    print(preds[["team_name", "seed", "p_champion", "blended_prob", "model_edge"]]
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output_v2")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
```

---

### File 3: `feature_pipeline/config.py` (EDIT — add new constants)

Add these constants alongside existing ones:

```python
# ── Game-level model config ──

GAME_MODEL_FEATURES = [
    # Kaggle regular season aggregates
    "kg_win_pct", "kg_fg_pct", "kg_fg3_pct", "kg_ft_pct", "kg_efg_pct",
    "kg_off_reb_pg", "kg_def_reb_pg", "kg_ast_pg", "kg_to_pg",
    "kg_stl_pg", "kg_blk_pg", "kg_scoring_margin", "kg_opp_fg_pct",
    "kg_margin_last5", "kg_margin_last10",
    # Massey ordinals
    "massey_POM", "massey_SAG", "massey_RPI", "massey_MOR",
    "massey_WLK", "massey_DOL", "massey_COL",
    # Seed
    "seed_num",
    # Derived
    "consensus_rank", "rank_spread",
]

# Features from existing pipeline to merge (available 2005+ only, NaN before)
ENRICHMENT_FEATURES = [
    "net_rank", "kpi", "sor", "bpi",
    "net_sos", "rpi_sos", "net_nc_sos",
    "overall_win_pct", "road_win_pct", "nc_win_pct", "conf_win_pct",
    "q1_win_pct", "resume_score",
    "win_entropy", "cusum_peak",
    "net_rank_yoy", "kpi_yoy", "sor_yoy",
    "total_awards",
]

# Market features (2025-2026 only)
ENRICHMENT_MARKET_FEATURES = [
    "mkt_vwap", "mkt_ofi", "mkt_trade_count", "mkt_volatility",
]

GAME_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": -1,
}

# 2026 Final Four bracket (hardcoded from actual bracket)
BRACKET_2026 = {
    "semi1": ("Arizona", "Connecticut"),
    "semi2": ("Illinois", "Michigan"),
}
```

---

## Data Flow Diagram

```
MRegularSeasonDetailedResults.csv ──┐
  (124K games, DayNum <= 132)       │
                                    ├──► build_team_season_features()
MMasseyOrdinals.csv ────────────────┤     → (Season, TeamID, 30+ features)
  (5.8M rows, DayNum <= 133)       │
                                    │
MNCAATourneySeeds.csv ──────────────┘
  (2.7K seed assignments)

                                          │
                                          ▼
                                    enrich_with_existing_features()
                                     merges team sheets, SOS, awards
                                          │
                                          ▼
MNCAATourneyCompactResults.csv ───► build_game_pairs()
  (2.6K tournament games)            → (Season, TeamA, TeamB, diff_*, label)
                                      ~1,400 rows (2003-2025)
                                          │
                                          ▼
                                    train_game_model()
                                     LightGBM, leave-one-year-out CV
                                     → model + CV metrics
                                          │
                                          ▼
                                    predict_final_four()
                                     Monte Carlo bracket simulation
                                     → P(champion) for each team
                                          │
                                          ▼
Kalshi trades ──────────────────►   blend_with_market()
  (2025-2026)                        → blended probs + model_edge
```

---

## Edge Cases and Gotchas

### 1. Team appears as both winner and loser
When computing regular season stats for TeamID X, you must:
- Pull rows where `WTeamID == X` (team won) — use W* columns for team stats, L* for opponent
- Pull rows where `LTeamID == X` (team lost) — use L* columns for team stats, W* for opponent
- Combine both sets of games for aggregation

```python
# Example for FG%
as_winner = games[games["WTeamID"] == tid][["WFGM", "WFGA"]]
as_loser = games[games["LTeamID"] == tid][["LFGM", "LFGA"]]
as_winner.columns = ["FGM", "FGA"]
as_loser.columns = ["FGM", "FGA"]
all_games = pd.concat([as_winner, as_loser])
fg_pct = all_games["FGM"].sum() / all_games["FGA"].sum()
```

### 2. Canonical pair ordering
Always `TeamA = min(id1, id2)`, `TeamB = max(id1, id2)`. This ensures:
- No duplicate pairs (A-vs-B and B-vs-A)
- The model learns symmetric differences (positive diff = TeamA is better)
- At prediction time, use the same ordering

### 3. DayNum cutoff varies slightly by year
The 2021 bubble tournament had shifted DayNums. For regular season features, `DayNum <= 132` is safe for all years. For Massey ordinals, use `RankingDayNum <= 133` (Selection Sunday rankings).

### 4. 2026 has no tournament results yet
`max_season=2025` in `build_game_pairs`. 2026 is prediction-only. Ensure `build_team_season_features` still computes features for 2026 teams (regular season data exists).

### 5. Missing features for early years
Team sheets only exist 2005+. Features like `net_rank`, `kpi`, `sor` will be NaN for 2003-2004 games. This is fine — LightGBM handles NaN natively. Do NOT impute.

### 6. Seed parsing
Seeds in `MNCAATourneySeeds.csv` look like `"W01"`, `"X16a"`, `"Z11b"`. Parse as:
```python
def parse_seed(seed_str):
    return int(seed_str[1:3])  # "W01" -> 1, "X16a" -> 16
```

### 7. No 2020
Skip everywhere. `years = [y for y in range(2003, 2026) if y != 2020]`.

---

## Validation Criteria

The implementation is correct when:

1. **`build_team_season_features`** produces ~1,500 rows with no tournament game data leaking in.
   - Verify: `assert team_df.groupby("Season").size().max() == 68` (ish — some years have play-in).
   - Verify: no feature computed from DayNum > 132.

2. **`build_game_pairs`** produces ~1,400 rows with balanced labels.
   - Verify: `assert 0.45 < pairs_df["team_a_wins"].mean() < 0.55` (should be ~0.50 due to canonical ordering).

3. **`train_game_model`** achieves reasonable CV metrics.
   - Expected: AUC 0.70-0.80 (higher-seeded team usually wins, but upsets happen).
   - Expected: log-loss < 0.65 (better than seed-only baseline of ~0.62).
   - If AUC < 0.60, something is wrong (features not joining correctly, label is random).

4. **`predict_final_four`** produces probabilities summing to 1.0 and varying meaningfully across teams.
   - Verify: `assert abs(preds["p_champion"].sum() - 1.0) < 0.01`.

5. **No tournament game features used**: The feature set must not include `kg_tourney_*` or any data derived from DayNum > 132.

---

## Testing Plan

### Unit tests (`tests/test_game_model.py`)

```python
def test_team_features_no_lookahead():
    """Verify no game with DayNum > 132 is used."""
    # Load raw games, filter to DayNum > 132, check they're excluded

def test_pair_ordering_canonical():
    """Verify TeamA < TeamB always."""
    pairs = build_game_pairs(...)
    assert (pairs["TeamA"] < pairs["TeamB"]).all()

def test_pair_labels_balanced():
    """Canonical ordering should give ~50/50 labels."""
    pairs = build_game_pairs(...)
    assert 0.40 < pairs["team_a_wins"].mean() < 0.60

def test_diff_features_antisymmetric():
    """diff(A,B) == -diff(B,A)."""
    # Pick a game, compute diffs both ways, verify

def test_no_2020():
    """Season 2020 must not appear anywhere."""
    team_df = build_team_season_features(...)
    assert 2020 not in team_df["Season"].values

def test_predict_sums_to_one():
    """Championship probs must sum to 1."""
    preds = predict_final_four(...)
    assert abs(preds["p_champion"].sum() - 1.0) < 0.01

def test_simulation_deterministic_with_seed():
    """Same random seed → same predictions."""
    p1 = predict_final_four(..., random_seed=42)
    p2 = predict_final_four(..., random_seed=42)
    assert p1["p_champion"].equals(p2["p_champion"])
```

---

## What NOT to Change

- `data_loader.py` — leave as-is. The new pipeline reads Kaggle CSVs directly for game-level data but still uses `load_all()` for enrichment features.
- `feature_engineering.py` — leave as-is. Called via `build_features()` for enrichment only.
- `feature_importance.py` — leave as-is. Can be called on the new model's features post-hoc.
- `model.py` — leave as-is. The old Final-Four-only pairwise model stays for comparison.
- `market_features.py` — leave as-is. Called for market enrichment.
- `name_resolver.py` — leave as-is. Used for TeamID resolution.
- `config.py` — only ADD new constants (see above). Don't modify existing ones.

---

## Optional Bayesian Extension (Phase 2, not in scope now)

Once the LightGBM pairwise model is working, a natural extension is:

1. **Bradley-Terry with Bayesian team strength**: Instead of feature diffs, model `P(A beats B) = sigma(strength_A - strength_B)` where `strength` is a latent parameter per team-season, estimated via MCMC (PyMC or numpyro).
2. **Hierarchical priors**: Team strength ~ Normal(conference_mean, tau). Conference means provide regularization for small-sample teams.
3. **Time-varying strength**: Allow strength to evolve within a season (random walk on DayNum).

This is more principled but harder to implement and slower to iterate on. Start with LightGBM diffs.
