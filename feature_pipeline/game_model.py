"""
game_model.py
-------------
Pairwise game-level model for NCAA tournament prediction (Phase 1A + 1B).

Reframes prediction as "who wins THIS game?" using all ~1,400 tournament
games (2003-2025) instead of only ~84 Final Four rows.

Functions:
  build_team_season_features  — one feature vector per (Season, TeamID)
  enrich_with_existing_features — merge team-sheet/award features
  build_game_pairs             — one row per tournament game, diff features
                                  (include_path=True adds path diff features)
  train_game_model             — LightGBM with leave-one-year-out CV
  predict_final_four           — Monte Carlo bracket simulation
                                  (include_path=True uses real E8 path data)
  blend_with_market            — blend model probs with Kalshi market
  compute_path_features        — cumulative in-tournament stats for one team
  load_actual_path_features    — load real E8 path features for a set of teams
"""

import os
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

from feature_pipeline.config import (
    GAME_MODEL_FEATURES,
    ENRICHMENT_FEATURES,
    ENRICHMENT_MARKET_FEATURES,
    GAME_LGBM_PARAMS,
    MASSEY_SYSTEMS,
    PATH_FEATURES,
)
from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id, build_teams_df


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

SKIP_SEASON = 2020
VALID_SEASONS = [y for y in range(2003, 2027) if y != SKIP_SEASON]


def parse_seed(seed_str: str) -> int:
    """Parse NCAA seed string to integer. 'W01' → 1, 'X16a' → 16."""
    if not isinstance(seed_str, str) or len(seed_str) < 3:
        return np.nan
    try:
        return int(seed_str[1:3])
    except (ValueError, TypeError):
        return np.nan


def daynum_to_round(daynum: int) -> int:
    """Map DayNum to tournament round number."""
    if daynum <= 136:
        return 1   # First Four / R64
    elif daynum <= 138:
        return 2   # R32
    elif daynum <= 145:
        return 3   # Sweet 16
    elif daynum <= 148:
        return 4   # Elite 8
    elif daynum <= 152:
        return 5   # Final Four
    else:
        return 6   # Championship


# ─────────────────────────────────────────────────────────────────────────────
#  Opponent rank lookup for SOS-stratified features
# ─────────────────────────────────────────────────────────────────────────────

def _add_opponent_ranks(
    games: pd.DataFrame,
    massey: pd.DataFrame,
    systems: list[str],
    max_daynum: int = 132,
) -> pd.DataFrame:
    """
    For each game row, look up the opponent's most recent rank (at or before
    that game's DayNum) from MMasseyOrdinals, for each requested system.
    Adds columns opp_{sys.lower()}_rank.

    Uses pd.merge_asof per season — O(n log n), safe for ~300K rows.
    Opponent ranks are NaN when the system has no data for that era (e.g.
    NET before 2019).
    """
    result_seasons = []
    for season, sg in games.groupby("Season"):
        sg = sg.sort_values("DayNum").copy()
        for sys in systems:
            col = f"opp_{sys.lower()}_rank"
            sys_ranks = (
                massey[
                    (massey["Season"] == season)
                    & (massey["SystemName"] == sys)
                    & (massey["RankingDayNum"] <= max_daynum)
                ][["TeamID", "RankingDayNum", "OrdinalRank"]]
                .rename(columns={
                    "TeamID": "OpponentTeamID",
                    "RankingDayNum": "DayNum",
                    "OrdinalRank": col,
                })
                .sort_values("DayNum")
            )
            if sys_ranks.empty:
                sg[col] = np.nan
            else:
                sg = pd.merge_asof(
                    sg, sys_ranks,
                    on="DayNum", by="OpponentTeamID",
                    direction="backward",
                )
        result_seasons.append(sg)
    return pd.concat(result_seasons, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Function 1: build_team_season_features
# ─────────────────────────────────────────────────────────────────────────────

def build_team_season_features(data_dir: str, min_season: int = 2003) -> pd.DataFrame:
    """
    Build one feature vector per (Season, TeamID) for all tournament teams,
    using ONLY pre-tournament data (DayNum <= 132).

    Reads Kaggle CSVs directly (does not call load_all or build_features).

    Returns DataFrame with columns:
        Season, TeamID, kg_wins, kg_losses, kg_win_pct, kg_fg_pct, ...,
        massey_POM, ..., seed_num, consensus_rank, rank_spread
    """
    kaggle_dir = os.path.join(data_dir, "kaggle")

    # ── Regular season detailed results ──────────────────────────────────────
    rs_path = os.path.join(kaggle_dir, "MRegularSeasonDetailedResults.csv")
    rs = pd.read_csv(rs_path)
    rs = rs[rs["DayNum"] <= 132]
    rs = rs[rs["Season"] >= min_season]
    rs = rs[rs["Season"] != SKIP_SEASON]

    # Build per-team stats by combining win rows and loss rows
    # Win rows: team is winner
    w_cols = {
        "Season": "Season", "DayNum": "DayNum",
        "WTeamID": "TeamID", "WScore": "Pts", "LScore": "OppPts",
        "WFGM": "FGM", "WFGA": "FGA",
        "WFGM3": "FGM3", "WFGA3": "FGA3",
        "WFTM": "FTM", "WFTA": "FTA",
        "WOR": "OReb", "WDR": "DReb",
        "WAst": "Ast", "WTO": "TO",
        "WStl": "Stl", "WBlk": "Blk",
        "LFGM": "OppFGM", "LFGA": "OppFGA",
        "NumOT": "NumOT",
        "_won": "_won",
    }
    w_rows = rs.copy()
    w_rows["_won"] = 1
    w_rows = w_rows.rename(columns={
        "WTeamID": "TeamID", "WScore": "Pts", "LScore": "OppPts",
        "WFGM": "FGM", "WFGA": "FGA",
        "WFGM3": "FGM3", "WFGA3": "FGA3",
        "WFTM": "FTM", "WFTA": "FTA",
        "WOR": "OReb", "WDR": "DReb",
        "WAst": "Ast", "WTO": "TO",
        "WStl": "Stl", "WBlk": "Blk",
        "LFGM": "OppFGM", "LFGA": "OppFGA",
        "LOR": "OppOReb", "LDR": "OppDReb", "LTO": "OppTO",
        "LTeamID": "OpponentTeamID",
    })

    # Loss rows: team is loser
    l_rows = rs.copy()
    l_rows["_won"] = 0
    l_rows = l_rows.rename(columns={
        "LTeamID": "TeamID", "LScore": "Pts", "WScore": "OppPts",
        "LFGM": "FGM", "LFGA": "FGA",
        "LFGM3": "FGM3", "LFGA3": "FGA3",
        "LFTM": "FTM", "LFTA": "FTA",
        "LOR": "OReb", "LDR": "DReb",
        "LAst": "Ast", "LTO": "TO",
        "LStl": "Stl", "LBlk": "Blk",
        "WFGM": "OppFGM", "WFGA": "OppFGA",
        "WOR": "OppOReb", "WDR": "OppDReb", "WTO": "OppTO",
        "WTeamID": "OpponentTeamID",
    })

    needed_cols = ["Season", "DayNum", "TeamID", "OpponentTeamID", "Pts", "OppPts",
                   "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                   "OReb", "DReb", "Ast", "TO", "Stl", "Blk",
                   "OppFGM", "OppFGA", "OppOReb", "OppDReb", "OppTO",
                   "NumOT", "_won"]
    games = pd.concat(
        [w_rows[needed_cols], l_rows[needed_cols]],
        ignore_index=True
    )

    # ── Load Massey once (reused for opponent ranks + team-level features) ────
    massey_path = os.path.join(kaggle_dir, "MMasseyOrdinals.csv")
    massey_raw = pd.read_csv(massey_path)
    massey_raw = massey_raw[massey_raw["Season"] >= min_season]
    massey_raw = massey_raw[massey_raw["Season"] != SKIP_SEASON]

    # Add per-game opponent NET and POM ranks for SOS-stratified features.
    # NET is NaN pre-2019 (not in Massey that era) — agg_team handles NaN gracefully.
    games = _add_opponent_ranks(games, massey_raw, systems=["NET", "POM"])
    games["opp_avg_rank"] = games[["opp_net_rank", "opp_pom_rank"]].mean(axis=1)

    # ── Per-team aggregates ───────────────────────────────────────────────────
    STRONG_THRESH = 100  # top-100 opponent = "strong"

    def agg_team(g):
        n = len(g)
        wins = g["_won"].sum()
        margin = (g["Pts"] - g["OppPts"]).mean()

        # FG%
        fg_pct = g["FGM"].sum() / g["FGA"].sum() if g["FGA"].sum() > 0 else np.nan
        fg3_pct = g["FGM3"].sum() / g["FGA3"].sum() if g["FGA3"].sum() > 0 else np.nan
        ft_pct = g["FTM"].sum() / g["FTA"].sum() if g["FTA"].sum() > 0 else np.nan
        efg_pct = ((g["FGM"].sum() + 0.5 * g["FGM3"].sum()) / g["FGA"].sum()
                   if g["FGA"].sum() > 0 else np.nan)
        opp_fg_pct = (g["OppFGM"].sum() / g["OppFGA"].sum()
                      if g["OppFGA"].sum() > 0 else np.nan)

        # Per-game stats
        off_reb_pg = g["OReb"].mean()
        def_reb_pg = g["DReb"].mean()
        total_reb_pg = off_reb_pg + def_reb_pg
        ast_pg = g["Ast"].mean()
        to_pg = g["TO"].mean()
        stl_pg = g["Stl"].mean()
        blk_pg = g["Blk"].mean()

        # Margin stats (require opponent columns)
        off_reb_margin = (g["OReb"] - g["OppOReb"]).mean()
        def_reb_margin = (g["DReb"] - g["OppDReb"]).mean()
        total_reb_margin = (g["OReb"] + g["DReb"] - g["OppOReb"] - g["OppDReb"]).mean()
        to_margin = (g["OppTO"] - g["TO"]).mean()  # positive = we force more TOs than we give

        # ── Recent form (last N games sorted by DayNum) ───────────────────────
        sorted_g = g.sort_values("DayNum", ascending=False)

        def last_n_margin(n_games):
            last = sorted_g.head(n_games)
            return (last["Pts"] - last["OppPts"]).mean() if len(last) > 0 else np.nan

        def last_n_fg_pct(n_games):
            last = sorted_g.head(n_games)
            return last["FGM"].sum() / last["FGA"].sum() if last["FGA"].sum() > 0 else np.nan

        def last_n_to_margin(n_games):
            last = sorted_g.head(n_games)
            return (last["OppTO"] - last["TO"]).mean() if len(last) > 0 else np.nan

        def last_n_reb_margin(n_games):
            last = sorted_g.head(n_games)
            return (last["OReb"] + last["DReb"] - last["OppOReb"] - last["OppDReb"]).mean() \
                if len(last) > 0 else np.nan

        margin_last5  = last_n_margin(5)
        margin_last10 = last_n_margin(10)
        fg_pct_last5  = last_n_fg_pct(5)
        to_margin_last5  = last_n_to_margin(5)
        reb_margin_last5 = last_n_reb_margin(5)

        # ── Family 2: SOS-stratified features ────────────────────────────────
        # For each ranking system (NET, POM, their average), split games into
        # strong (rank ≤ 100) vs rest and compare performance.
        # margin_dropoff > 0 means the team performs worse against good opponents.
        def strong_opp_feats(rank_col: str, prefix: str) -> dict:
            ranks = g[rank_col] if rank_col in g.columns else pd.Series(dtype=float)
            strong = g[ranks.notna() & (ranks <= STRONG_THRESH)]
            if len(strong) == 0:
                return {
                    f"{prefix}_strong_win_pct":    np.nan,
                    f"{prefix}_strong_margin":     np.nan,
                    f"{prefix}_margin_dropoff":    np.nan,
                    f"{prefix}_strong_to_margin":  np.nan,
                    f"{prefix}_strong_reb_margin": np.nan,
                }
            s_margin = (strong["Pts"] - strong["OppPts"]).mean()
            return {
                f"{prefix}_strong_win_pct":    strong["_won"].mean(),
                f"{prefix}_strong_margin":     s_margin,
                f"{prefix}_margin_dropoff":    margin - s_margin,  # >0 = drops off vs good teams
                f"{prefix}_strong_to_margin":  (strong["OppTO"] - strong["TO"]).mean(),
                f"{prefix}_strong_reb_margin": (
                    strong["OReb"] + strong["DReb"]
                    - strong["OppOReb"] - strong["OppDReb"]
                ).mean(),
            }

        sos_feats = {}
        sos_feats.update(strong_opp_feats("opp_net_rank", "kg_net"))
        sos_feats.update(strong_opp_feats("opp_pom_rank", "kg_pom"))
        sos_feats.update(strong_opp_feats("opp_avg_rank", "kg_sos"))

        return pd.Series({
            "kg_wins": wins,
            "kg_losses": n - wins,
            "kg_win_pct": wins / n if n > 0 else np.nan,
            "kg_fg_pct": fg_pct,
            "kg_fg3_pct": fg3_pct,
            "kg_ft_pct": ft_pct,
            "kg_efg_pct": efg_pct,
            "kg_off_reb_pg": off_reb_pg,
            "kg_def_reb_pg": def_reb_pg,
            "kg_total_reb_pg": total_reb_pg,
            "kg_off_reb_margin": off_reb_margin,
            "kg_def_reb_margin": def_reb_margin,
            "kg_total_reb_margin": total_reb_margin,
            "kg_ast_pg": ast_pg,
            "kg_to_pg": to_pg,
            "kg_to_margin": to_margin,
            "kg_stl_pg": stl_pg,
            "kg_blk_pg": blk_pg,
            "kg_scoring_margin": margin,
            "kg_opp_fg_pct": opp_fg_pct,
            "kg_margin_last5":  margin_last5,
            "kg_margin_last10": margin_last10,
            # Family 1: delta (recent form vs season average)
            "kg_margin_last5_delta":     margin_last5  - margin,
            "kg_margin_last10_delta":    margin_last10 - margin,
            "kg_fg_pct_last5_delta":     fg_pct_last5  - fg_pct,
            "kg_to_margin_last5_delta":  to_margin_last5  - to_margin,
            "kg_reb_margin_last5_delta": reb_margin_last5 - total_reb_margin,
            # Family 2: SOS-stratified
            **sos_feats,
        })

    team_stats = games.groupby(["Season", "TeamID"]).apply(agg_team).reset_index()

    # ── Massey ordinals (reuse massey_raw already loaded above) ──────────────
    massey = massey_raw[massey_raw["RankingDayNum"] <= 133].copy()
    massey = massey[massey["SystemName"].isin(MASSEY_SYSTEMS)]

    # For each (Season, TeamID, SystemName): take row with max RankingDayNum
    massey_latest = (
        massey.sort_values("RankingDayNum")
        .groupby(["Season", "TeamID", "SystemName"])
        .last()
        .reset_index()
    )[["Season", "TeamID", "SystemName", "OrdinalRank"]]

    # Pivot: each system becomes a column
    massey_pivot = massey_latest.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="OrdinalRank",
        aggfunc="last",
    ).reset_index()
    massey_pivot.columns.name = None

    # Rename to massey_{system}
    rename_map = {sys: f"massey_{sys}" for sys in MASSEY_SYSTEMS}
    massey_pivot = massey_pivot.rename(columns=rename_map)

    # Derived: consensus_rank (mean of available systems), rank_spread (std)
    massey_cols = [f"massey_{s}" for s in MASSEY_SYSTEMS if f"massey_{s}" in massey_pivot.columns]
    massey_pivot["consensus_rank"] = massey_pivot[massey_cols].mean(axis=1)
    massey_pivot["rank_spread"] = massey_pivot[massey_cols].std(axis=1)

    # massey_PMW: average of POM, MOR, WLK — three independent model-based
    # predictive systems (excludes formula-based RPI and SOS metrics).
    # Captures "computer model consensus" as a distinct signal from full consensus_rank.
    pmw_cols = [f"massey_{s}" for s in ["POM", "MOR", "WLK"] if f"massey_{s}" in massey_pivot.columns]
    massey_pivot["massey_PMW"] = massey_pivot[pmw_cols].mean(axis=1)

    # ── Seeds ─────────────────────────────────────────────────────────────────
    seeds_path = os.path.join(kaggle_dir, "MNCAATourneySeeds.csv")
    seeds = pd.read_csv(seeds_path)
    seeds = seeds[seeds["Season"] >= min_season]
    seeds = seeds[seeds["Season"] != SKIP_SEASON]
    seeds["seed_num"] = seeds["Seed"].apply(parse_seed)
    seeds = seeds[["Season", "TeamID", "seed_num"]]

    # ── Join everything ───────────────────────────────────────────────────────
    # Start from tournament seeds (defines which teams are in scope)
    df = seeds.copy()
    df = df.merge(team_stats, on=["Season", "TeamID"], how="left")
    df = df.merge(massey_pivot, on=["Season", "TeamID"], how="left")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Function 2: enrich_with_existing_features
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_existing_features(team_df: pd.DataFrame,
                                  existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Kaggle-derived team features with features from the existing pipeline
    (team sheets, awards, Massey from team sheets, market data).

    team_df:    output of build_team_season_features — has Season, TeamID columns
    existing_df: output of load_all() |> build_features() — has year, team columns

    Returns team_df enriched with ENRICHMENT_FEATURES and ENRICHMENT_MARKET_FEATURES.
    """
    from feature_pipeline.config import TEAM_NAME_MAP

    kaggle_dir = None
    # Infer kaggle_dir from existing data — we need it for build_id_lookup
    # Try common paths
    for candidate in ["data/kaggle", "../data/kaggle", "../../data/kaggle"]:
        if os.path.exists(os.path.join(candidate, "MTeams.csv")):
            kaggle_dir = candidate
            break
    if kaggle_dir is None:
        warnings.warn("Could not find kaggle data directory for name resolution; enrichment skipped.")
        return team_df

    lookup = build_id_lookup(kaggle_dir)

    # Resolve existing_df team names → TeamID
    enrich = existing_df.copy()
    enrich["TeamID"] = enrich["team"].apply(
        lambda n: resolve_team_id(n, lookup, TEAM_NAME_MAP)
    )
    enrich = enrich.rename(columns={"year": "Season"})
    enrich = enrich.dropna(subset=["TeamID"])
    enrich["TeamID"] = enrich["TeamID"].astype(int)
    enrich["Season"] = enrich["Season"].astype(int)

    # Select enrichment columns that exist in existing_df
    all_enrich_cols = ENRICHMENT_FEATURES + ENRICHMENT_MARKET_FEATURES
    available_cols = [c for c in all_enrich_cols if c in enrich.columns]
    merge_df = enrich[["Season", "TeamID"] + available_cols].copy()

    # Drop duplicate (Season, TeamID) rows — keep last (most recent data)
    merge_df = merge_df.drop_duplicates(subset=["Season", "TeamID"], keep="last")

    result = team_df.merge(merge_df, on=["Season", "TeamID"], how="left")
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Function 3: build_game_pairs
# ─────────────────────────────────────────────────────────────────────────────

def build_game_pairs(team_df: pd.DataFrame,
                     tourney_results_path: str,
                     min_season: int = 2003,
                     max_season: int = 2025,
                     include_path: bool = False,
                     tourney_detailed_path: str = None) -> pd.DataFrame:
    """
    Create one row per tournament game with difference features.

    team_df: output of build_team_season_features (or enriched version)
    tourney_results_path: path to MNCAATourneyCompactResults.csv
    include_path: if True, add diff_path_* columns from compute_path_features
    tourney_detailed_path: path to MNCAATourneyDetailedResults.csv (required if include_path=True)

    Returns DataFrame with: Season, DayNum, TeamA, TeamB,
        diff_*, seed_matchup, round_num, team_a_wins [, diff_path_*]
    """
    results = pd.read_csv(tourney_results_path)
    results = results[
        (results["Season"] >= min_season) &
        (results["Season"] <= max_season) &
        (results["Season"] != SKIP_SEASON)
    ]

    # Build a lookup: (Season, TeamID) → feature dict
    feature_cols = [c for c in team_df.columns if c not in ("Season", "TeamID")]
    team_df_indexed = team_df.set_index(["Season", "TeamID"])

    # ── Path feature setup ────────────────────────────────────────────────────
    # Pre-build per-team tournament game history indexed by (Season, TeamID)
    tourn_detailed = None
    seeds_df_path = None
    _team_tourn_games: dict = {}  # (Season, TeamID) → list of game rows, sorted by DayNum

    if include_path:
        if tourney_detailed_path and os.path.exists(tourney_detailed_path):
            tourn_detailed = pd.read_csv(tourney_detailed_path)
            tourn_detailed = tourn_detailed[
                (tourn_detailed["Season"] >= min_season) &
                (tourn_detailed["Season"] <= max_season) &
                (tourn_detailed["Season"] != SKIP_SEASON)
            ]
        else:
            warnings.warn("tourney_detailed_path not found; include_path disabled.")
            include_path = False

        if include_path:
            # Load seeds for opponent seed lookup
            seeds_csv = os.path.join(os.path.dirname(tourney_results_path),
                                     "MNCAATourneySeeds.csv")
            if os.path.exists(seeds_csv):
                seeds_df_path = pd.read_csv(seeds_csv)
                seeds_df_path["seed_num"] = seeds_df_path["Seed"].apply(parse_seed)
                seeds_df_path = seeds_df_path[["Season", "TeamID", "seed_num"]]

            # Build lookup: (Season, TeamID) → sorted list of game dicts
            for _, g in tourn_detailed.iterrows():
                for tid, is_winner in [(int(g["WTeamID"]), True), (int(g["LTeamID"]), False)]:
                    key = (int(g["Season"]), tid)
                    if key not in _team_tourn_games:
                        _team_tourn_games[key] = []
                    _team_tourn_games[key].append({
                        "DayNum": int(g["DayNum"]),
                        "won": int(is_winner),
                        "margin": (g["WScore"] - g["LScore"]) * (1 if is_winner else -1),
                        "FGM": g["WFGM"] if is_winner else g["LFGM"],
                        "FGA": g["WFGA"] if is_winner else g["LFGA"],
                        "OppFGM": g["LFGM"] if is_winner else g["WFGM"],
                        "OppFGA": g["LFGA"] if is_winner else g["WFGA"],
                        "NumOT": g["NumOT"],
                        "opp_team_id": int(g["LTeamID"]) if is_winner else int(g["WTeamID"]),
                    })
            # Sort each team's games by DayNum
            for key in _team_tourn_games:
                _team_tourn_games[key].sort(key=lambda x: x["DayNum"])

    rows = []
    for _, game in results.iterrows():
        season = int(game["Season"])
        daynum = int(game["DayNum"])
        wteam = int(game["WTeamID"])
        lteam = int(game["LTeamID"])

        # Canonical ordering: TeamA = min(id), TeamB = max(id)
        team_a = min(wteam, lteam)
        team_b = max(wteam, lteam)
        team_a_wins = 1 if team_a == wteam else 0

        # Look up features
        key_a = (season, team_a)
        key_b = (season, team_b)
        if key_a not in team_df_indexed.index or key_b not in team_df_indexed.index:
            continue

        feat_a = team_df_indexed.loc[key_a]
        feat_b = team_df_indexed.loc[key_b]

        row = {
            "Season": season,
            "DayNum": daynum,
            "TeamA": team_a,
            "TeamB": team_b,
            "team_a_wins": team_a_wins,
            "round_num": daynum_to_round(daynum),
        }

        # Diff features
        for col in feature_cols:
            va = feat_a[col] if col in feat_a.index else np.nan
            vb = feat_b[col] if col in feat_b.index else np.nan
            try:
                row[f"diff_{col}"] = float(va) - float(vb)
            except (TypeError, ValueError):
                row[f"diff_{col}"] = np.nan

        # Seed matchup string
        seed_a = feat_a.get("seed_num", np.nan) if hasattr(feat_a, "get") else feat_a["seed_num"] if "seed_num" in feat_a.index else np.nan
        seed_b = feat_b.get("seed_num", np.nan) if hasattr(feat_b, "get") else feat_b["seed_num"] if "seed_num" in feat_b.index else np.nan
        try:
            s_lo = int(min(seed_a, seed_b))
            s_hi = int(max(seed_a, seed_b))
            row["seed_matchup"] = f"{s_lo}v{s_hi}"
        except (TypeError, ValueError):
            row["seed_matchup"] = "unknown"

        # ── Path features ─────────────────────────────────────────────────────
        if include_path:
            prior_a = [g for g in _team_tourn_games.get(key_a, []) if g["DayNum"] < daynum]
            prior_b = [g for g in _team_tourn_games.get(key_b, []) if g["DayNum"] < daynum]
            pf_a = compute_path_features(team_a, season, prior_a, seeds_df_path)
            pf_b = compute_path_features(team_b, season, prior_b, seeds_df_path)
            for feat in PATH_FEATURES:
                va = pf_a.get(feat, np.nan)
                vb = pf_b.get(feat, np.nan)
                try:
                    row[f"diff_{feat}"] = float(va) - float(vb)
                except (TypeError, ValueError):
                    row[f"diff_{feat}"] = np.nan

        rows.append(row)

    pairs_df = pd.DataFrame(rows)
    return pairs_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Function 4: train_game_model
# ─────────────────────────────────────────────────────────────────────────────

def train_game_model(pairs_df: pd.DataFrame,
                     feature_cols: list = None,
                     params: dict = None) -> dict:
    """
    Train LightGBM on pairwise game data with leave-one-year-out CV.

    Returns dict with:
        model, cv_results, oof_preds, feature_importance, feature_cols
    """
    if params is None:
        params = GAME_LGBM_PARAMS.copy()

    if feature_cols is None:
        feature_cols = sorted([c for c in pairs_df.columns if c.startswith("diff_")])

    label_col = "team_a_wins"
    seasons = sorted(pairs_df["Season"].unique())

    cv_rows = []
    oof_rows = []

    print(f"  Training on {len(feature_cols)} features, {len(seasons)} CV folds, {len(pairs_df)} games")

    for season in seasons:
        train_mask = pairs_df["Season"] != season
        val_mask = pairs_df["Season"] == season

        X_train = pairs_df.loc[train_mask, feature_cols]
        y_train = pairs_df.loc[train_mask, label_col]
        X_val = pairs_df.loc[val_mask, feature_cols]
        y_val = pairs_df.loc[val_mask, label_col]

        if len(X_val) == 0:
            continue

        mdl = LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(X_train, y_train)

        proba = mdl.predict_proba(X_val)[:, 1]
        pred_class = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, proba) if len(y_val.unique()) > 1 else np.nan
        ll = log_loss(y_val, proba)
        brier = brier_score_loss(y_val, proba)
        acc = accuracy_score(y_val, pred_class)

        cv_rows.append({
            "season": season,
            "auc": auc,
            "log_loss": ll,
            "brier": brier,
            "accuracy": acc,
            "n_games": int(val_mask.sum()),
        })

        val_idx = pairs_df.index[val_mask]
        for i, idx in enumerate(val_idx):
            oof_rows.append({
                "index": idx,
                "Season": season,
                "TeamA": pairs_df.loc[idx, "TeamA"],
                "TeamB": pairs_df.loc[idx, "TeamB"],
                "team_a_wins": pairs_df.loc[idx, label_col],
                "pred_prob": proba[i],
            })

    cv_results = pd.DataFrame(cv_rows)

    # Final model trained on ALL data
    X_all = pairs_df[feature_cols]
    y_all = pairs_df[label_col]
    final_model = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_all, y_all)

    # Feature importance (gain-based)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    mean_auc = cv_results["auc"].mean()
    mean_ll = cv_results["log_loss"].mean()
    print(f"  CV AUC: {mean_auc:.4f}  CV Log-Loss: {mean_ll:.4f}")

    return {
        "model": final_model,
        "cv_results": cv_results,
        "oof_preds": pd.DataFrame(oof_rows),
        "feature_importance": importance,
        "feature_cols": feature_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Function 5: predict_final_four
# ─────────────────────────────────────────────────────────────────────────────

def predict_final_four(model,
                       team_df: pd.DataFrame,
                       final_four_teams: list,
                       feature_cols: list,
                       n_sims: int = 10000,
                       random_seed: int = None,
                       include_path: bool = False,
                       actual_path_features: dict = None) -> pd.DataFrame:
    """
    Predict championship probabilities for 4 Final Four teams via Monte Carlo.

    final_four_teams: list of 4 TeamIDs [semi1_teamA, semi1_teamB, semi2_teamA, semi2_teamB]
        Bracket: semi1 = [0] vs [1], semi2 = [2] vs [3], final = winners

    include_path: if True, incorporate path features into matchup predictions.
    actual_path_features: dict {TeamID: path_feature_dict} from compute_path_features
        using REAL tournament results through E8.  Required when include_path=True.

    Returns DataFrame with: TeamID, team_name, seed, p_win_semi, p_win_final, p_champion
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    if len(final_four_teams) != 4:
        raise ValueError(f"Expected 4 teams, got {len(final_four_teams)}")

    if include_path and actual_path_features is None:
        warnings.warn("include_path=True but actual_path_features is None; disabling path features.")
        include_path = False

    # feature_cols are diff_* columns; base cols are without diff_ prefix
    base_cols = [c.replace("diff_", "", 1) for c in feature_cols]
    path_feat_set = set(PATH_FEATURES)

    # Build team feature lookup for prediction season
    pred_season = team_df["Season"].max()
    season_df = team_df[team_df["Season"] == pred_season].set_index("TeamID")

    def get_team_features(tid):
        if tid in season_df.index:
            return season_df.loc[tid]
        return pd.Series({c: np.nan for c in base_cols})

    def compute_diff_features(tid_a, tid_b, path_a: dict = None, path_b: dict = None):
        """Build diff feature vector with canonical ordering (tid_a < tid_b assumed)."""
        fa = get_team_features(tid_a)
        fb = get_team_features(tid_b)
        diffs = []
        for bc in base_cols:
            if include_path and bc in path_feat_set:
                # Use path feature override when provided
                va = path_a.get(bc, np.nan) if path_a else np.nan
                vb = path_b.get(bc, np.nan) if path_b else np.nan
            else:
                try:
                    va = float(fa.get(bc, np.nan)) if hasattr(fa, "get") else float(fa[bc]) if bc in fa.index else np.nan
                    vb = float(fb.get(bc, np.nan)) if hasattr(fb, "get") else float(fb[bc]) if bc in fb.index else np.nan
                except (TypeError, ValueError):
                    va, vb = np.nan, np.nan
            try:
                diffs.append(float(va) - float(vb))
            except (TypeError, ValueError):
                diffs.append(np.nan)
        return np.array(diffs, dtype=float).reshape(1, -1)

    def get_win_prob(tid_a, tid_b, path_a: dict = None, path_b: dict = None):
        """Return P(tid_a beats tid_b) using canonical ordering."""
        if tid_a < tid_b:
            X = compute_diff_features(tid_a, tid_b, path_a, path_b)
        else:
            X = compute_diff_features(tid_b, tid_a, path_b, path_a)
        p = float(model.predict_proba(pd.DataFrame(X, columns=feature_cols))[:, 1][0])
        return p if tid_a < tid_b else 1.0 - p

    t0, t1, t2, t3 = final_four_teams  # semi1: t0 vs t1, semi2: t2 vs t3

    # Pre-compute semi win probs (use actual path features — deterministic)
    pf = actual_path_features or {}
    p_t0_beats_t1 = get_win_prob(t0, t1, pf.get(t0), pf.get(t1))
    p_t2_beats_t3 = get_win_prob(t2, t3, pf.get(t2), pf.get(t3))

    # For Phase 1A (no path): pre-compute all 4 possible final probs too
    if not include_path:
        final_probs = {}
        for w1 in [t0, t1]:
            for w2 in [t2, t3]:
                final_probs[(w1, w2)] = get_win_prob(w1, w2)

    # Seed lookup for path feature updates
    _seed_lookup = {}
    if include_path:
        for tid in final_four_teams:
            row = season_df.loc[tid] if tid in season_df.index else None
            if row is not None:
                try:
                    _seed_lookup[tid] = float(row["seed_num"] if "seed_num" in row.index else np.nan)
                except (TypeError, ValueError):
                    _seed_lookup[tid] = np.nan

    # Monte Carlo simulation
    win_semi = {t: 0 for t in final_four_teams}
    win_final = {t: 0 for t in final_four_teams}
    win_champ = {t: 0 for t in final_four_teams}

    draws_semi1 = rng.random(n_sims)
    draws_semi2 = rng.random(n_sims)
    draws_final = rng.random(n_sims)

    for i in range(n_sims):
        w1 = t0 if draws_semi1[i] < p_t0_beats_t1 else t1
        l1 = t1 if w1 == t0 else t0
        w2 = t2 if draws_semi2[i] < p_t2_beats_t3 else t3
        l2 = t3 if w2 == t2 else t2

        win_semi[w1] += 1
        win_semi[w2] += 1

        if include_path:
            # Update path features for finalists with simulated semi results
            sim_margin_1 = 20.0 * (p_t0_beats_t1 - 0.5) if w1 == t0 else 20.0 * ((1 - p_t0_beats_t1) - 0.5)
            sim_margin_2 = 20.0 * (p_t2_beats_t3 - 0.5) if w2 == t2 else 20.0 * ((1 - p_t2_beats_t3) - 0.5)
            pf_w1 = _update_path_features(pf.get(w1, {}), sim_margin_1, _seed_lookup.get(l1, np.nan))
            pf_w2 = _update_path_features(pf.get(w2, {}), sim_margin_2, _seed_lookup.get(l2, np.nan))
            p_w1_beats_w2 = get_win_prob(w1, w2, pf_w1, pf_w2)
        else:
            p_w1_beats_w2 = final_probs[(w1, w2)]

        champ = w1 if draws_final[i] < p_w1_beats_w2 else w2

        win_final[w1] += 1
        win_final[w2] += 1
        win_champ[champ] += 1

    # Resolve team names
    kaggle_dir = None
    for candidate in ["data/kaggle", "../data/kaggle"]:
        if os.path.exists(os.path.join(candidate, "MTeams.csv")):
            kaggle_dir = candidate
            break

    teams_df = build_teams_df(kaggle_dir) if kaggle_dir else pd.DataFrame()

    def get_name(tid):
        if len(teams_df) == 0:
            return str(tid)
        row = teams_df[teams_df["TeamID"] == tid]
        return row.iloc[0]["TeamName"] if len(row) > 0 else str(tid)

    def get_seed(tid):
        row = season_df.loc[tid] if tid in season_df.index else None
        if row is None:
            return np.nan
        return row.get("seed_num", np.nan) if hasattr(row, "get") else (
            row["seed_num"] if "seed_num" in row.index else np.nan
        )

    records = []
    for tid in final_four_teams:
        records.append({
            "TeamID": tid,
            "team_name": get_name(tid),
            "seed": get_seed(tid),
            "p_win_semi": win_semi[tid] / n_sims,
            "p_win_final": win_final[tid] / n_sims,
            "p_champion": win_champ[tid] / n_sims,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  Function 6: blend_with_market
# ─────────────────────────────────────────────────────────────────────────────

def blend_with_market(model_probs: pd.DataFrame,
                      market_df: pd.DataFrame,
                      market_weight: float = 0.3) -> pd.DataFrame:
    """
    Blend model predictions with Kalshi market implied probabilities.

    model_probs: output of predict_final_four — has TeamID, p_champion
    market_df:   market features with mkt_vwap column and TeamID or team column

    Returns model_probs with added columns:
        mkt_implied, blended_prob, model_edge
    """
    result = model_probs.copy()

    # Resolve market TeamIDs if needed
    mkt = market_df.copy()
    if "TeamID" not in mkt.columns and "team" in mkt.columns:
        from feature_pipeline.config import TEAM_NAME_MAP
        kaggle_dir = None
        for candidate in ["data/kaggle", "../data/kaggle"]:
            if os.path.exists(os.path.join(candidate, "MTeams.csv")):
                kaggle_dir = candidate
                break
        if kaggle_dir:
            lookup = build_id_lookup(kaggle_dir)
            mkt["TeamID"] = mkt["team"].apply(
                lambda n: resolve_team_id(n, lookup, TEAM_NAME_MAP)
            )

    # Normalize VWAP to sum to 1 across the 4 teams
    ff_team_ids = set(result["TeamID"].tolist())
    mkt_ff = mkt[mkt["TeamID"].isin(ff_team_ids)].copy()

    if "mkt_vwap" not in mkt_ff.columns or len(mkt_ff) == 0:
        warnings.warn("No market VWAP data found for Final Four teams; blend skipped.")
        result["mkt_implied"] = np.nan
        result["blended_prob"] = result["p_champion"]
        result["model_edge"] = np.nan
        return result

    # Keep one row per TeamID (last trade)
    mkt_ff = mkt_ff.sort_values("TeamID").drop_duplicates(subset=["TeamID"], keep="last")
    vwap_total = mkt_ff["mkt_vwap"].sum()
    mkt_ff = mkt_ff.set_index("TeamID")

    result["mkt_implied"] = result["TeamID"].apply(
        lambda tid: mkt_ff.loc[tid, "mkt_vwap"] / vwap_total if tid in mkt_ff.index else np.nan
    )

    # Blend
    result["blended_prob"] = (
        (1 - market_weight) * result["p_champion"]
        + market_weight * result["mkt_implied"].fillna(result["p_champion"])
    )

    # Renormalize
    blend_total = result["blended_prob"].sum()
    if blend_total > 0:
        result["blended_prob"] = result["blended_prob"] / blend_total

    result["model_edge"] = result["p_champion"] - result["mkt_implied"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1B helpers: path features
# ─────────────────────────────────────────────────────────────────────────────

def compute_path_features(team_id: int,
                          season: int,
                          prior_games: list,
                          seeds_df: pd.DataFrame = None) -> dict:
    """
    Compute cumulative tournament path features for one team.

    prior_games: list of dicts (pre-filtered to DayNum < current game, same season).
        Each dict has keys: DayNum, won, margin, FGM, FGA, OppFGM, OppFGA,
                             NumOT, opp_team_id
    seeds_df: DataFrame with columns Season, TeamID, seed_num (for opponent seed lookup).

    Returns dict with PATH_FEATURES keys.
    Returns all NaN values when prior_games is empty (first-round game).
    """
    nan_result = {f: np.nan for f in PATH_FEATURES}

    if not prior_games:
        nan_result["path_games_played"] = 0
        return nan_result

    n = len(prior_games)
    margins = [g["margin"] for g in prior_games]
    fgm_total = sum(g["FGM"] for g in prior_games)
    fga_total = sum(g["FGA"] for g in prior_games)
    opp_fgm_total = sum(g["OppFGM"] for g in prior_games)
    opp_fga_total = sum(g["OppFGA"] for g in prior_games)
    ot_games = sum(1 for g in prior_games if g["NumOT"] > 0)

    fg_pct = fgm_total / fga_total if fga_total > 0 else np.nan
    opp_fg_pct = opp_fgm_total / opp_fga_total if opp_fga_total > 0 else np.nan

    # Opponent seed lookup
    avg_opp_seed = np.nan
    best_opp_seed = np.nan
    if seeds_df is not None and len(seeds_df) > 0:
        opp_seeds = []
        for g in prior_games:
            opp_id = g["opp_team_id"]
            row = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == opp_id)]
            if len(row) > 0:
                s = row.iloc[0]["seed_num"]
                if not np.isnan(float(s)):
                    opp_seeds.append(float(s))
        if opp_seeds:
            avg_opp_seed = float(np.mean(opp_seeds))
            best_opp_seed = float(np.min(opp_seeds))  # lower seed number = tougher opponent

    # Momentum: last_game_margin - first_game_margin (games sorted by DayNum)
    momentum = float(margins[-1] - margins[0]) if n >= 2 else 0.0

    return {
        "path_games_played": n,
        "path_avg_margin":   float(np.mean(margins)),
        "path_worst_margin": float(np.min(margins)),
        "path_best_margin":  float(np.max(margins)),
        "path_avg_opp_seed": avg_opp_seed,
        "path_best_opp_seed": best_opp_seed,
        "path_fg_pct":       fg_pct,
        "path_opp_fg_pct":   opp_fg_pct,
        "path_ot_games":     ot_games,
        "path_momentum":     momentum,
    }


def _update_path_features(pf: dict, sim_margin: float, loser_seed: float) -> dict:
    """
    Update path features after a simulated game.

    Used in predict_final_four to update finalists' path stats with their
    simulated semifinal result before computing the championship matchup.

    sim_margin: simulated scoring margin (positive = winner's perspective).
    loser_seed: seed of the opponent just beaten.
    """
    updated = dict(pf)  # shallow copy
    n = pf.get("path_games_played", 0)
    if np.isnan(n):
        n = 0
    n = int(n)

    new_n = n + 1
    old_avg = pf.get("path_avg_margin", sim_margin)
    if np.isnan(old_avg):
        old_avg = sim_margin

    updated["path_games_played"] = new_n
    updated["path_avg_margin"] = (old_avg * n + sim_margin) / new_n
    updated["path_worst_margin"] = min(
        pf.get("path_worst_margin", sim_margin) if not np.isnan(pf.get("path_worst_margin", np.nan)) else sim_margin,
        sim_margin,
    )
    updated["path_best_margin"] = max(
        pf.get("path_best_margin", sim_margin) if not np.isnan(pf.get("path_best_margin", np.nan)) else sim_margin,
        sim_margin,
    )
    # Momentum: sim_margin vs prior average (positive = performing above average)
    updated["path_momentum"] = sim_margin - old_avg

    if not np.isnan(float(loser_seed) if not isinstance(loser_seed, float) else loser_seed):
        ls = float(loser_seed)
        old_avg_seed = pf.get("path_avg_opp_seed", ls)
        if np.isnan(old_avg_seed):
            old_avg_seed = ls
        updated["path_avg_opp_seed"] = (old_avg_seed * n + ls) / new_n
        old_best_seed = pf.get("path_best_opp_seed", ls)
        if np.isnan(old_best_seed):
            old_best_seed = ls
        updated["path_best_opp_seed"] = min(old_best_seed, ls)

    return updated


def load_actual_path_features(data_dir: str,
                              season: int,
                              team_ids: list) -> dict:
    """
    Load real tournament path features for a set of teams up to (but not
    including) the Final Four games (DayNum < 152).

    Used in run_v2.py to provide actual E8 stats to predict_final_four.

    Returns dict {TeamID: path_feature_dict}.
    """
    kaggle_dir = os.path.join(data_dir, "kaggle")
    detailed_path = os.path.join(kaggle_dir, "MNCAATourneyDetailedResults.csv")
    seeds_path = os.path.join(kaggle_dir, "MNCAATourneySeeds.csv")

    if not os.path.exists(detailed_path):
        warnings.warn(f"MNCAATourneyDetailedResults.csv not found at {detailed_path}")
        return {tid: {"path_games_played": 0, **{f: np.nan for f in PATH_FEATURES if f != "path_games_played"}}
                for tid in team_ids}

    tourn = pd.read_csv(detailed_path)
    tourn = tourn[(tourn["Season"] == season) & (tourn["DayNum"] < 152)]

    seeds_df = None
    if os.path.exists(seeds_path):
        seeds_df = pd.read_csv(seeds_path)
        seeds_df["seed_num"] = seeds_df["Seed"].apply(parse_seed)
        seeds_df = seeds_df[["Season", "TeamID", "seed_num"]]

    result = {}
    for tid in team_ids:
        if tid is None:
            continue
        # Extract games where this team appeared
        as_winner = tourn[tourn["WTeamID"] == tid]
        as_loser = tourn[tourn["LTeamID"] == tid]

        prior_games = []
        for _, g in as_winner.iterrows():
            prior_games.append({
                "DayNum": int(g["DayNum"]),
                "won": 1,
                "margin": float(g["WScore"] - g["LScore"]),
                "FGM": float(g["WFGM"]),
                "FGA": float(g["WFGA"]),
                "OppFGM": float(g["LFGM"]),
                "OppFGA": float(g["LFGA"]),
                "NumOT": int(g["NumOT"]),
                "opp_team_id": int(g["LTeamID"]),
            })
        for _, g in as_loser.iterrows():
            prior_games.append({
                "DayNum": int(g["DayNum"]),
                "won": 0,
                "margin": float(g["LScore"] - g["WScore"]),
                "FGM": float(g["LFGM"]),
                "FGA": float(g["LFGA"]),
                "OppFGM": float(g["WFGM"]),
                "OppFGA": float(g["WFGA"]),
                "NumOT": int(g["NumOT"]),
                "opp_team_id": int(g["WTeamID"]),
            })

        prior_games.sort(key=lambda x: x["DayNum"])
        result[tid] = compute_path_features(tid, season, prior_games, seeds_df)

    return result
