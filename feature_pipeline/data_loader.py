"""
data_loader.py
--------------
Loads every year's Team Sheet CSV, the champions table, and the awards tables,
normalises team names, parses W-L records, and returns a single master DataFrame
with one row per (year, team).

Usage:
    from feature_pipeline.data_loader import load_all
    df = load_all(data_dir="data")
"""

import re
import glob
import os
import warnings
import pandas as pd
import numpy as np
from feature_pipeline.config import (
    COLUMN_ALIASES, TEAM_NAME_MAP,
    MASSEY_SYSTEMS, TEAM_STATS_FILES, TEAM_STATS_SKIP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_team(name: str) -> str:
    """Strip whitespace, apply known alias map."""
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


def parse_record(s) -> tuple:
    """'23-5' → (23, 5, 0.821).  Returns (NaN, NaN, NaN) on failure."""
    if not isinstance(s, str):
        return np.nan, np.nan, np.nan
    m = re.match(r"(\d+)-(\d+)", s.strip())
    if m:
        w, l = int(m.group(1)), int(m.group(2))
        total = w + l
        pct = w / total if total > 0 else np.nan
        return float(w), float(l), pct
    return np.nan, np.nan, np.nan


def safe_int(x):
    """Convert to int, return NaN on failure."""
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Team Sheet loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_one_team_sheet(path: str, year: int) -> pd.DataFrame:
    """Load a single year's team sheet CSV and return a normalised DataFrame."""
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
        return pd.DataFrame()

    # Rename columns to canonical names
    df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns}, inplace=True)

    # Must have a team column
    if "team" not in df.columns:
        warnings.warn(f"No team column found in {path} — skipping")
        return pd.DataFrame()

    df["team"] = df["team"].apply(normalise_team)
    df["year"] = year

    # Parse W-L records -------------------------------------------------
    for raw_col, prefix in [
        ("overall_record", "overall"),
        ("nc_record",      "nc"),
        ("road_record",    "road"),
    ]:
        if raw_col in df.columns:
            parsed = df[raw_col].apply(parse_record)
            df[f"{prefix}_wins"]   = parsed.apply(lambda x: x[0])
            df[f"{prefix}_losses"] = parsed.apply(lambda x: x[1])
            df[f"{prefix}_win_pct"]= parsed.apply(lambda x: x[2])

    # Parse conference record for conf_wins / conf_losses
    # conf_record often looks like "Big 12 (13-1)" — extract the part in parens
    if "conf_record" in df.columns:
        def _parse_conf(s):
            if not isinstance(s, str):
                return np.nan, np.nan, np.nan
            m = re.search(r"\((\d+)-(\d+)\)", s)
            if m:
                w, l = int(m.group(1)), int(m.group(2))
                t = w + l
                return float(w), float(l), w / t if t > 0 else np.nan
            return np.nan, np.nan, np.nan
        parsed = df["conf_record"].apply(_parse_conf)
        df["conf_wins"]    = parsed.apply(lambda x: x[0])
        df["conf_losses"]  = parsed.apply(lambda x: x[1])
        df["conf_win_pct"] = parsed.apply(lambda x: x[2])

    # Convert numeric columns
    NUMERIC = [
        "net_rank", "kpi", "sor", "bpi", "pom", "sag",
        "net_sos", "rpi_sos", "net_nc_sos", "rpi_nc_sos",
        "avg_net_wins", "avg_net_losses",
    ]
    # Also grab any quadrant columns that made it through (e.g. q1_overall, etc.)
    quadrant_cols = [c for c in df.columns if re.match(r"q[1-4]_", c)]
    for col in NUMERIC + quadrant_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_int)

    return df


def load_team_sheets(data_dir: str) -> pd.DataFrame:
    """Load all *_Team_Sheet*.csv files and stack them."""
    pattern = os.path.join(data_dir, "*_Team_Sheet*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        warnings.warn(f"No team sheet CSVs found under {data_dir}")
        return pd.DataFrame()

    frames = []
    for path in files:
        # Extract year from filename
        m = re.search(r"(\d{4})", os.path.basename(path))
        if not m:
            warnings.warn(f"Cannot parse year from {path}")
            continue
        year = int(m.group(1))
        df = _load_one_team_sheet(path, year)
        if not df.empty:
            frames.append(df)
            print(f"  Loaded {year}: {len(df)} teams from {os.path.basename(path)}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


# ─────────────────────────────────────────────────────────────────────────────
# Champions / Final Four loader
# ─────────────────────────────────────────────────────────────────────────────

FINISH_ORDER = ["Champion", "Runner-Up", "Third Place", "Fourth Place"]


def load_champions(data_dir: str) -> pd.DataFrame:
    """
    Load yearly_champions.csv.
    Returns long-form: one row per (year, team) with columns
    finish, finish_rank (1=champion … 4=semis loss), champion_flag.
    """
    path = os.path.join(data_dir, "yearly_champions.csv")
    if not os.path.exists(path):
        warnings.warn(f"Champions file not found: {path}")
        return pd.DataFrame()

    raw = pd.read_csv(path, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]

    rows = []
    for _, r in raw.iterrows():
        year = safe_int(r.get("Year", np.nan))
        if pd.isna(year):
            continue
        for rank, finish_col in enumerate(FINISH_ORDER, start=1):
            team = r.get(finish_col, np.nan)
            if pd.isna(team) or str(team).strip() == "":
                continue
            rows.append({
                "year":          year,
                "team":          normalise_team(str(team).strip()),
                "finish":        finish_col,
                "finish_rank":   rank,   # 1=champion, 4=semis loss
                "champion_flag": int(rank == 1),
                "overtime":      r.get("Overtime", ""),
                "vacated":       r.get("Champion_Vacated", ""),
            })

    df = pd.DataFrame(rows)
    # Mark vacated championships so we can optionally exclude them
    df["vacated"] = df["vacated"].fillna("").apply(lambda x: 1 if str(x).strip() == "1" else 0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Awards loader
# ─────────────────────────────────────────────────────────────────────────────

def load_awards(data_dir: str) -> pd.DataFrame:
    """
    Merge yearly_award_winners.csv and yearly_sporting_news_player.csv.
    Returns one row per year with binary flags: has_<award>_winner.
    These will later be joined onto the team-level frame.
    """
    rows = []

    # Award winners (many awards per year)
    aw_path = os.path.join(data_dir, "yearly_award_winners.csv")
    if os.path.exists(aw_path):
        aw = pd.read_csv(aw_path, dtype=str)
        aw.columns = [c.strip() for c in aw.columns]
        # Each row: Year + pairs of <Award>_Player, <Award>_Team
        team_cols = [c for c in aw.columns if c.endswith("_Team")]
        for _, r in aw.iterrows():
            year = safe_int(r.get("Year", np.nan))
            if pd.isna(year):
                continue
            for tc in team_cols:
                award_prefix = tc.replace("_Team", "").lower()
                team = r.get(tc, "")
                if isinstance(team, str) and team.strip():
                    rows.append({
                        "year":  year,
                        "team":  normalise_team(team.strip()),
                        f"has_{award_prefix}_award": 1,
                    })

    # Sporting News Player of the Year
    sn_path = os.path.join(data_dir, "yearly_sporting_news_player.csv")
    if os.path.exists(sn_path):
        sn = pd.read_csv(sn_path, dtype=str)
        sn.columns = [c.strip() for c in sn.columns]
        for _, r in sn.iterrows():
            year = safe_int(r.get("Year", np.nan))
            team = r.get("School", "")
            if pd.isna(year) or not isinstance(team, str):
                continue
            rows.append({
                "year": year,
                "team": normalise_team(team.strip()),
                "has_sporting_news_award": 1,
            })

    if not rows:
        return pd.DataFrame()

    # Collapse to one row per (year, team), binary flags
    awards_df = pd.DataFrame(rows)
    award_flag_cols = [c for c in awards_df.columns if c.startswith("has_")]
    awards_wide = (
        awards_df
        .groupby(["year", "team"])[award_flag_cols]
        .max()          # 1 if team has any of this award in this year
        .reset_index()
    )
    return awards_wide


# ─────────────────────────────────────────────────────────────────────────────
# Location loader
# ─────────────────────────────────────────────────────────────────────────────

def load_locations(data_dir: str) -> pd.DataFrame:
    """Load yearly_champion_location.csv → year + state of final."""
    path = os.path.join(data_dir, "yearly_champion_location.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["year"] = df["Year"].apply(safe_int)
    df["final_state"] = df.get("State", pd.Series(dtype=str))
    return df[["year", "final_state"]].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle: Tournament labels  (Tier 1 survival labels)
# ─────────────────────────────────────────────────────────────────────────────

# DayNum ranges per round (men's standard schedule).
# 2021 bubble year had E8 spread across 145–148 due to unusual scheduling.
_ROUND_DAYNUM = {
    "play_in": (134, 135),
    "r1":      (136, 137),
    "r2":      (138, 140),  # 140 used in 2021
    "s16":     (143, 144),
    "e8":      (145, 148),  # 147-148 used in 2021 bubble
    "ff":      (152, 152),
    "champ":   (154, 154),
}


def _daynum_to_round(daynum: int) -> str | None:
    """Map a DayNum to the round name, or None if outside tournament."""
    for rnd, (lo, hi) in _ROUND_DAYNUM.items():
        if lo <= daynum <= hi:
            return rnd
    return None


def load_tournament_labels(kaggle_dir: str,
                           min_season: int = 2003) -> pd.DataFrame:
    """
    Build Tier 1 survival labels for all seeded tournament teams.

    For each (Season, TeamID), derives:
      - seed_num      : integer seed 1–16
      - furthest_round: 0=play-in loss … 6=champion
      - Binary columns: survived_r1, survived_r2, survived_s16,
                        survived_e8, made_ff, champion

    Output: one row per (season, team_id), years >= min_season, skip 2020.
    """
    from feature_pipeline.name_resolver import build_id_lookup, resolve_team_name

    seeds_path   = os.path.join(kaggle_dir, "MNCAATourneySeeds.csv")
    results_path = os.path.join(kaggle_dir, "MNCAATourneyCompactResults.csv")
    teams_path   = os.path.join(kaggle_dir, "MTeams.csv")

    if not all(os.path.exists(p) for p in [seeds_path, results_path, teams_path]):
        warnings.warn("Missing Kaggle tournament files — skipping tournament labels")
        return pd.DataFrame()

    seeds   = pd.read_csv(seeds_path)
    results = pd.read_csv(results_path)
    teams_df = pd.read_csv(teams_path, dtype={"TeamID": int})

    seeds = seeds[seeds["Season"] >= min_season]
    seeds = seeds[seeds["Season"] != 2020]

    # Parse seed: "W01" → region="W", seed_num=1; "W16a" → seed_num=16
    def _parse_seed(s: str):
        m = re.match(r"[WXYZ](\d+)", str(s))
        return int(m.group(1)) if m else np.nan

    seeds["seed_num"] = seeds["Seed"].apply(_parse_seed)

    # Build win-tracker: for each (Season, TeamID), find the furthest round won
    round_order = ["play_in", "r1", "r2", "s16", "e8", "ff", "champ"]
    round_to_int = {r: i for i, r in enumerate(round_order)}

    # Teams that won each game → they survived that round
    wins = results[results["Season"] >= min_season].copy()
    wins = wins[wins["Season"] != 2020]
    wins["round"] = wins["DayNum"].apply(_daynum_to_round)
    wins = wins.dropna(subset=["round"])

    # For each team, get the highest round they WON
    won = (
        wins.groupby(["Season", "WTeamID"])["round"]
        .apply(lambda rs: max(rs, key=lambda r: round_to_int.get(r, -1)))
        .reset_index()
        .rename(columns={"WTeamID": "TeamID", "round": "best_win_round"})
    )

    # Merge seeds with wins
    df = seeds.merge(won, on=["Season", "TeamID"], how="left")
    df["best_win_round"] = df["best_win_round"].fillna("none")

    def _furthest(best_win: str, seeded: bool = True) -> int:
        if best_win == "champ": return 6
        if best_win == "ff":    return 5
        if best_win == "e8":    return 4
        if best_win == "s16":   return 3
        if best_win == "r2":    return 2
        if best_win == "r1":    return 1
        if best_win == "play_in": return 0  # survived play-in but lost R1
        return 0  # seeded but lost in first game they played

    df["furthest_round"] = df["best_win_round"].apply(_furthest)

    # Binary survival columns
    # survived_rN = won that round's game (now in the next round)
    # made_ff     = won E8 game (furthest_round ≥ 4) → now in Final Four
    # in_champ    = won FF semifinal (furthest_round ≥ 5) → in championship game
    # champion    = won championship game (furthest_round == 6)
    df["survived_r1"]  = (df["furthest_round"] >= 1).astype(int)
    df["survived_r2"]  = (df["furthest_round"] >= 2).astype(int)
    df["survived_s16"] = (df["furthest_round"] >= 3).astype(int)
    df["survived_e8"]  = (df["furthest_round"] >= 4).astype(int)
    df["in_champ"]     = (df["furthest_round"] >= 5).astype(int)  # won FF semi = in championship
    df["champion"]     = (df["furthest_round"] == 6).astype(int)

    # made_ff: used DayNum=152 directly (FF is always DayNum 152 in ALL years,
    # including 2021 bubble). Both teams that play AND both that win = all 4 FF teams.
    ff_teams = (
        results[results["DayNum"] == 152]
        .groupby("Season")[["WTeamID", "LTeamID"]]
        .apply(lambda g: set(g["WTeamID"]) | set(g["LTeamID"]))
    )
    def _in_ff(row):
        yr_ff = ff_teams.get(row["Season"], set())
        return int(row["TeamID"] in yr_ff)
    df["made_ff"] = df.apply(_in_ff, axis=1)

    # Add canonical team name for joining with non-Kaggle data
    tid_to_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    df["team_kaggle"] = df["TeamID"].map(tid_to_name)
    df["team"] = df["team_kaggle"].apply(normalise_team)

    df = df.rename(columns={"Season": "year"})
    keep_cols = [
        "year", "TeamID", "team", "Seed", "seed_num",
        "furthest_round", "survived_r1", "survived_r2",
        "survived_s16", "survived_e8", "made_ff", "champion",
    ]
    return df[keep_cols].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle: Game-derived features
# ─────────────────────────────────────────────────────────────────────────────

def load_kaggle_game_stats(kaggle_dir: str,
                           min_season: int = 2003) -> pd.DataFrame:
    """
    Aggregate per-team season stats from Kaggle game-by-game data.

    Regular season features (DayNum ≤ 132):
      kg_fg_pct, kg_fg3_pct, kg_ft_pct, kg_efg_pct,
      kg_off_reb_pg, kg_def_reb_pg, kg_ast_pg, kg_to_pg,
      kg_stl_pg, kg_blk_pg, kg_scoring_margin,
      kg_margin_last3, kg_margin_last5, kg_margin_last10, kg_margin_last15, kg_margin_last20,
      kg_road_win_pct, kg_opp_fg_pct, kg_opp_efg_pct, kg_ast_to_ratio

    Tournament path features (DayNum 134–146, safe for Final Four prediction):
      kg_tourney_avg_margin, kg_tourney_worst_margin,
      kg_tourney_fg_pct, kg_tourney_opp_fg_pct, kg_rounds_survived

    Returns: one row per (year, TeamID) for all D1 teams.
    """
    reg_path   = os.path.join(kaggle_dir, "MRegularSeasonDetailedResults.csv")
    tourn_path = os.path.join(kaggle_dir, "MNCAATourneyDetailedResults.csv")

    if not os.path.exists(reg_path):
        warnings.warn("MRegularSeasonDetailedResults.csv not found")
        return pd.DataFrame()

    reg = pd.read_csv(reg_path)
    reg = reg[(reg["Season"] >= min_season) & (reg["DayNum"] <= 132)]
    reg = reg[reg["Season"] != 2020]

    def _team_game_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot each game row into TWO rows — one per team —
        with consistent column names: FGM, FGA, FGM3, FTM, FTA,
        OR, DR, Ast, TO, Stl, Blk, PF, Score, OppScore, IsRoad, NumOT.
        """
        w = df.rename(columns={
            "WTeamID": "TeamID", "LTeamID": "OppID",
            "WScore": "Score",   "LScore": "OppScore",
            "WFGM": "FGM", "WFGA": "FGA", "WFGM3": "FGM3", "WFGA3": "FGA3",
            "WFTM": "FTM", "WFTA": "FTA",
            "WOR": "OR", "WDR": "DR", "WAst": "Ast",
            "WTO": "TO", "WStl": "Stl", "WBlk": "Blk", "WPF": "PF",
            "LFGM": "OppFGM", "LFGA": "OppFGA",
            "LFGM3": "OppFGM3", "LFGA3": "OppFGA3",
        })
        w["IsRoad"] = (w["WLoc"] == "A").astype(int)
        w["Won"]    = 1

        l = df.rename(columns={
            "LTeamID": "TeamID", "WTeamID": "OppID",
            "LScore": "Score",   "WScore": "OppScore",
            "LFGM": "FGM", "LFGA": "FGA", "LFGM3": "FGM3", "LFGA3": "FGA3",
            "LFTM": "FTM", "LFTA": "FTA",
            "LOR": "OR", "LDR": "DR", "LAst": "Ast",
            "LTO": "TO", "LStl": "Stl", "LBlk": "Blk", "LPF": "PF",
            "WFGM": "OppFGM", "WFGA": "OppFGA",
            "WFGM3": "OppFGM3", "WFGA3": "OppFGA3",
        })
        l["IsRoad"] = (l["WLoc"] == "H").astype(int)  # opp was home → we were road
        l["Won"]    = 0

        keep = ["Season", "DayNum", "TeamID", "OppID", "Score", "OppScore",
                "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF",
                "OppFGM", "OppFGA", "OppFGM3", "OppFGA3",
                "IsRoad", "Won", "NumOT"]
        keep = [c for c in keep if c in w.columns]
        return pd.concat([w[keep], l[keep]], ignore_index=True)

    reg_long = _team_game_rows(reg)
    reg_long["Margin"] = reg_long["Score"] - reg_long["OppScore"]

    # ── Regular season aggregates ─────────────────────────────────────────────
    def _agg_season(grp: pd.DataFrame) -> pd.Series:
        g = len(grp)
        if g == 0:
            return pd.Series(dtype=float)

        fgm = grp["FGM"].sum(); fga = grp["FGA"].sum()
        fgm3 = grp["FGM3"].sum(); fga3 = grp["FGA3"].sum()
        ftm = grp["FTM"].sum(); fta = grp["FTA"].sum()
        opp_fgm = grp["OppFGM"].sum(); opp_fga = grp["OppFGA"].sum()
        opp_fgm3 = grp["OppFGM3"].sum(); opp_fga3 = grp["OppFGA3"].sum()

        # Recent form windows by DayNum
        recent3  = grp.nlargest(3,  "DayNum")
        recent5  = grp.nlargest(5,  "DayNum")
        recent10 = grp.nlargest(10, "DayNum")
        recent15 = grp.nlargest(15, "DayNum")
        recent20 = grp.nlargest(20, "DayNum")

        road_games = grp[grp["IsRoad"] == 1]
        road_wins  = road_games[road_games["Won"] == 1]

        return pd.Series({
            "kg_fg_pct":        fgm / fga  if fga > 0 else np.nan,
            "kg_fg3_pct":       fgm3 / fga3 if fga3 > 0 else np.nan,
            "kg_ft_pct":        ftm / fta  if fta > 0 else np.nan,
            "kg_efg_pct":       (fgm + 0.5 * fgm3) / fga if fga > 0 else np.nan,
            "kg_off_reb_pg":    grp["OR"].sum() / g,
            "kg_def_reb_pg":    grp["DR"].sum() / g,
            "kg_ast_pg":        grp["Ast"].sum() / g,
            "kg_to_pg":         grp["TO"].sum() / g,
            "kg_stl_pg":        grp["Stl"].sum() / g,
            "kg_blk_pg":        grp["Blk"].sum() / g,
            "kg_scoring_margin":  grp["Margin"].mean(),
            "kg_margin_last3":    recent3["Margin"].mean()  if len(recent3)  > 0 else np.nan,
            "kg_margin_last5":    recent5["Margin"].mean()  if len(recent5)  > 0 else np.nan,
            "kg_margin_last10":   recent10["Margin"].mean() if len(recent10) > 0 else np.nan,
            "kg_margin_last15":   recent15["Margin"].mean() if len(recent15) > 0 else np.nan,
            "kg_margin_last20":   recent20["Margin"].mean() if len(recent20) > 0 else np.nan,
            "kg_days_since_last_game": 132 - grp["DayNum"].max(),
            "kg_road_win_pct":  len(road_wins) / len(road_games) if len(road_games) > 0 else np.nan,
            "kg_opp_fg_pct":    opp_fgm / opp_fga if opp_fga > 0 else np.nan,
            "kg_opp_efg_pct":   (opp_fgm + 0.5 * opp_fgm3) / opp_fga if opp_fga > 0 else np.nan,
            "kg_ast_to_ratio":  grp["Ast"].sum() / max(grp["TO"].sum(), 1),
            "kg_games":         g,
        })

    reg_stats = (
        reg_long.groupby(["Season", "TeamID"])
        .apply(_agg_season, include_groups=False)
        .reset_index()
        .rename(columns={"Season": "year"})
    )

    # ── Tournament path features (DayNum 134–146) ─────────────────────────────
    tourn_stats = pd.DataFrame()
    if os.path.exists(tourn_path):
        tourn = pd.read_csv(tourn_path)
        tourn = tourn[(tourn["Season"] >= min_season) &
                      (tourn["DayNum"] >= 134) &
                      (tourn["DayNum"] <= 146)]
        tourn = tourn[tourn["Season"] != 2020]

        if len(tourn) > 0:
            tourn_long = _team_game_rows(tourn)
            tourn_long["Margin"] = tourn_long["Score"] - tourn_long["OppScore"]

            def _agg_tourn(grp: pd.DataFrame) -> pd.Series:
                g = len(grp)
                fgm = grp["FGM"].sum(); fga = grp["FGA"].sum()
                opp_fgm = grp["OppFGM"].sum(); opp_fga = grp["OppFGA"].sum()
                wins = grp[grp["Won"] == 1]
                return pd.Series({
                    "kg_tourney_avg_margin":  grp["Margin"].mean(),
                    "kg_tourney_worst_margin":grp["Margin"].min(),
                    "kg_tourney_fg_pct":      fgm / fga if fga > 0 else np.nan,
                    "kg_tourney_opp_fg_pct":  opp_fgm / opp_fga if opp_fga > 0 else np.nan,
                    "kg_rounds_survived":     len(wins),
                })

            tourn_stats = (
                tourn_long.groupby(["Season", "TeamID"])
                .apply(_agg_tourn, include_groups=False)
                .reset_index()
                .rename(columns={"Season": "year"})
            )

    # Merge regular season + tournament path
    if not tourn_stats.empty:
        result = reg_stats.merge(tourn_stats, on=["year", "TeamID"], how="left")
    else:
        result = reg_stats
        for col in ["kg_tourney_avg_margin", "kg_tourney_worst_margin",
                    "kg_tourney_fg_pct", "kg_tourney_opp_fg_pct", "kg_rounds_survived"]:
            result[col] = np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle: Massey ordinal rankings
# ─────────────────────────────────────────────────────────────────────────────

def load_massey_ordinals(kaggle_dir: str,
                         systems: list = None,
                         min_season: int = 2003) -> pd.DataFrame:
    """
    Load pre-tournament (RankingDayNum == 133) rankings for selected systems.

    Returns one row per (year, TeamID) with columns massey_POM, massey_SAG, etc.
    """
    if systems is None:
        systems = MASSEY_SYSTEMS

    path = os.path.join(kaggle_dir, "MMasseyOrdinals.csv")
    if not os.path.exists(path):
        warnings.warn("MMasseyOrdinals.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Pre-tournament rankings: keep latest available snapshot <= day 133 per team/system
    # (some systems, e.g. WLK/COL/DOL, publish through day 128 rather than 133)
    df = df[
        (df["RankingDayNum"] <= 133) &
        (df["SystemName"].isin(systems)) &
        (df["Season"] >= min_season) &
        (df["Season"] != 2020)
    ]
    # Keep only the latest DayNum per (Season, TeamID, SystemName)
    df = (
        df.sort_values("RankingDayNum")
        .groupby(["Season", "TeamID", "SystemName"], as_index=False)
        .last()
    )

    wide = (
        df.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="OrdinalRank",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns = (
        ["year", "TeamID"] +
        [f"massey_{s}" for s in wide.columns[2:]]
    )
    return wide


# ─────────────────────────────────────────────────────────────────────────────
# BPI pre-Final Four (latest RankingDayNum < 152)
# ─────────────────────────────────────────────────────────────────────────────

def load_bpi_at_finals(kaggle_dir: str, min_season: int = 2008) -> pd.DataFrame:
    """
    Load ESPN BPI rankings immediately before the Final Four (RankingDayNum < 152)
    from MMasseyOrdinals.  Uses the latest available ranking per (Season, TeamID)
    that predates the Final Four games (DayNum 152), so there is no lookahead.

    Returns one row per (year, TeamID) with column kg_bpi_at_finals.
    Coverage: 2008–2026.
    """
    path = os.path.join(kaggle_dir, "MMasseyOrdinals.csv")
    if not os.path.exists(path):
        warnings.warn("MMasseyOrdinals.csv not found; skipping BPI at finals")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[
        (df["SystemName"] == "BPI") &
        (df["RankingDayNum"] < 152) &
        (df["Season"] >= min_season) &
        (df["Season"] != 2020)
    ][["Season", "TeamID", "RankingDayNum", "OrdinalRank"]].copy()

    # Keep the latest ranking day per (Season, TeamID)
    df = (
        df.sort_values("RankingDayNum")
          .groupby(["Season", "TeamID"], as_index=False)
          .last()
    )

    df = df.rename(columns={"Season": "year", "OrdinalRank": "kg_bpi_at_finals"})
    return df[["year", "TeamID", "kg_bpi_at_finals"]]


# ─────────────────────────────────────────────────────────────────────────────
# Team stats loader  (from {year}-team-stats/ directories)
# ─────────────────────────────────────────────────────────────────────────────

def load_team_stats(data_dir: str,
                    years: list = None) -> pd.DataFrame:
    """
    Load per-team season aggregate stats from the {year}-team-stats/ directories.

    Files are full-season snapshots (regular season + conf tourney + NCAA tourney
    through Elite Eight). Safe for Final Four prediction (Tier 2). NOT safe as
    Tier 1 features for early-round survival prediction.

    Returns: one row per (year, team) with ts_* prefixed columns.
    """
    from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id, strip_conference

    lookup = build_id_lookup(os.path.join(data_dir, "kaggle"))

    # Discover available years
    all_year_dirs = glob.glob(os.path.join(data_dir, "*-team-stats"))
    available_years = []
    for d in all_year_dirs:
        m = re.search(r"(\d{4})-team-stats", d)
        if m:
            yr = int(m.group(1))
            if yr != 2020:
                available_years.append(yr)

    if years is not None:
        available_years = [y for y in available_years if y in years]

    all_frames = []

    for year in sorted(available_years):
        stats_dir = os.path.join(data_dir, f"{year}-team-stats")
        csv_files = glob.glob(os.path.join(stats_dir, "*.csv"))

        # For each target stat, find the matching file
        year_stat_dfs: dict[str, pd.DataFrame] = {}

        for canonical_col, cfg in TEAM_STATS_FILES.items():
            if year < cfg["min_year"]:
                continue

            # Find file matching any of the patterns
            matched_file = None
            for csv_path in csv_files:
                # Extract stat name: filename before '__' or before '.csv'
                fname = os.path.basename(csv_path)
                stat_name = fname.split("__")[0] if "__" in fname else fname.replace(".csv", "")

                # Skip files in the skip list
                if any(stat_name.startswith(skip) for skip in TEAM_STATS_SKIP):
                    continue

                if any(stat_name == pat or stat_name.startswith(pat)
                       for pat in cfg["patterns"]):
                    matched_file = csv_path
                    break

            if matched_file is None:
                continue

            try:
                df = pd.read_csv(matched_file, dtype=str)
            except Exception as e:
                warnings.warn(f"Could not read {matched_file}: {e}")
                continue

            if "Team" not in df.columns or cfg["stat_col"] not in df.columns:
                # Try to find the stat column by partial match
                stat_candidates = [c for c in df.columns if cfg["stat_col"] in c]
                if not stat_candidates or "Team" not in df.columns:
                    continue
                stat_col = stat_candidates[0]
            else:
                stat_col = cfg["stat_col"]

            # Build (team_name, stat_value) pairs
            sub = df[["Team", stat_col]].copy()
            sub.columns = ["raw_team", "stat_val"]
            sub["raw_team"] = sub["raw_team"].astype(str).str.strip()
            # Strip conference from team name, then normalise
            sub["team"] = sub["raw_team"].apply(
                lambda n: normalise_team(strip_conference(n))
            )
            sub["stat_val"] = pd.to_numeric(sub["stat_val"], errors="coerce")
            sub = sub.dropna(subset=["team", "stat_val"])
            sub = sub[sub["team"].str.len() > 0]

            year_stat_dfs[canonical_col] = sub.set_index("team")["stat_val"]

        if not year_stat_dfs:
            continue

        # Combine all stats for this year into one wide DataFrame
        year_df = pd.DataFrame(year_stat_dfs)
        year_df.index.name = "team"
        year_df = year_df.reset_index()
        year_df["year"] = year
        all_frames.append(year_df)

    if not all_frames:
        warnings.warn("No team-stats data loaded")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True, sort=False)
    print(f"  Loaded team stats: {len(result)} rows across "
          f"{result['year'].nunique()} years, "
          f"{len(result['team'].unique())} unique teams")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Master merge
# ─────────────────────────────────────────────────────────────────────────────

def load_all(data_dir: str = "data",
             include_kaggle: bool = True,
             include_team_stats: bool = False,
             include_market: bool = False,
             verbose: bool = True) -> pd.DataFrame:
    """
    Returns the master DataFrame with one row per (year, team).

    Args:
        include_kaggle:     Merge Kaggle game stats, tournament labels,
                            Massey ordinals. Expands rows to all tournament
                            teams (not just Final Four).
        include_team_stats: Merge {year}-team-stats/ season aggregate stats.
                            Use for Tier 2 (Final Four) features only.
        include_market:     Merge Kalshi market microstructure features
                            (2025–2026 only).
        verbose:            Print progress.

    Includes:
    - all team sheet metrics (numeric)
    - parsed W-L records and win %
    - finish label columns (finish_rank, champion_flag) for Final Four teams
    - award flags (has_*_award)
    - final_location_state
    - in_final_four flag
    - Tier 1 survival labels (if include_kaggle)
    - Kaggle game-derived features + tournament path features (if include_kaggle)
    - Massey ordinal rankings (if include_kaggle)
    - ts_* team stats (if include_team_stats)
    """
    kaggle_dir   = os.path.join(data_dir, "kaggle")
    sheets_dir   = os.path.join(data_dir, "team_sheets")
    yearlys_dir  = os.path.join(data_dir, "yearlys")

    if verbose:
        print("=== Loading team sheets ===")
    sheets = load_team_sheets(sheets_dir)

    if verbose:
        print("\n=== Loading champions ===")
    champs = load_champions(yearlys_dir)

    if verbose:
        print("\n=== Loading awards ===")
    awards = load_awards(yearlys_dir)

    if verbose:
        print("\n=== Loading locations ===")
    locs = load_locations(yearlys_dir)

    # ── Merge champions onto sheets ───────────────────────────────────────
    if not sheets.empty and not champs.empty:
        df = sheets.merge(
            champs[["year", "team", "finish", "finish_rank",
                    "champion_flag", "overtime", "vacated"]],
            on=["year", "team"],
            how="left",
        )
        df["in_final_four"] = df["finish_rank"].notna().astype(int)
        df["finish_rank"]   = df["finish_rank"].fillna(0).astype(int)
        df["champion_flag"] = df["champion_flag"].fillna(0).astype(int)
    else:
        df = sheets.copy() if not sheets.empty else champs.copy()
        df["in_final_four"] = 0
        df["finish_rank"]   = 0
        df["champion_flag"] = 0

    # ── Merge awards ──────────────────────────────────────────────────────
    if not awards.empty and not df.empty:
        df = df.merge(awards, on=["year", "team"], how="left")
        award_cols = [c for c in df.columns if c.startswith("has_")]
        df[award_cols] = df[award_cols].fillna(0).astype(int)

    # ── Merge locations ───────────────────────────────────────────────────
    if not locs.empty and not df.empty:
        df = df.merge(locs, on="year", how="left")

    # ── Kaggle integration ────────────────────────────────────────────────
    if include_kaggle and os.path.isdir(kaggle_dir):
        if verbose:
            print("\n=== Loading Kaggle tournament labels ===")
        tourn_labels = load_tournament_labels(kaggle_dir)

        if verbose:
            print("\n=== Loading Kaggle game stats ===")
        game_stats = load_kaggle_game_stats(kaggle_dir)

        if verbose:
            print("\n=== Loading Massey ordinals ===")
        massey = load_massey_ordinals(kaggle_dir)

        if not df.empty and not tourn_labels.empty:
            # Merge survival labels onto existing frame (Final Four teams first)
            df = df.merge(
                tourn_labels[["year", "team", "TeamID", "Seed", "seed_num",
                               "furthest_round", "survived_r1", "survived_r2",
                               "survived_s16", "survived_e8", "made_ff", "champion"]],
                on=["year", "team"],
                how="left",
            )
            # Add remaining tournament teams not in team sheets
            # (teams that didn't make Final Four but were seeded)
            ff_keys = set(zip(df["year"], df["team"]))
            extra_rows = tourn_labels[
                ~tourn_labels.apply(
                    lambda r: (r["year"], r["team"]) in ff_keys, axis=1
                )
            ].copy()
            if not extra_rows.empty:
                df = pd.concat([df, extra_rows], ignore_index=True, sort=False)
                df["in_final_four"]  = df["in_final_four"].fillna(0).astype(int)
                df["finish_rank"]    = df["finish_rank"].fillna(0).astype(int)
                df["champion_flag"]  = df["champion_flag"].fillna(0).astype(int)

        # Merge game stats on (year, TeamID) if TeamID is available, else skip
        if not game_stats.empty and "TeamID" in df.columns:
            df = df.merge(game_stats, on=["year", "TeamID"], how="left")
        elif not game_stats.empty:
            # Fall back to merging on (year, team) via team name resolution
            from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id
            lookup = build_id_lookup(kaggle_dir)
            teams_df = pd.read_csv(os.path.join(kaggle_dir, "MTeams.csv"),
                                   dtype={"TeamID": int})
            tid_map = dict(zip(teams_df["TeamName"].apply(str.lower),
                               teams_df["TeamID"]))
            game_stats_named = game_stats.copy()
            game_stats_named["team"] = game_stats_named["TeamID"].map(
                {v: k.title() for k, v in tid_map.items()}
            )
            df = df.merge(
                game_stats_named.drop(columns=["TeamID"]),
                on=["year", "team"], how="left"
            )

        # Merge Massey ordinals
        if not massey.empty and "TeamID" in df.columns:
            df = df.merge(massey, on=["year", "TeamID"], how="left")

        # Merge BPI at Final Four (DayNum=152)
        if verbose:
            print("\n=== Loading BPI at finals ===")
        bpi_finals = load_bpi_at_finals(kaggle_dir)
        if not bpi_finals.empty and "TeamID" in df.columns:
            df = df.merge(bpi_finals, on=["year", "TeamID"], how="left")

            # ── Coalesce sources: fill team-sheet gaps with Massey where available ─
            # Team sheets cover 2005-2026; Massey covers 2003-2026 for POM/RPI but not SAG.
            # For years/teams where team sheets are missing (e.g. 2003–2004, non-FF
            # teams in Tier 1 frame), use the Massey value as fallback.
            _coalesce_pairs = [
                ("pom",      "massey_POM"),   # POM: both have it; Massey extends to 2003
                ("sag",      "massey_SAG"),   # SAG: Massey only through 2023; TS fills gaps
                ("rpi_sos",  "massey_RPI"),   # RPI rank: treat as interchangeable
            ]
            for ts_col, massey_col in _coalesce_pairs:
                if ts_col in df.columns and massey_col in df.columns:
                    # Fill missing team-sheet value with Massey fallback
                    df[ts_col] = df[ts_col].fillna(df[massey_col])
                    # And vice versa: fill missing Massey with team-sheet value
                    df[massey_col] = df[massey_col].fillna(df[ts_col])
                    if verbose:
                        filled = (df[ts_col].notna().sum())
                        print(f"  Coalesced {ts_col} ↔ {massey_col}: {filled} non-null rows")

    # ── Team stats integration ────────────────────────────────────────────
    if include_team_stats:
        if verbose:
            print("\n=== Loading team stats ===")
        ts = load_team_stats(data_dir)
        if not ts.empty and not df.empty:
            df = df.merge(ts, on=["year", "team"], how="left")

    # ── Market data integration ───────────────────────────────────────────
    if include_market:
        try:
            from feature_pipeline.market_features import load_kalshi_trades, compute_market_features
            if verbose:
                print("\n=== Loading market features ===")
            trades = load_kalshi_trades(data_dir)
            if not trades.empty:
                mkt = compute_market_features(trades)
                df = df.merge(mkt, on=["year", "team"], how="left")
        except ImportError:
            warnings.warn("market_features.py not yet available — skipping market data")

    # ── Final housekeeping ────────────────────────────────────────────────
    if not df.empty:
        df.sort_values(["year", "finish_rank"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

    if verbose:
        n_years   = df["year"].nunique() if not df.empty else 0
        n_teams   = len(df)
        n_ff      = df["in_final_four"].sum() if not df.empty else 0
        n_seeded  = df["seed_num"].notna().sum() if "seed_num" in df.columns else 0
        print(f"\n=== Master frame: {n_teams} rows, {n_years} years, "
              f"{n_ff} Final Four appearances, {n_seeded} seeded team-seasons ===")

    return df
