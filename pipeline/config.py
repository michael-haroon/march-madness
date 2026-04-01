"""
config.py
---------
Central configuration: which features exist in which years,
column name aliases, missing-data strategy, and model hyperparameters.
"""

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = "data"                          # relative to project root

# ── Feature availability by year ─────────────────────────────────────────────
# This is the key table.  When a feature group is unavailable for a year,
# its columns are filled with NaN and a binary missingness indicator is added.
FEATURE_AVAILABILITY = {
    # year: set of feature groups available
    2005: {"rpi", "sos", "record"},
    2006: {"rpi", "sos", "record"},
    2007: {"rpi", "sos", "record"},
    2008: {"rpi", "sos", "record"},
    2009: {"rpi", "sos", "record"},
    2010: {"rpi", "sos", "record"},
    2011: {"rpi", "sos", "record"},
    2012: {"rpi", "sos", "record"},
    2013: {"rpi", "sos", "record"},
    2014: {"rpi", "sos", "record"},
    2015: {"rpi", "sos", "record"},
    2016: {"rpi", "sos", "record", "kpi", "sor"},
    2017: {"rpi", "sos", "record", "kpi", "sor"},
    2018: {"rpi", "sos", "record", "kpi", "sor"},
    2019: {"rpi", "sos", "record", "kpi", "sor"},
    # NET introduced 2019 by NCAA; 2020 = no tournament (COVID)
    2021: {"rpi", "sos", "record", "kpi", "sor", "net", "bpi", "pom", "sag", "quadrant",
           "kaggle", "team_stats_core"},
    2022: {"rpi", "sos", "record", "kpi", "sor", "net", "bpi", "pom", "sag", "quadrant",
           "kaggle", "team_stats_core", "team_stats_extended"},
    2023: {"rpi", "sos", "record", "kpi", "sor", "net", "bpi", "pom", "sag", "quadrant",
           "kaggle", "team_stats_core", "team_stats_extended"},
    2024: {"rpi", "sos", "record", "kpi", "sor", "net", "bpi", "pom",        "quadrant",  # SAG gone
           "kaggle", "team_stats_core", "team_stats_extended"},
    2025: {"rpi", "sos", "record", "kpi", "sor", "net", "bpi", "pom",        "quadrant",
           "kaggle", "team_stats_core", "team_stats_extended", "market"},
    2026: {"rpi", "sos", "record", "kpi", "sor", "net", "bpi", "pom", "sag", "quadrant",
           "kaggle", "tourney_path", "team_stats_core", "team_stats_extended", "market"},
}
# Back-fill kaggle and team_stats_core for earlier years
for _yr in list(range(2003, 2021)) + [2021]:
    if _yr == 2020:
        continue
    FEATURE_AVAILABILITY.setdefault(_yr, set()).update({"kaggle", "team_stats_core"})
for _yr in range(2016, 2021):
    FEATURE_AVAILABILITY.setdefault(_yr, set()).add("team_stats_extended")
# tourney_path: available for all complete seasons from Kaggle (2003-2025)
for _yr in range(2003, 2026):
    if _yr == 2020:
        continue
    FEATURE_AVAILABILITY.setdefault(_yr, set()).add("tourney_path")

# ── Canonical column names after loading ─────────────────────────────────────
# Maps raw CSV column names (across years) → canonical internal name.
# Add more aliases as you discover them in older PDFs.
COLUMN_ALIASES = {
    # Team identification
    "Team":             "team",
    "TEAM":             "team",
    "School":           "team",
    "team_name":        "team",

    # NET rank
    "NET_Rank":         "net_rank",
    "NET":              "net_rank",
    "Net":              "net_rank",

    # Result-based metrics
    "RB_KPI":           "kpi",
    "KPI":              "kpi",
    "RB_SOR":           "sor",
    "SOR":              "sor",

    # Predictive metrics
    "PM_BPI":           "bpi",
    "BPI":              "bpi",
    "PM_POM":           "pom",
    "POM":              "pom",
    "PM_SAG":           "sag",
    "SAG":              "sag",

    # Strength of schedule
    "NET_SOS":          "net_sos",
    "RPI_SOS":          "rpi_sos",
    "NET_NonConf_SOS":  "net_nc_sos",
    "RPI_NonConf_SOS":  "rpi_nc_sos",
    "SOS_D1":           "rpi_sos",          # pre-2021: overall RPI SOS score
    "SOS_NonConf":      "rpi_nc_sos",       # pre-2021: non-conf RPI SOS score
    "Opp_SOS_D1":       "opp_sos_d1",
    "Opp_SOS_NonConf":  "opp_sos_nonconf",
    "RPI_Rank_D1":      "rpi_rank_d1",      # RPI rank overall (distinct from SOS score)
    "RPI_Rank_NonConf": "rpi_rank_nonconf",

    # Records
    "Overall_Record":   "overall_record",
    "NonConf_Record":   "nc_record",
    "Road_Record":      "road_record",
    "Conference_Record":"conf_record",
    "Record":           "overall_record",   # pre-2021 team sheets
    "Avg_NET_Wins":     "avg_net_wins",
    "Avg_NET_Losses":   "avg_net_losses",
    "Avg_RPI_Win":      "avg_net_wins",     # pre-2021 RPI equivalent
    "Avg_RPI_Loss":     "avg_net_losses",

    # New-era metrics (2025+)
    "RB_WAB":           "rb_wab",
    "PM_T-Rank":        "pm_trank",
}

# ── Name normalisation: team name variants → canonical name ──────────────────
# Extend this as you encounter more inconsistencies.
TEAM_NAME_MAP = {
    "Loyola Chicago":       "Loyola (Ill.)",
    "Loyola (Ill)":         "Loyola (Ill.)",
    "UConn":                "Connecticut",
    "UCONN":                "Connecticut",
    "St. John's":           "St. John's (NY)",
    "Saint John's":         "St. John's (NY)",
    "NC State":             "North Carolina State",
    "Ole Miss":             "Mississippi",
    "Southern Cal":         "USC",
    "Miami (FL)":           "Miami",
    "Miami (Fla.)":         "Miami",
    "Texas A&M":            "Texas A&M",
    "VCU":                  "Virginia Commonwealth",
    "UNLV":                 "Nevada Las Vegas",
    "LSU":                  "Louisiana State",
    "TCU":                  "Texas Christian",
    "SMU":                  "Southern Methodist",
    "UCF":                  "Central Florida",
    "UAB":                  "Alabama Birmingham",
    "UTSA":                 "Texas San Antonio",
    "UTEP":                 "Texas El Paso",
}

# ── Missing data strategy ─────────────────────────────────────────────────────
# "indicator" = add a binary flag column + fill NaN with median
# "drop"      = drop rows where this feature is NaN
# "zero"      = fill with 0 (for rank-difference features this is neutral)
MISSING_STRATEGY = {
    "net_rank":    "indicator",
    "kpi":         "indicator",
    "sor":         "indicator",
    "bpi":         "indicator",
    "pom":         "indicator",
    "sag":         "indicator",
    "net_sos":     "indicator",
    "rpi_sos":     "indicator",
    "win_pct":     "zero",
    "q1_win_pct":  "indicator",
    "entropy":     "indicator",
    "cusum_peak":  "zero",
    "rb_wab":      "indicator",
    "pm_trank":    "indicator",
    # Kaggle game-derived features
    "kg_fg_pct":         "indicator",
    "kg_fg3_pct":        "indicator",
    "kg_ft_pct":         "indicator",
    "kg_efg_pct":        "indicator",
    "kg_scoring_margin": "zero",
    "kg_recent_margin":  "zero",
    "kg_road_win_pct":   "zero",
    "kg_ast_to_ratio":   "indicator",
    "kg_opp_fg_pct":     "indicator",
    "kg_opp_efg_pct":    "indicator",
    # Tournament path features
    "kg_tourney_avg_margin":  "zero",
    "kg_tourney_worst_margin":"zero",
    "kg_tourney_fg_pct":      "indicator",
    "kg_tourney_opp_fg_pct":  "indicator",
    "kg_rounds_survived":     "zero",
    # Team-stats features — all indicator (missing = not in this era)
    **{col: "indicator" for col in [
        "ts_fg_pct", "ts_fg_pct_def", "ts_ft_pct", "ts_three_pct",
        "ts_three_pct_def", "ts_scoring_margin", "ts_rebound_margin",
        "ts_turnover_margin", "ts_assists_pg", "ts_steals_pg",
        "ts_blocks_pg", "ts_fouls_pg", "ts_reb_pg",
        "ts_efg_pct", "ts_off_reb_pg", "ts_def_reb_pg",
        "ts_bench_pts_pg", "ts_fastbreak_pts_pg", "ts_to_forced_pg",
    ]},
    # Market microstructure
    "mkt_vwap":        "indicator",
    "mkt_last_price":  "indicator",
    "mkt_ofi":         "zero",
    "mkt_momentum":    "zero",
    "mkt_trade_count": "zero",
    "mkt_volatility":  "zero",
    "mkt_price_range": "zero",
}

# ── Kaggle Massey ordinal systems to load (pre-tournament DayNum=133 only) ────
# Chosen for long-running continuity and broad coverage.
MASSEY_SYSTEMS = ["POM", "SAG", "RPI", "MOR", "WLK", "DOL", "COL"]

# ── Team-stats file configuration ─────────────────────────────────────────────
# Each entry: canonical output column → {patterns: filename prefixes to match,
#                                         stat_col: column to extract,
#                                         min_year: first year available}
TEAM_STATS_FILES = {
    "ts_fg_pct":         {"patterns": ["Field Goal Percentage", "Field-Goal Percentage"],
                          "stat_col": "FG%",       "min_year": 2003},
    "ts_fg_pct_def":     {"patterns": ["Field Goal Percentage Defense", "Field-Goal Percentage Defense"],
                          "stat_col": "OPP FG%",   "min_year": 2003},
    "ts_ft_pct":         {"patterns": ["Free Throw Percentage", "Free-Throw Percentage"],
                          "stat_col": "FT%",        "min_year": 2003},
    "ts_three_pct":      {"patterns": ["Three Point Percentage", "Three-Point Field-Goal Percentage"],
                          "stat_col": "3FG%",       "min_year": 2003},
    "ts_three_pct_def":  {"patterns": ["Three Point Percentage Defense", "Three Pt FG Defense"],
                          "stat_col": "Pct",        "min_year": 2009},
    "ts_scoring_margin": {"patterns": ["Scoring Margin"],
                          "stat_col": "SCR MAR",    "min_year": 2003},
    "ts_rebound_margin": {"patterns": ["Rebound Margin"],
                          "stat_col": "REB MAR",    "min_year": 2003},
    "ts_turnover_margin":{"patterns": ["Turnover Margin"],
                          "stat_col": "Ratio",      "min_year": 2009},
    "ts_assists_pg":     {"patterns": ["Assists Per Game"],
                          "stat_col": "APG",        "min_year": 2003},
    "ts_steals_pg":      {"patterns": ["Steals Per Game"],
                          "stat_col": "STPG",       "min_year": 2003},
    "ts_blocks_pg":      {"patterns": ["Blocks Per Game", "Blocked Shots Per Game"],
                          "stat_col": "BKPG",       "min_year": 2003},
    "ts_fouls_pg":       {"patterns": ["Fouls Per Game", "Personal Fouls Per Game"],
                          "stat_col": "PFPG",       "min_year": 2003},
    "ts_reb_pg":         {"patterns": ["Rebounds Per Game"],
                          "stat_col": "RPG",        "min_year": 2003},
    # Post-2016 only
    "ts_efg_pct":        {"patterns": ["Effective FG pct"],
                          "stat_col": "Pct",        "min_year": 2016},
    "ts_off_reb_pg":     {"patterns": ["Rebounds _Offensive_ Per Game",
                                       "Offensive Rebounds Per Game"],
                          "stat_col": "RPG",        "min_year": 2016},
    "ts_def_reb_pg":     {"patterns": ["Rebounds _Defensive_ Per Game",
                                       "Defensive Rebounds per Game"],
                          "stat_col": "RPG",        "min_year": 2016},
    "ts_bench_pts_pg":   {"patterns": ["Bench Points per game"],
                          "stat_col": "PPG",        "min_year": 2016},
    "ts_fastbreak_pts_pg":{"patterns": ["Fastbreak Points"],
                           "stat_col": "PPG",       "min_year": 2016},
    "ts_to_forced_pg":   {"patterns": ["Turnovers Forced Per Game", "Turnovers Forced"],
                          "stat_col": "Avg",        "min_year": 2016},
}

# Files to skip — matched against the stat name (filename prefix before '__')
# These are redundant per README guidance
TEAM_STATS_SKIP = [
    "Assist_Turnover Ratio",
    "Fewest",
    "Final Opp",
    "Free Throw Attempts Per Game",
    "Free Throws Made Per Game",
    "Scoring Defense",
    "Scoring Offense",
    "Total",
    "Three Pointers Per Game",
    "Three Point Attempts Per Game",
    "Three-Point Field Goals Per Game",
    "Turnovers Per Game",
    "Winning Percentage",
    "Won-Lost",
]

# ── Labeling ──────────────────────────────────────────────────────────────────
# For the pairwise approach each observation is a matchup between two Final 4
# teams.  Features are DIFFERENCES (team_A - team_B).  Target = 1 if A wins.
LABEL_MODE = "pairwise"          # or "rank4" (ordinal 1-4) or "binary" (1 winner)

# Finish → ordinal label used in rank4 mode
FINISH_TO_LABEL = {
    "Champion":     4,
    "Runner-Up":    3,
    "Third Place":  2,   # both semis losers get this (no true 3/4 game since 1981)
    "Fourth Place": 2,
}

# ── Model hyperparameters ─────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        4,          # shallow → reduces overfit on small N
    "min_child_samples": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "class_weight":     "balanced",
    "random_state":     42,
    "verbose":          -1,
}

RF_PARAMS = {
    "n_estimators":         1000,
    "max_features":         1,      # de Prado MDI: one random feature per level
    "criterion":            "entropy",
    "class_weight":         "balanced_subsample",
    "min_weight_fraction_leaf": 0.02,
    "random_state":         42,
    "n_jobs":               -1,
}
