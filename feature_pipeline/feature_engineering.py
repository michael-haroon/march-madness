"""
feature_engineering.py
-----------------------
Translates de Prado's AFML / ML for Asset Managers concepts into
basketball-domain features.

Concept map:
  Finance → Basketball
  ─────────────────────────────────────────────────────────
  Price series           → Season win-rate trajectory
  Order flow imbalance   → Q1 win rate (signal quality)
  Entropy (LZ)           → Entropy of recent win/loss sequence
  CUSUM filter           → Detect momentum shift in win rate
  Fracdiff              → Year-over-year rating change with memory
  Triple barrier label   → Championship outcome (upper/lower/vertical)
  Meta-label             → Did predicted favourite actually win?
  Sample weight          → Recent years weight more than older ones
  MDI / MDA / SFI / CFI  → Feature importance on Final Four frame
"""

import re
import numpy as np
import pandas as pd
from itertools import combinations
from feature_pipeline.config import MISSING_STRATEGY


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Entropy of win/loss sequence  (de Prado Ch.18)
# ─────────────────────────────────────────────────────────────────────────────

def lz_entropy(binary_string: str) -> float:
    """
    Lempel-Ziv complexity estimate of a binary string.
    Encodes the information content of a win/loss sequence.
    A team on a clean winning streak → low entropy.
    A team alternating wins/losses  → high entropy.
    """
    s = binary_string.replace(" ", "")
    if len(s) < 2:
        return np.nan
    i, k, c = 0, 1, 1
    while k + c <= len(s):
        if s[k: k + c] in s[i: k]:
            c += 1
        else:
            i = k
            k += c
            c = 1
    n = len(s)
    # Normalise so it is in [0, 1]
    return (c * np.log2(n)) / n if n > 1 else np.nan


def win_sequence_entropy(wins: float, losses: float) -> float:
    """
    Approximates the LZ entropy from win/loss counts alone.
    We encode wins as '1' and losses as '0', alternating perfectly
    as the worst case, then mix.  A quick but robust proxy when
    we don't have game-by-game sequences.
    """
    try:
        wins  = int(wins)  if wins is not None and not (isinstance(wins, float) and np.isnan(wins))  else 0
        losses= int(losses) if losses is not None and not (isinstance(losses, float) and np.isnan(losses)) else 0
    except (ValueError, TypeError):
        return np.nan
    total = wins + losses
    if total < 2:
        return np.nan
    p = wins / total
    if p == 0 or p == 1:
        return 0.0          # pure streak → zero entropy
    # Shannon entropy (normalised to [0,1])
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ─────────────────────────────────────────────────────────────────────────────
#  2.  CUSUM  –  detect momentum shift in win rate  (de Prado Ch.2 & Ch.17)
# ─────────────────────────────────────────────────────────────────────────────

def cusum_peak(win_pct_series: pd.Series, h: float = 0.05) -> float:
    """
    Symmetric CUSUM filter applied to a team's win-rate time series across years.
    Returns the maximum absolute cumulative deviation above threshold h.
    This is a proxy for "has this team's momentum shifted significantly?"

    In practice:  feed in a Series of [year0_win_pct, year1_win_pct, ...]
    for a single team, ordered chronologically.

    Returns a scalar: the peak CUSUM statistic (magnitude of the largest shift).
    """
    s_pos = 0.0
    s_neg = 0.0
    peak  = 0.0
    y_prev = win_pct_series.iloc[0] if len(win_pct_series) > 0 else 0.5

    for y_t in win_pct_series:
        delta = y_t - y_prev
        s_pos = max(0.0, s_pos + delta)
        s_neg = min(0.0, s_neg + delta)
        peak  = max(peak, abs(s_pos), abs(s_neg))
        if abs(s_pos) >= h:
            s_pos = 0.0
        if abs(s_neg) >= h:
            s_neg = 0.0
        y_prev = y_t

    return peak


def add_cusum_feature(df: pd.DataFrame,
                      win_pct_col: str = "overall_win_pct",
                      h: float = 0.05) -> pd.DataFrame:
    """
    For each team, compute rolling CUSUM peak over its win-pct history.
    Returns df with new column 'cusum_peak_{win_pct_col}'.
    """
    df = df.copy()
    out_col = f"cusum_peak"
    df[out_col] = np.nan

    for team, grp in df.groupby("team"):
        grp_sorted = grp.sort_values("year")
        series = grp_sorted[win_pct_col].dropna()
        if len(series) < 2:
            continue
        peak = cusum_peak(series, h=h)
        df.loc[grp_sorted.index, out_col] = peak   # same value for all years of this team

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Year-over-year rank change  (inspired by fracdiff – de Prado Ch.5)
# ─────────────────────────────────────────────────────────────────────────────

def add_yoy_rank_change(df: pd.DataFrame,
                        rank_cols: list = None) -> pd.DataFrame:
    """
    Year-over-year change in each ranking metric, with exponential decay weight
    to give more importance to recent change than older change.
    Analogous to fractional differentiation: preserves some long-run memory
    while also capturing short-run momentum.

    Adds columns: {col}_yoy_change, {col}_yoy_pct_change for each rank_col.
    """
    if rank_cols is None:
        rank_cols = [c for c in df.columns
                     if c in ["net_rank", "kpi", "sor", "bpi", "pom", "sag"]]

    df = df.copy().sort_values(["team", "year"])

    for col in rank_cols:
        if col not in df.columns:
            continue
        df[f"{col}_yoy"] = (
            df.groupby("team")[col]
              .transform(lambda x: x.diff())   # lower rank # = improvement
        )
        # Sign: negative means improved (e.g. 5 → 2 = diff of -3 = got better)
        # Flip so positive = improvement
        df[f"{col}_yoy"] = -df[f"{col}_yoy"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Quadrant quality features  (de Prado: "informative features" Ch.19)
# ─────────────────────────────────────────────────────────────────────────────

def parse_quad_record(s) -> tuple:
    """'19-2' → (19, 2, 0.905).  Handles NaN."""
    if not isinstance(s, str):
        return np.nan, np.nan, np.nan
    m = re.match(r"(\d+)-(\d+)", s.strip())
    if m:
        w, l = int(m.group(1)), int(m.group(2))
        t = w + l
        return float(w), float(l), w / t if t > 0 else np.nan
    return np.nan, np.nan, np.nan


def add_quadrant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse quadrant W-L strings into numeric win %, and compute derived features:
    - q1_win_pct:  quality of wins (hardest games)
    - q1_loss_rate: vulnerability
    - quad_score:  weighted composite  (Q1 wins are worth more)
    - resume_score: Q1W×4 + Q2W×2 - Q1L×3 - Q2L  (common analyst formula)
    """
    df = df.copy()

    quad_map = {
        "q1_overall": "q1",
        "q2_overall": "q2",
        "q3_overall": "q3",
        "q4_overall": "q4",
    }

    for raw_col, prefix in quad_map.items():
        if raw_col not in df.columns:
            continue
        parsed = df[raw_col].apply(parse_quad_record)
        df[f"{prefix}_wins"]    = parsed.apply(lambda x: x[0])
        df[f"{prefix}_losses"]  = parsed.apply(lambda x: x[1])
        df[f"{prefix}_win_pct"] = parsed.apply(lambda x: x[2])

    # ── Composite resume score ────────────────────────────────────────────
    # Weights: Q1 win=+4, Q2 win=+2, Q3 win=+1, Q4 win=0
    #           Q1 loss=-3, Q2 loss=-2, Q3 loss=-1, Q4 loss=-0.5
    def _resume(row):
        try:
            return (
                  4 * (row.get("q1_wins",  0) or 0)
                + 2 * (row.get("q2_wins",  0) or 0)
                + 1 * (row.get("q3_wins",  0) or 0)
                - 3 * (row.get("q1_losses",0) or 0)
                - 2 * (row.get("q2_losses",0) or 0)
                - 1 * (row.get("q3_losses",0) or 0)
                - 0.5*(row.get("q4_losses",0) or 0)
            )
        except Exception:
            return np.nan

    df["resume_score"] = df.apply(_resume, axis=1)

    # Q1 win % is the single best proxy for tournament readiness
    if "q1_win_pct" in df.columns:
        df["q1_win_pct"] = pd.to_numeric(df["q1_win_pct"], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Consensus rank  (averaging predictive metrics)
# ─────────────────────────────────────────────────────────────────────────────

def add_consensus_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average rank across available predictive metrics.
    Lower = better.  Handles NaN by averaging what is present.
    Also adds rank_spread (std across metrics) as a confidence signal.
    """
    df = df.copy()
    pred_cols = [c for c in ["bpi", "pom", "sag"] if c in df.columns]
    rb_cols   = [c for c in ["kpi", "sor"]          if c in df.columns]
    all_metric_cols = pred_cols + rb_cols + (["net_rank"] if "net_rank" in df.columns else [])

    if all_metric_cols:
        df["consensus_rank"] = df[all_metric_cols].mean(axis=1, skipna=True)
        df["rank_spread"]    = df[all_metric_cols].std(axis=1, skipna=True)

    # Predictive-only consensus
    if pred_cols:
        df["pred_consensus"] = df[pred_cols].mean(axis=1, skipna=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  6.  Award flag aggregation
# ─────────────────────────────────────────────────────────────────────────────

def add_award_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Sum all has_*_award columns into a single 'total_awards' feature."""
    df = df.copy()
    award_cols = [c for c in df.columns if c.startswith("has_")]
    if award_cols:
        df["total_awards"] = df[award_cols].sum(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  7.  Time-decay sample weights  (de Prado Ch.4)
# ─────────────────────────────────────────────────────────────────────────────

def time_decay_weights(years: pd.Series, c: float = 0.5) -> pd.Series:
    """
    Assign higher weight to more recent tournament years.
    c=1 → no decay (all equal).
    c=0 → weights converge to 0 for the oldest year.
    c<0 → oldest fraction (|c|) gets zero weight.

    Returns a Series of weights summing to len(years).
    """
    unique_years = sorted(years.unique())
    n = len(unique_years)
    # Linear map from oldest→0 to newest→1
    year_pos = {y: i / (n - 1) if n > 1 else 1.0 for i, y in enumerate(unique_years)}
    raw = years.map(year_pos)

    if c >= 0:
        # d[x] = c + (1-c)*x  so oldest→c, newest→1
        weights = c + (1 - c) * raw
    else:
        # Oldest |c| fraction gets 0
        threshold = -c
        weights = (raw - threshold).clip(lower=0)
        weights = weights / weights.max() if weights.max() > 0 else weights

    # Rescale to sum to n (sklearn convention: default weight per sample = 1)
    total = weights.sum()
    if total > 0:
        weights = weights * n / total
    return weights


# ─────────────────────────────────────────────────────────────────────────────
#  8.  Missing data handling  (de Prado: indicator + median fill)
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame,
                   strategy: dict = None) -> pd.DataFrame:
    """
    For each feature in strategy:
      'indicator' → add _missing binary column, fill NaN with column median
      'zero'      → fill NaN with 0
      'drop'      → drop rows where NaN (use sparingly)
    """
    if strategy is None:
        strategy = MISSING_STRATEGY

    df = df.copy()
    for col, method in strategy.items():
        if col not in df.columns:
            continue
        is_null = df[col].isna()
        if not is_null.any():
            continue

        if method == "indicator":
            # Coerce to numeric first (strings like "1" from CSV loading)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            is_null = df[col].isna()
            df[f"{col}_missing"] = is_null.astype(int)
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)

        elif method == "zero":
            df[col] = df[col].fillna(0)

        elif method == "drop":
            df = df[~is_null]

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  9.  Pairwise feature construction  (for Bradley-Terry model)
# ─────────────────────────────────────────────────────────────────────────────

def build_pairwise_frame(df: pd.DataFrame,
                         feature_cols: list,
                         label_col: str = "finish_rank") -> pd.DataFrame:
    """
    For the Final Four frame (in_final_four == 1), construct one row per
    matchup pair (A vs B).  Features = A_metric - B_metric (rank difference).
    Target = 1 if A finished better (lower finish_rank) than B.

    This converts a 4-team ranking problem into ~6 binary comparisons per year.
    Over 20 years → ~120 pairwise observations: much more tractable.

    Returns a DataFrame with:
      year, team_a, team_b, feature diffs, target (a_wins)
    """
    ff = df[df["in_final_four"] == 1].copy()
    rows = []

    for year, grp in ff.groupby("year"):
        teams = grp.set_index("team")
        team_list = list(teams.index)

        for a, b in combinations(team_list, 2):
            row = {"year": year, "team_a": a, "team_b": b}

            # Feature differences  (lower rank = better, so A_rank - B_rank:
            # negative means A is ranked better)
            for col in feature_cols:
                if col in teams.columns:
                    row[f"diff_{col}"] = (
                        teams.loc[a, col] - teams.loc[b, col]
                        if (pd.notna(teams.loc[a, col]) and pd.notna(teams.loc[b, col]))
                        else np.nan
                    )

            # Target: did A finish better than B?
            rank_a = teams.loc[a, label_col] if label_col in teams.columns else np.nan
            rank_b = teams.loc[b, label_col] if label_col in teams.columns else np.nan
            if pd.notna(rank_a) and pd.notna(rank_b):
                row["a_wins"] = int(rank_a < rank_b)  # lower rank = better finish
            else:
                row["a_wins"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  10.  Master feature builder
# ─────────────────────────────────────────────────────────────────────────────

def _log_step(label: str, before_cols: set, df: pd.DataFrame) -> None:
    added   = sorted(set(df.columns) - before_cols)
    removed = sorted(before_cols - set(df.columns))
    parts = []
    if added:
        parts.append(f"+{len(added)} [{', '.join(added)}]")
    if removed:
        parts.append(f"-{len(removed)} [{', '.join(removed)}]")
    change = "  (no column change)" if not parts else "  " + " | ".join(parts)
    print(f"  [{label}] shape → {df.shape}{change}")


def build_features(df: pd.DataFrame,
                   run_pca: bool = True,
                   run_reconcile: bool = True,
                   run_redundancy_audit: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Run all feature engineering steps on the master DataFrame.
    Returns an enriched DataFrame ready for feature importance analysis.

    Args:
        run_pca:              Replace raw ts_* columns with PCA components.
        run_reconcile:        Drop ts_* columns redundant with Kaggle equivalents.
        run_redundancy_audit: Print ONC cluster report for cross-feature redundancy.
    """
    print(f"Starting feature engineering. Shape: {df.shape}")

    prev = set(df.columns)
    print("Building quadrant features...")
    df = add_quadrant_features(df)
    _log_step("quadrant", prev, df)

    prev = set(df.columns)
    print("Building YoY rank changes (fracdiff proxy)...")
    df = add_yoy_rank_change(df)
    _log_step("yoy_rank", prev, df)

    prev = set(df.columns)
    print("Building CUSUM momentum feature...")
    if "overall_win_pct" in df.columns:
        df = add_cusum_feature(df, win_pct_col="overall_win_pct")
    _log_step("cusum", prev, df)

    prev = set(df.columns)
    print("Building consensus rank...")
    df = add_consensus_rank(df)
    _log_step("consensus", prev, df)

    prev = set(df.columns)
    print("Building win-sequence entropy...")
    if "overall_wins" in df.columns and "overall_losses" in df.columns:
        df["win_entropy"] = df.apply(
            lambda r: win_sequence_entropy(r["overall_wins"], r["overall_losses"]),
            axis=1,
        )
    _log_step("entropy", prev, df)

    prev = set(df.columns)
    print("Building award composite...")
    df = add_award_composite(df)
    _log_step("awards", prev, df)

    # Cross-source reconciliation: drop ts_* cols redundant with Kaggle
    has_ts = any(c.startswith("ts_") for c in df.columns)
    has_kg = any(c.startswith("kg_") for c in df.columns)
    if run_reconcile and has_ts and has_kg:
        prev = set(df.columns)
        print("Cross-source reconciliation (dropping redundant ts_ columns)...")
        df, _reconcile_report = reconcile_cross_source(df, verbose=True)
        _log_step("reconcile", prev, df)

    # PCA reduction of remaining ts_* features
    if run_pca and any(c.startswith("ts_") for c in df.columns):
        prev = set(df.columns)
        print("PCA reduction of team-stats features...")
        df, _loadings = reduce_team_stats_pca(df, verbose=True)
        _log_step("pca", prev, df)

    # Optional: full redundancy audit
    if run_redundancy_audit:
        print("\nRunning feature redundancy audit (ONC clustering)...")
        _audit = audit_feature_redundancy(df, verbose=True)

    prev = set(df.columns)
    print("Handling missing values...")
    df = handle_missing(df)
    _log_step("handle_missing", prev, df)

    prev = set(df.columns)
    print("Building time-decay weights...")
    df["sample_weight"] = time_decay_weights(df["year"], c=0.3)
    _log_step("sample_weights", prev, df)

    print(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: list of engineered feature columns
# ─────────────────────────────────────────────────────────────────────────────

CORE_RANK_FEATURES = [
    "net_rank", "kpi", "sor", "bpi", "pom", "sag",
    "consensus_rank", "pred_consensus", "rank_spread",
]
SOS_FEATURES = [
    "net_sos", "rpi_sos", "net_nc_sos", "rpi_nc_sos",
    "avg_net_wins", "avg_net_losses",
]
RECORD_FEATURES = [
    "overall_win_pct", "nc_win_pct", "road_win_pct", "conf_win_pct",
    "q1_win_pct", "resume_score",
]
DEPRADO_FEATURES = [
    "win_entropy", "cusum_peak",
    "net_rank_yoy", "kpi_yoy", "sor_yoy", "bpi_yoy", "pom_yoy",
]
AWARD_FEATURES = ["total_awards"]

# New feature groups from data integration
KAGGLE_FEATURES = [
    "kg_fg_pct", "kg_fg3_pct", "kg_ft_pct", "kg_efg_pct",
    "kg_off_reb_pg", "kg_def_reb_pg", "kg_ast_pg", "kg_to_pg",
    "kg_stl_pg", "kg_blk_pg", "kg_scoring_margin",
    "kg_margin_last3", "kg_margin_last5", "kg_margin_last10", "kg_margin_last15", "kg_margin_last20",
    "kg_days_since_last_game",
    "kg_road_win_pct", "kg_opp_fg_pct", "kg_opp_efg_pct", "kg_ast_to_ratio",
]
TOURNEY_PATH_FEATURES = [
    "kg_tourney_avg_margin", "kg_tourney_worst_margin",
    "kg_tourney_fg_pct", "kg_tourney_opp_fg_pct", "kg_rounds_survived",
    "kg_bpi_at_finals",   # ESPN BPI rank, latest pre-Final Four (DayNum<152); 2008–2026
]
MASSEY_FEATURES = [
    "massey_POM", "massey_SAG", "massey_RPI",
    "massey_MOR", "massey_WLK", "massey_DOL", "massey_COL",
]
TEAM_STATS_FEATURES = [
    "ts_fg_pct", "ts_fg_pct_def", "ts_ft_pct", "ts_three_pct",
    "ts_three_pct_def", "ts_scoring_margin", "ts_rebound_margin",
    "ts_turnover_margin", "ts_assists_pg", "ts_steals_pg",
    "ts_blocks_pg", "ts_fouls_pg", "ts_reb_pg",
    "ts_efg_pct", "ts_off_reb_pg", "ts_def_reb_pg",
    "ts_bench_pts_pg", "ts_fastbreak_pts_pg", "ts_to_forced_pg",
]
TEAM_STATS_PC_FEATURES = [f"ts_pc{i}" for i in range(1, 6)]
MARKET_FEATURES = [
    "mkt_vwap", "mkt_last_price", "mkt_ofi",
    "mkt_momentum", "mkt_trade_count", "mkt_volatility", "mkt_price_range",
]

# Tier 1 features: regular season only (DayNum ≤ 132), safe for all-round survival prediction
TIER1_FEATURES = (
    CORE_RANK_FEATURES + SOS_FEATURES + RECORD_FEATURES +
    DEPRADO_FEATURES + AWARD_FEATURES + KAGGLE_FEATURES + MASSEY_FEATURES
)

# Tier 2 features: all features including tournament path (safe only for Final Four prediction)
ALL_FEATURES = TIER1_FEATURES + TOURNEY_PATH_FEATURES + TEAM_STATS_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
#  11.  Cross-source reconciliation
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_cross_source(df: pd.DataFrame,
                           verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    For stats available from both Kaggle game aggregation and team-stats files,
    always prefer the Kaggle version — it covers all years with consistent
    game-level aggregation. The ts_ files are spot data for a subset of years
    and are redundant wherever kg_ exists.

    Correlation is printed as a sanity check only, not used to gate decisions.

    Returns:
        df_clean: DataFrame with ts_ columns dropped wherever kg_ equivalent exists
        report:   dict of {(kg_col, ts_col): {'r': float, 'action': str}}
    """
    # Candidate pairs: same underlying stat from both sources — always drop ts_
    candidate_pairs = [
        ("kg_fg_pct",        "ts_fg_pct"),
        ("kg_opp_fg_pct",    "ts_fg_pct_def"),
        ("kg_ft_pct",        "ts_ft_pct"),
        ("kg_fg3_pct",       "ts_three_pct"),
        ("kg_scoring_margin","ts_scoring_margin"),
        ("kg_efg_pct",       "ts_efg_pct"),
        ("kg_off_reb_pg",    "ts_off_reb_pg"),
        ("kg_def_reb_pg",    "ts_def_reb_pg"),
        ("kg_ast_pg",        "ts_assists_pg"),
        ("kg_stl_pg",        "ts_steals_pg"),
        ("kg_blk_pg",        "ts_blocks_pg"),
    ]

    report = {}
    cols_to_drop = []

    for kg_col, ts_col in candidate_pairs:
        if kg_col not in df.columns or ts_col not in df.columns:
            continue

        both = df[[kg_col, ts_col]].dropna()
        r = both[kg_col].corr(both[ts_col]) if len(both) >= 30 else np.nan

        cols_to_drop.append(ts_col)
        report[(kg_col, ts_col)] = {"r": round(r, 3) if not np.isnan(r) else np.nan,
                                    "n": len(both), "action": "drop_ts"}

        if verbose:
            r_str = f"r={r:.3f}" if not np.isnan(r) else "r=n/a"
            print(f"  {kg_col:28s} vs {ts_col:28s}  {r_str}  → drop ts (always prefer Kaggle)")

    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    if verbose:
        print(f"\n  Dropped {len(cols_to_drop)} redundant ts_ columns: {cols_to_drop}")

    return df_clean, report


# ─────────────────────────────────────────────────────────────────────────────
#  12.  Feature redundancy audit  (correlation + ONC clustering)
# ─────────────────────────────────────────────────────────────────────────────

def audit_feature_redundancy(df: pd.DataFrame,
                             feature_cols: list = None,
                             threshold: float = 0.85,
                             verbose: bool = True) -> dict:
    """
    Compute correlation matrix, flag highly-correlated pairs, group via ONC.

    Returns:
        {
          'corr': pd.DataFrame,
          'high_corr_pairs': list of (col_a, col_b, r),
          'clusters': dict of cluster_id → [feature_names],
          'keep_list': list of recommended features (one per cluster),
        }
    """
    from feature_pipeline.feature_importance import onc_cluster

    if feature_cols is None:
        # Use all numeric columns that are not metadata/labels
        exclude = {"year", "TeamID", "seed_num", "furthest_round", "finish_rank",
                   "champion_flag", "in_final_four", "made_ff", "champion",
                   "survived_r1", "survived_r2", "survived_s16", "survived_e8",
                   "in_champ", "sample_weight"}
        feature_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in exclude and not c.endswith("_missing")
        ]

    sub = df[feature_cols].dropna(thresh=int(0.5 * len(df)))
    sub = sub.loc[:, sub.notna().sum() > 30]  # need enough non-null values
    feature_cols = list(sub.columns)

    corr = sub.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i, col_a in enumerate(feature_cols):
        for col_b in feature_cols[i + 1:]:
            if col_a in corr.index and col_b in corr.index:
                r = corr.loc[col_a, col_b]
                if abs(r) >= threshold:
                    high_corr_pairs.append((col_a, col_b, round(r, 3)))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if verbose:
        print(f"\n  Features analysed: {len(feature_cols)}")
        print(f"  Highly correlated pairs (|r| ≥ {threshold}): {len(high_corr_pairs)}")
        for a, b, r in high_corr_pairs[:10]:
            print(f"    {a:30s} ↔ {b:30s}  r={r:+.3f}")
        if len(high_corr_pairs) > 10:
            print(f"    ... ({len(high_corr_pairs) - 10} more)")

    # ONC clustering
    clusters = onc_cluster(corr, max_clusters=max(2, len(feature_cols) // 3))

    # Keep-list: one feature per cluster (prefer non-missing, longer history)
    keep_list = []
    for cid, members in clusters.items():
        # Prefer Kaggle/team-sheet features over derived ones
        priority_order = [
            m for m in members if m.startswith("kg_") or m.startswith("massey_")
        ] + [
            m for m in members if not m.startswith(("kg_", "massey_", "ts_"))
        ] + [
            m for m in members if m.startswith("ts_")
        ]
        keep_list.append(priority_order[0] if priority_order else members[0])

    if verbose:
        print(f"\n  ONC clusters found: {len(clusters)}")
        for cid, members in sorted(clusters.items()):
            print(f"    Cluster {cid}: {members}")

    return {
        "corr":            corr,
        "high_corr_pairs": high_corr_pairs,
        "clusters":        clusters,
        "keep_list":       keep_list,
        "feature_cols":    feature_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  13.  PCA reduction of team-stats features
# ─────────────────────────────────────────────────────────────────────────────

def reduce_team_stats_pca(df: pd.DataFrame,
                          ts_cols: list = None,
                          variance_threshold: float = 0.90,
                          verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace ts_* raw features with PCA components.

    Standardises within each year (to handle era differences in basketball stats),
    then runs PCA retaining enough components to explain variance_threshold.

    Returns:
        df_with_pca: DataFrame with ts_pc1, ts_pc2, ... added (raw ts_* removed)
        loadings:    DataFrame of component loadings for interpretation
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if ts_cols is None:
        ts_cols = [c for c in TEAM_STATS_FEATURES if c in df.columns]

    if len(ts_cols) < 2:
        if verbose:
            print("  Not enough ts_ columns for PCA — skipping")
        return df, pd.DataFrame()

    df = df.copy()

    # Year-wise standardisation to remove era effects
    standardised_parts = []
    for year, grp in df.groupby("year"):
        sub = grp[ts_cols].copy()
        valid_cols = sub.columns[sub.notna().sum() >= 3].tolist()
        if not valid_cols:
            continue
        scaler = StandardScaler()
        scaled = scaler.fit_transform(sub[valid_cols].fillna(sub[valid_cols].median()))
        part = pd.DataFrame(scaled, index=grp.index, columns=valid_cols)
        standardised_parts.append(part)

    if not standardised_parts:
        return df, pd.DataFrame()

    X_scaled = pd.concat(standardised_parts).reindex(df.index).fillna(0)
    valid_ts = X_scaled.columns.tolist()

    # PCA
    pca = PCA()
    pca.fit(X_scaled.values)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_components = max(2, min(n_components, len(valid_ts)))

    pca_final = PCA(n_components=n_components)
    components = pca_final.fit_transform(X_scaled.values)

    pc_cols = [f"ts_pc{i+1}" for i in range(n_components)]
    for i, col in enumerate(pc_cols):
        df[col] = np.nan
        df.loc[X_scaled.index, col] = components[:, i]

    # Loadings matrix for interpretation
    loadings = pd.DataFrame(
        pca_final.components_.T,
        index=valid_ts,
        columns=pc_cols,
    )

    if verbose:
        print(f"\n  PCA on {len(valid_ts)} ts_ features → {n_components} components")
        print(f"  Variance explained: {cumvar[n_components-1]:.1%}")
        print(f"  Top loadings per component:")
        for pc in pc_cols:
            top = loadings[pc].abs().nlargest(3).index.tolist()
            print(f"    {pc}: {top}")

    # Remove raw ts_ columns that were included in PCA
    df = df.drop(columns=[c for c in valid_ts if c in df.columns], errors="ignore")

    return df, loadings
