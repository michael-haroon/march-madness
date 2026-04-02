"""
market_features.py
------------------
Kalshi prediction market microstructure features for March Madness.

Data:
  data/market_data_store/historical-endpoint/year=2025/ticker=KXMARMAD-25-*/
  data/market_data_store/markets-endpoint/year=2025|2026/ticker=KXMARMAD-2X-*/

Schema per parquet file:
  trade_id, yes_price_dollars (str→float), no_price_dollars (str→float),
  count_fp (str→float), taker_side ('yes'/'no'), created_time (datetime64[us, UTC])

NOTE: Only 2 years of data (2025–2026). Do NOT run MDI/MDA on these features.
Use SFI only, or treat mkt_vwap as the primary model in a meta-labeling setup.

Kalshi has markets for all 68 tournament teams, not just Final Four.
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Trade data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_kalshi_trades(data_dir: str) -> pd.DataFrame:
    """
    Load all Kalshi trades from both endpoints.

    Sources:
      data/market_data_store/historical-endpoint/year=2025/ticker=*/
      data/market_data_store/markets-endpoint/year=2025|2026/ticker=*/

    Returns a DataFrame with columns:
      trade_id, yes_price, no_price, count_fp, taker_side,
      created_time (UTC datetime), team, year, ticker
    """
    market_dir = os.path.join(data_dir, "market_data_store")
    if not os.path.isdir(market_dir):
        warnings.warn(f"market_data_store not found at {market_dir}")
        return pd.DataFrame()

    # Load ticker → team name mapping
    mapping_path = os.path.join(market_dir, "kalshi_name_maping.csv")
    if not os.path.exists(mapping_path):
        warnings.warn("kalshi_name_maping.csv not found")
        ticker_to_team = {}
        ticker_to_year = {}
    else:
        mapping = pd.read_csv(mapping_path)
        ticker_to_team = dict(zip(mapping["ticker"], mapping["team_name"]))
        # Derive year from ticker: KXMARMAD-25-* → 2025, KXMARMAD-26-* → 2026
        def _ticker_year(ticker):
            m = re.search(r"KXMARMAD-(\d+)-", str(ticker))
            if m:
                yr = int(m.group(1))
                return 2000 + yr
            return np.nan
        ticker_to_year = {t: _ticker_year(t) for t in mapping["ticker"]}

    # Scan both endpoints
    endpoints = [
        os.path.join(market_dir, "historical-endpoint"),
        os.path.join(market_dir, "markets-endpoint"),
    ]

    all_frames = []
    seen_trade_ids: set[str] = set()

    for endpoint in endpoints:
        if not os.path.isdir(endpoint):
            continue
        parquet_files = glob.glob(
            os.path.join(endpoint, "**", "*.parquet"), recursive=True
        )
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
            except Exception as e:
                warnings.warn(f"Could not read {pf}: {e}")
                continue

            # Deduplicate by trade_id
            if "trade_id" in df.columns:
                df = df[~df["trade_id"].isin(seen_trade_ids)]
                seen_trade_ids.update(df["trade_id"].tolist())

            # Extract ticker from directory path or column
            ticker = None
            if "ticker" in df.columns:
                ticker = df["ticker"].iloc[0] if len(df) > 0 else None
            if ticker is None:
                m = re.search(r"ticker=([^/\\]+)", pf)
                if m:
                    ticker = m.group(1)

            if ticker is None:
                continue

            from feature_pipeline.data_loader import normalise_team
            raw_team = ticker_to_team.get(ticker, ticker)
            df["ticker"]    = ticker
            df["team"]      = normalise_team(raw_team) if raw_team else raw_team
            df["year"]      = ticker_to_year.get(ticker, np.nan)

            all_frames.append(df)

    if not all_frames:
        warnings.warn("No Kalshi trade data loaded")
        return pd.DataFrame()

    trades = pd.concat(all_frames, ignore_index=True)

    # Normalise columns
    trades["yes_price"] = pd.to_numeric(trades["yes_price_dollars"], errors="coerce")
    trades["no_price"]  = pd.to_numeric(trades["no_price_dollars"],  errors="coerce")
    trades["count_fp"]  = pd.to_numeric(trades["count_fp"],          errors="coerce")

    # Ensure datetime with UTC tz
    if "created_time" in trades.columns:
        trades["created_time"] = pd.to_datetime(trades["created_time"], utc=True)

    trades = trades.dropna(subset=["yes_price", "count_fp", "year"])
    trades["year"] = trades["year"].astype(int)

    print(f"  Loaded {len(trades):,} Kalshi trades "
          f"({trades['year'].value_counts().to_dict()})")
    return trades


# ─────────────────────────────────────────────────────────────────────────────
#  Microstructure feature computation
# ─────────────────────────────────────────────────────────────────────────────

# Map year → Final Four tip-off time (UTC).  Trades AT or AFTER this time are
# post-game-start and must be excluded to avoid lookahead (DayNum >= 152).
# Saturday night tip-offs are typically ~9 PM ET = 01:00 UTC Sunday.
_FF_TIPOFF_UTC: dict[int, str] = {
    2025: "2025-04-05 01:00:00+00:00",   # Final Four Sat Apr 5 2025, ~9 PM ET
    2026: "2026-04-04 01:00:00+00:00",   # Final Four Sat Apr 5 2026, ~9 PM ET
}


def compute_market_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team market microstructure features from trade data.

    Features computed for each (year, team):
      mkt_vwap         – Volume-weighted average yes_price (best market probability estimate)
      mkt_last_price   – Last yes_price before cutoff
      mkt_ofi          – Order flow imbalance (yes_taker_vol - no_taker_vol) / total_vol
      mkt_momentum     – (last 24h VWAP - prior 7d VWAP) / prior 7d VWAP
      mkt_trade_count  – Total trades (liquidity proxy)
      mkt_volatility   – Std of yes_price (price uncertainty)
      mkt_price_range  – Max - min yes_price (full range)

    Trades on or after the Final Four tip-off (DayNum >= 152) are excluded to
    prevent lookahead.  Cutoff times are defined in _FF_TIPOFF_UTC; years without
    a listed cutoff keep all trades (with a warning).

    Returns: one row per (year, team).
    """
    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades = trades.sort_values(["year", "team", "created_time"])

    # Apply per-year pre-Final-Four cutoff
    filtered_parts = []
    for yr, grp in trades.groupby("year"):
        if yr in _FF_TIPOFF_UTC:
            cutoff = pd.Timestamp(_FF_TIPOFF_UTC[yr])
            n_before = len(grp)
            grp = grp[grp["created_time"] < cutoff]
            n_after = len(grp)
            if n_before != n_after:
                import warnings
                warnings.warn(
                    f"market_features: dropped {n_before - n_after:,} trades at/after "
                    f"Final Four tip-off for year={yr} (DayNum>=152 cutoff)"
                )
        else:
            import warnings
            warnings.warn(
                f"market_features: no Final Four tip-off date defined for year={yr}; "
                "using all trades (potential lookahead if tournament is complete)"
            )
        filtered_parts.append(grp)
    trades = pd.concat(filtered_parts, ignore_index=True)

    records = []
    for (year, team), grp in trades.groupby(["year", "team"]):
        if len(grp) == 0:
            continue

        total_vol = grp["count_fp"].sum()
        yes_vol   = grp[grp["taker_side"] == "yes"]["count_fp"].sum()
        no_vol    = grp[grp["taker_side"] == "no"]["count_fp"].sum()

        # VWAP
        vwap = (grp["yes_price"] * grp["count_fp"]).sum() / total_vol if total_vol > 0 else np.nan

        # Last price
        last_price = grp.iloc[-1]["yes_price"] if len(grp) > 0 else np.nan

        # OFI: (yes_buyer_vol - no_buyer_vol) / total_vol
        ofi = (yes_vol - no_vol) / total_vol if total_vol > 0 else 0.0

        # Momentum: last 24h VWAP vs prior 7d VWAP
        latest_time = grp["created_time"].max()
        last_24h = grp[grp["created_time"] >= latest_time - pd.Timedelta("1D")]
        prior_7d = grp[
            (grp["created_time"] < latest_time - pd.Timedelta("1D")) &
            (grp["created_time"] >= latest_time - pd.Timedelta("8D"))
        ]
        def _vwap(sub):
            v = sub["count_fp"].sum()
            return (sub["yes_price"] * sub["count_fp"]).sum() / v if v > 0 else np.nan
        vwap_24h = _vwap(last_24h)
        vwap_7d  = _vwap(prior_7d)
        momentum = (vwap_24h - vwap_7d) / vwap_7d if (vwap_7d and vwap_7d > 0) else np.nan

        # Volatility and range (last 7d)
        recent = grp[grp["created_time"] >= latest_time - pd.Timedelta("7D")]
        volatility  = recent["yes_price"].std() if len(recent) > 1 else 0.0
        price_range = recent["yes_price"].max() - recent["yes_price"].min() if len(recent) > 0 else 0.0

        records.append({
            "year":            year,
            "team":            team,
            "mkt_vwap":        round(vwap, 4) if vwap is not None else np.nan,
            "mkt_last_price":  round(last_price, 4),
            "mkt_ofi":         round(ofi, 4),
            "mkt_momentum":    round(momentum, 4) if momentum is not None else np.nan,
            "mkt_trade_count": len(grp),
            "mkt_volatility":  round(float(volatility), 4),
            "mkt_price_range": round(float(price_range), 4),
        })

    result = pd.DataFrame(records)
    print(f"  Computed market features for {len(result)} (year, team) pairs")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Kalshi-as-primary-model: implied probabilities for Final Four teams
# ─────────────────────────────────────────────────────────────────────────────

def get_ff_market_probs(mkt_features: pd.DataFrame,
                        ff_teams: pd.DataFrame) -> pd.DataFrame:
    """
    For Final Four teams with market data, return normalised implied probabilities.

    ff_teams must have columns: year, team (only FF teams).
    Joins on (year, team) using mkt_vwap as the market's implied probability.
    Normalises within each year so FF probs sum to 1.

    Returns ff_teams with mkt_vwap and mkt_implied_prob added.
    """
    result = ff_teams.merge(
        mkt_features[["year", "team", "mkt_vwap", "mkt_ofi", "mkt_trade_count"]],
        on=["year", "team"],
        how="left",
    )

    # Normalise VWAP within each year (FF teams only)
    for year, grp in result.groupby("year"):
        total = grp["mkt_vwap"].sum()
        if total > 0:
            result.loc[grp.index, "mkt_implied_prob"] = grp["mkt_vwap"] / total

    return result
