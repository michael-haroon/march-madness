"""
Integrate NCAA Nitty-Gritty NET SOS data into MMasseyOrdinals.csv.

For each CSV in data/raw/nitty_gritty/csvs/{year}/:
  1. Parse the date from the filename (thru_games_MM_DD_YYYY*.csv)
  2. Compute Season + DayNum via feature_pipeline.season_utils
  3. Map team name → Kaggle TeamID via feature_pipeline.name_resolver
  4. Emit rows for NETSOS → SOS and NETNonConfSOS/NETNCSOS → NC_SOS
     into MMasseyOrdinals.csv

Team-name mapping uses the shared name_resolver (build_id_lookup +
resolve_team_id) with TEAM_NAME_MAP overrides from feature_pipeline.config.
This handles (AQ) suffixes, UConn → Connecticut, fuzzy matching, etc.

Usage:
  conda run -n tasty python -m feature_pipeline.scripts.integrate_nitty_gritty_sos
  conda run -n tasty python -m feature_pipeline.scripts.integrate_nitty_gritty_sos --write
  conda run -n tasty python -m feature_pipeline.scripts.integrate_nitty_gritty_sos --verbose
  conda run -n tasty python -m feature_pipeline.scripts.integrate_nitty_gritty_sos --log-file run.log
"""

import argparse
import glob
import logging
import os
import re
import sys
from datetime import datetime, date

import pandas as pd

from feature_pipeline.season_utils import build_season_table, get_season_and_daynum
from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id
from feature_pipeline.config import TEAM_NAME_MAP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NITTY_DIR = "data/raw/nitty_gritty/csvs"
MSEASONS  = "data/kaggle/MSeasons.csv"
MASSEY    = "data/kaggle/MMasseyOrdinals.csv"
KAGGLE    = "data/kaggle"

# Column name variants across seasons → canonical SystemName in MMasseyOrdinals
SOS_COL_MAP = {
    "NETSOS":         "SOS_D1",
    "NETNonConfSOS":  "SOS_NC",
    "NETNCSOS":       "SOS_NC",
}

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool, log_file: str | None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Filename date parsing
# ---------------------------------------------------------------------------

def parse_nitty_date(filename: str) -> date | None:
    """
    Parse calendar date from a nitty-gritty filename.

    Handles:
      thru_games_MM_DD_YYYY.csv
      thru_games_MM_DD_YYYY (Selections).csv
      thru_games_MM_DD_YYYY (Final).csv
      thru_games_initial.csv  → None (no date)
    """
    m = re.search(r"thru_games_(\d{2})_(\d{2})_(\d{4})", filename)
    if m:
        mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(yyyy, mm, dd)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = True, verbose: bool = False, log_file: str | None = None) -> None:
    _setup_logging(verbose, log_file)

    log.info("=" * 72)
    log.info("STEP 1  Load reference tables")
    log.info("=" * 72)
    season_table = build_season_table(MSEASONS)
    lookup = build_id_lookup(KAGGLE)
    log.info("Seasons available : %d", len(season_table))
    log.info("Name lookup size  : %d", len(lookup))

    log.info("=" * 72)
    log.info("STEP 2  Discover and parse nitty-gritty files")
    log.info("=" * 72)

    all_files = sorted(glob.glob(f"{NITTY_DIR}/**/*.csv", recursive=True))
    log.info("Total files found : %d", len(all_files))

    rows_out       = []
    skipped_files  = []
    unmapped_teams: dict[str, list[str]] = {}   # name → [filenames]
    file_stats     = []

    for fpath in all_files:
        fname = os.path.basename(fpath)

        # ── Parse date ───────────────────────────────────────────────────────
        file_date = parse_nitty_date(fname)
        if file_date is None:
            log.debug("SKIP  %s  — cannot parse date", fname)
            skipped_files.append((fname, "cannot parse date"))
            continue

        season, daynum = get_season_and_daynum(file_date, season_table)
        if season is None:
            log.warning("SKIP  %s  — date %s out of range for any season", fname, file_date)
            skipped_files.append((fname, f"date {file_date} out of range for any season"))
            continue

        log.debug("FILE  %s  → season=%d  daynum=%d", fname, season, daynum)

        # ── Load CSV ─────────────────────────────────────────────────────────
        df = pd.read_csv(fpath)
        if "Team" not in df.columns:
            log.warning("SKIP  %s  — no Team column", fname)
            skipped_files.append((fname, "no Team column"))
            continue

        # Which SOS columns are present?
        sos_cols = {col: sys_name
                    for col, sys_name in SOS_COL_MAP.items()
                    if col in df.columns}
        if not sos_cols:
            log.warning("SKIP  %s  — no SOS columns found (cols: %s)", fname, list(df.columns))
            skipped_files.append((fname, "no SOS columns found"))
            continue

        # Drop rows where all data columns are empty
        data_cols = [c for c in df.columns if c != "Team"]
        df = df[df[data_cols].notna().any(axis=1)].reset_index(drop=True)

        # ── Process rows ─────────────────────────────────────────────────────
        file_ok = file_skip = file_unmap = 0
        for _, row in df.iterrows():
            raw_name = str(row["Team"]).strip()
            if not raw_name or raw_name == "nan":
                log.debug("  SKIP blank team row in %s", fname)
                file_skip += 1
                continue

            team_id = resolve_team_id(raw_name, lookup, TEAM_NAME_MAP)
            if team_id is None:
                log.debug("  UNMAP  '%s'  in %s", raw_name, fname)
                unmapped_teams.setdefault(raw_name, []).append(fname)
                file_unmap += 1
                continue

            for col, sys_name in sos_cols.items():
                val = row[col]
                try:
                    rank = int(float(val))
                except (ValueError, TypeError):
                    continue
                if rank <= 0:
                    continue
                log.debug(
                    "  ROW  season=%d daynum=%d sys=%s team=%s(%d) rank=%d",
                    season, daynum, sys_name, raw_name, team_id, rank,
                )
                rows_out.append({
                    "Season":        season,
                    "RankingDayNum": daynum,
                    "SystemName":    sys_name,
                    "TeamID":        team_id,
                    "OrdinalRank":   rank,
                })
            file_ok += 1

        log.debug("  %s  ok=%d  skip=%d  unmap=%d", fname, file_ok, file_skip, file_unmap)
        file_stats.append((fname, season, daynum, file_ok, file_skip, file_unmap))

    log.info("Files skipped: %d", len(skipped_files))
    for fname, reason in skipped_files:
        log.info("  SKIP  %s  — %s", fname, reason)

    log.info("=" * 72)
    log.info("STEP 3  Mapping summary")
    log.info("=" * 72)
    total_ok    = sum(s[3] for s in file_stats)
    total_skip  = sum(s[4] for s in file_stats)
    total_unmap = sum(s[5] for s in file_stats)
    log.info("Team rows mapped OK        : %d", total_ok)
    log.info("Team rows skipped (no name): %d", total_skip)
    log.info("Team rows unmapped         : %d", total_unmap)

    if unmapped_teams:
        log.warning("Unmapped team names (%d):", len(unmapped_teams))
        for name, files in sorted(unmapped_teams.items()):
            log.warning("  '%s'  (first seen: %s)", name, files[0])

    log.info("=" * 72)
    log.info("STEP 4  Build output DataFrame")
    log.info("=" * 72)
    new_df = pd.DataFrame(rows_out)
    if new_df.empty:
        log.error("No rows to add. Exiting.")
        return

    log.info("Total rows to add  : %d", len(new_df))
    log.info("Systems            : %s", sorted(new_df["SystemName"].unique()))
    log.info("Season range       : %d–%d", new_df["Season"].min(), new_df["Season"].max())
    log.info("DayNum range       : %d–%d", new_df["RankingDayNum"].min(), new_df["RankingDayNum"].max())

    # Per-season/system breakdown
    breakdown = (
        new_df.groupby(["Season", "SystemName"])
        .size()
        .reset_index(name="rows")
        .sort_values(["Season", "SystemName"])
    )
    log.info("Per-season/system breakdown:\n%s", breakdown.to_string(index=False))

    n_sample = min(10, len(new_df))
    sample = new_df.sample(n_sample, random_state=42).sort_values(["Season", "RankingDayNum"])
    log.info("Sample (%d rows):\n%s", n_sample, sample.to_string(index=False))

    log.info("=" * 72)
    log.info("STEP 5  Check against existing MMasseyOrdinals")
    log.info("=" * 72)
    massey = pd.read_csv(MASSEY)
    log.info("Existing rows: %d", len(massey))
    for sys_name in sorted(new_df["SystemName"].unique()):
        n = len(massey[massey["SystemName"] == sys_name])
        log.info("Existing %s rows: %d", sys_name, n)

    # Dedup: skip rows already present
    existing_keys = massey[
        massey["SystemName"].isin(new_df["SystemName"].unique())
    ][["Season", "RankingDayNum", "SystemName", "TeamID"]].drop_duplicates()

    before = len(new_df)
    new_df = new_df.merge(
        existing_keys,
        on=["Season", "RankingDayNum", "SystemName", "TeamID"],
        how="left",
        indicator=True,
    )
    new_df = new_df[new_df["_merge"] == "left_only"].drop(columns="_merge")
    log.info("Rows after dedup: %d  (removed %d duplicates)", len(new_df), before - len(new_df))

    if dry_run:
        log.info("=" * 72)
        log.info("DRY RUN — no changes written.")
        log.info("Re-run with --write to update MMasseyOrdinals.csv")
        log.info("=" * 72)
        return

    log.info("=" * 72)
    log.info("STEP 6  Write to MMasseyOrdinals.csv")
    log.info("=" * 72)
    combined = pd.concat([massey, new_df], ignore_index=True)
    combined.to_csv(MASSEY, index=False)
    verify = pd.read_csv(MASSEY)
    log.info("Written: %d total rows", len(combined))
    for sys_name in sorted(new_df["SystemName"].unique()):
        n = len(verify[verify["SystemName"] == sys_name])
        log.info("Verified %s: %d rows in file", sys_name, n)
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write to MMasseyOrdinals.csv (default: dry run)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="DEBUG-level logging: show every mapped/unmapped row")
    parser.add_argument("--log-file", metavar="PATH",
                        help="Also write log output to this file")
    args = parser.parse_args()
    main(dry_run=not args.write, verbose=args.verbose, log_file=args.log_file)
