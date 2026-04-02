"""
Integrate BPI ranking data into MMasseyOrdinals.csv.

Steps:
  1. Combine 2008-2012 and 2013-present BPI CSVs
  2. Map full "Team Nickname" names → Kaggle TeamID using a 3-tier strategy:
       Tier 1: Strip last word  ("Kansas Jayhawks"     → "kansas")
       Tier 2: Strip last 2 words ("North Carolina Tar Heels" → "north carolina")
       Tier 3: Hardcoded specials for 9 edge cases
  3. Validate — ABORT if any team is unmapped
  4. Append rows to MMasseyOrdinals.csv with SystemName='BPI', RankingDayNum=152

Run from repo root: python scripts/integrate_bpi_data_v3.py
Add --write flag to commit the change; default is dry-run only.
"""

import argparse
import pandas as pd
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BPI_NEW   = "data/raw/bpi/espn_bpi_team_rankings_2013_to_present.csv"
MTEAMS    = "data/kaggle/MTeams.csv"
SPELLINGS = "data/kaggle/MTeamSpellings.csv"
MASSEY    = "data/kaggle/MMasseyOrdinals.csv"

# ---------------------------------------------------------------------------
# Tier-3 hardcoded overrides  (BPI full name → exact spelling in MTeamSpellings)
# ---------------------------------------------------------------------------
SPECIAL_CASES = {
    "app state mountaineers":                "appalachian st",
    "miami hurricanes":                      "miami fl",
    "miami (fl) hurricanes":                 "miami fl",
    "queens university royals":              "queens (nc)",
    "st. thomas-minnesota tommies":          "st thomas mn",
    "saint francis red flash":               "saint francis (pa)",
    "san josé state spartans":               "san jose st",
    "san jose state spartans":               "san jose st",
    "ualbany great danes":                   "albany",
    "ul monroe warhawks":                    "louisiana-monroe",
    "ut rio grande valley vaqueros":         "utrgv",
    "gardner-webb runnin' bulldogs":         "gardner-webb",
}


def build_spelling_lookup(spellings_path):
    df = pd.read_csv(spellings_path)
    return dict(zip(df["TeamNameSpelling"].str.lower(), df["TeamID"].astype(int)))


def resolve_name(bpi_name: str, lookup: dict, dry_run: bool = True) -> int | None:
    """Return TeamID for a BPI full team name, or None if unresolvable."""
    name = bpi_name.lower().strip()

    if name in SPECIAL_CASES:
        candidate = SPECIAL_CASES[name]
        return lookup.get(candidate), f"special_cases -> {candidate}"

    parts = name.split()

    if len(parts) >= 2:
        short = " ".join(parts[:-1])
        if short in lookup:
            return lookup[short], f"strip_last -> {short}"

    if len(parts) >= 3:
        short2 = " ".join(parts[:-2])
        if short2 in lookup:
            return lookup[short2], f"strip_last_two -> {short2}"

    return None, "unresolved"



def main(dry_run: bool = True):
    print("=" * 72)
    print("STEP 1  Load and prepare ESPN BPI")
    print("=" * 72)
    bpi = pd.read_csv(BPI_NEW)
    bpi = bpi[bpi['season']==2026]
    
    bpi.columns = bpi.columns.str.lower()
    print(f"  2026 BPI: {len(bpi):,} rows, season confirmation {bpi['season'].min()}–{bpi['season'].max()}")

    print()
    print("=" * 72)
    print("STEP 2  Build team-name → TeamID lookup")
    print("=" * 72)
    lookup = build_spelling_lookup(SPELLINGS)
    print(f"  Spelling entries: {len(lookup):,}")

    print()
    print("=" * 72)
    print("STEP 3  Map BPI team names")
    print("=" * 72)
    if dry_run:
        bpi[["TeamID", "match_rule"]] = bpi["team"].apply(
            lambda n: pd.Series(resolve_name(n, lookup), dry_run=True)
        )
        print(bpi[["team", "TeamID", "match_rule"]].to_string(index=False))
    else:
        bpi[["TeamID", "match_rule"]] = bpi["team"].apply(
            lambda n: pd.Series(resolve_name(n, lookup, dry_run=False))
            )

    unmapped = bpi[bpi["TeamID"].isna()]
    if len(unmapped):
        print(f"\n  UNRESOLVED NAMES ({len(unmapped)} rows):")
        for name in sorted(unmapped["team"].unique()):
            print(f"    '{name}'")
        print()

    ok = bpi[bpi["TeamID"].notna()].copy()
    pct = 100 * len(ok) / len(bpi)
    print(f"  Mapped  : {len(ok):,} / {len(bpi):,} rows  ({pct:.2f}%)")

    if pct < 100:
        print("\n  *** MAPPING INCOMPLETE — fix SPECIAL_CASES or lookup before writing ***")
        if not dry_run:
            print("  Aborting write.")
            sys.exit(1)

    print()
    print("=" * 72)
    print("STEP 4  Build MMasseyOrdinals rows")
    print("=" * 72)
    new_rows = pd.DataFrame({
        "Season":        ok["season"].astype(int),
        "RankingDayNum": 149,
        "SystemName":    "BPI",
        "TeamID":        ok["TeamID"].astype(int),
        "OrdinalRank":   ok["bpi_rank"].astype(int),
    })

    # Sanity checks
    assert new_rows["Season"].between(2008, 2030).all(), "Unexpected season values"
    assert new_rows["OrdinalRank"].between(1, 500).all(), "Unexpected rank values"
    assert new_rows["TeamID"].notna().all(), "Null TeamIDs in output"
    assert new_rows.duplicated(subset=["Season", "TeamID"]).sum() == 0, \
        "Duplicate Season+TeamID pairs"

    print(f"  Rows to append: {len(new_rows):,}")
    print(f"  Seasons: {sorted(new_rows['Season'].unique())}")
    print()
    print("  Sample (5 rows):")
    print(new_rows.sample(5, random_state=42).sort_values("Season").to_string(index=False))

    print()
    print("=" * 72)
    print("STEP 5  Load MMasseyOrdinals and check for duplicates")
    print("=" * 72)
    massey = pd.read_csv(MASSEY)
    print(f"  Existing rows: {len(massey):,}")

    existing_bpi = massey[
        (massey["SystemName"] == "BPI") & (massey["RankingDayNum"] == 152)
    ]
    print(f"  Existing BPI DayNum=152 rows: {len(existing_bpi):,}")

    # Check overlap
    overlap = new_rows.merge(
        massey[massey["SystemName"] == "BPI"][["Season", "RankingDayNum", "TeamID"]],
        on=["Season", "RankingDayNum", "TeamID"],
        how="inner",
    )
    if len(overlap):
        print(f"\n  WARNING: {len(overlap)} rows already exist in Massey as BPI DayNum=152")
        print("  These will be SKIPPED (deduplication).")
        new_rows = new_rows[~new_rows.set_index(["Season", "TeamID"]).index.isin(
            overlap.set_index(["Season", "TeamID"]).index
        )]
        print(f"  Rows after dedup: {len(new_rows):,}")

    if dry_run:
        print()
        print("=" * 72)
        print("DRY RUN — no changes written.")
        print("Re-run with --write to update MMasseyOrdinals.csv")
        print("=" * 72)
        return

    print()
    print("=" * 72)
    print("STEP 6  Write to MMasseyOrdinals.csv")
    print("=" * 72)
    combined = pd.concat([massey, new_rows], ignore_index=True)
    combined.to_csv(MASSEY, index=False)
    print(f"  Written: {len(combined):,} rows total")
    verify = pd.read_csv(MASSEY)
    added = verify[(verify["SystemName"] == "BPI") & (verify["RankingDayNum"] == 152)]
    print(f"  Verification — BPI DayNum=152 rows in file: {len(added):,}")
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write to MMasseyOrdinals.csv (default: dry run only)",
    )
    args = parser.parse_args()
    main(dry_run=not args.write)
