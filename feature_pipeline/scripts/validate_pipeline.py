"""
validate_pipeline.py
--------------------
Post-stage validation checks for the RPI archive pipeline.

Usage (from repo root):
    conda run -n tasty python feature_pipeline/scripts/validate_pipeline.py --stage parse
    conda run -n tasty python feature_pipeline/scripts/validate_pipeline.py --stage resolve
    conda run -n tasty python feature_pipeline/scripts/validate_pipeline.py --stage integrate
    conda run -n tasty python feature_pipeline/scripts/validate_pipeline.py --all
"""

import argparse
import glob
import os
import sys

import pandas as pd

CSV_DIR  = "data/raw/test/rpi_archive"
PDF_DIR  = "data/raw/pdf/men/team_sheets/rpi_archive"
MASSEY   = "data/kaggle/MMasseyOrdinals.csv"
SOS_SYSTEMS = ["SOS_D1", "SOS_NC", "OSOS_D1", "OSOS_NC", "BPI"]


# ---------------------------------------------------------------------------

def validate_parse() -> bool:
    print("=" * 72)
    print("VALIDATION: Parse Stage")
    print("=" * 72)
    issues = []

    for year_dir in sorted(glob.glob(f"{PDF_DIR}/*")):
        year = os.path.basename(year_dir)
        if not year.isdigit():
            continue
        pdfs = glob.glob(f"{year_dir}/*.pdf")
        csvs = glob.glob(f"{CSV_DIR}/{year}/*.csv")
        # Filter out "(2)" duplicates that are intentionally excluded
        csvs = [c for c in csvs if "(2)" not in os.path.basename(c)]
        if len(csvs) != len(pdfs):
            issues.append(f"Year {year}: {len(csvs)} CSVs but {len(pdfs)} PDFs")

    # Spot-check ≤2014 CSVs for malformed Team names.
    # Post-2014 names may be raw PDF headers — that's expected and fixed by resolve_ambiguous_teams.
    all_csvs = [c for c in glob.glob(f"{CSV_DIR}/**/*.csv", recursive=True)
                if "(2)" not in os.path.basename(c)]
    pre2015_csvs = [c for c in all_csvs
                    if any(f"/{y}/" in c.replace("\\", "/") for y in range(2005, 2015))]
    step = max(1, len(pre2015_csvs) // 10)
    sample = pre2015_csvs[::step]

    malformed_examples = []
    for csv_path in sample:
        try:
            df = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue
        if "Team" not in df.columns:
            continue
        for val in df["Team"].dropna():
            s = str(val)
            if s.startswith("Of ") or "NATIONALCOLLEGIATE" in s or s == "nan":
                malformed_examples.append(f"{os.path.relpath(csv_path, CSV_DIR)}: '{s[:60]}'")

    if malformed_examples:
        issues.append(f"Malformed Team names in ≤2014 sample ({len(malformed_examples)}) — "
                      f"these should be clean via rpi_archive_copy reference:")
        for ex in malformed_examples[:10]:
            issues.append(f"  {ex}")

    if not issues:
        print(f"  PASS — {len(all_csvs)} CSVs, counts match PDFs per year, sample clean")
    else:
        print(f"  FAIL — {len(issues)} issues:")
        for iss in issues:
            print(f"    {iss}")
    return len(issues) == 0


# ---------------------------------------------------------------------------

def validate_resolve() -> bool:
    print("=" * 72)
    print("VALIDATION: Resolve Stage")
    print("=" * 72)
    issues = []

    # Known problem file must be clean
    problem = f"{CSV_DIR}/2019/2019-12-15_Team_Sheets_Cleaned.csv"
    if os.path.exists(problem):
        df = pd.read_csv(problem)
        if "Team" in df.columns:
            bad = df["Team"].astype(str).str.startswith("Of ").sum()
            if bad > 0:
                issues.append(f"2019-12-15 still has {bad} rows with 'Of ' prefix")
    else:
        issues.append(f"Expected file missing: {problem}")

    # No unresolved ambiguous names
    AMBIGUOUS = {"Miami", "Loyola", "Saint Francis"}
    all_csvs = [c for c in glob.glob(f"{CSV_DIR}/**/*.csv", recursive=True)
                if "(2)" not in os.path.basename(c)]
    ambig_found = {}
    for csv_path in all_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Team" not in df.columns:
            continue
        for val in df["Team"].dropna():
            if str(val).strip() in AMBIGUOUS:
                ambig_found[str(val).strip()] = os.path.relpath(csv_path, CSV_DIR)
    if ambig_found:
        issues.append(f"Unresolved ambiguous names: {ambig_found}")

    if not issues:
        print(f"  PASS — 2019-12-15 clean, no unresolved ambiguous names")
    else:
        print(f"  FAIL — {len(issues)} issues:")
        for iss in issues:
            print(f"    {iss}")
    return len(issues) == 0


# ---------------------------------------------------------------------------

def validate_integrate() -> bool:
    print("=" * 72)
    print("VALIDATION: Integrate Stage")
    print("=" * 72)
    issues = []

    if not os.path.exists(MASSEY):
        print(f"  FAIL — {MASSEY} not found")
        return False

    df = pd.read_csv(MASSEY)
    sos_df = df[df["SystemName"].isin(SOS_SYSTEMS)]

    # All systems present
    for sys in SOS_SYSTEMS:
        n = (df["SystemName"] == sys).sum()
        if n == 0:
            issues.append(f"SystemName={sys} has 0 rows")

    # No DayNum out of range
    bad_dn = sos_df[sos_df["RankingDayNum"] > 154]
    if len(bad_dn) > 0:
        issues.append(f"{len(bad_dn)} rows have RankingDayNum > 154")

    # No duplicates
    dup = sos_df.groupby(["Season", "RankingDayNum", "SystemName", "TeamID"]).size()
    dupes = dup[dup > 1]
    if len(dupes) > 0:
        issues.append(f"{len(dupes)} duplicate (Season, DayNum, SystemName, TeamID) keys")

    if not issues:
        print(f"  PASS — all {len(SOS_SYSTEMS)} systems present, no dupes, DayNum in range")
        for sys in SOS_SYSTEMS:
            n = (df["SystemName"] == sys).sum()
            print(f"    {sys}: {n:,} rows")
    else:
        print(f"  FAIL — {len(issues)} issues:")
        for iss in issues:
            print(f"    {iss}")
    return len(issues) == 0


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["parse", "resolve", "integrate"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        results = [
            ("parse",     validate_parse()),
            ("resolve",   validate_resolve()),
            ("integrate", validate_integrate()),
        ]
        print()
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        for stage, ok in results:
            print(f"  {stage:12s}: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if all(r[1] for r in results) else 1)
    elif args.stage == "parse":
        sys.exit(0 if validate_parse() else 1)
    elif args.stage == "resolve":
        sys.exit(0 if validate_resolve() else 1)
    elif args.stage == "integrate":
        sys.exit(0 if validate_integrate() else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
