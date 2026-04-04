"""
resolve_ambiguous_teams.py

For every CSV in rpi_archive/:
  1. Extract the clean team name from each row's raw Team column value,
     handling all observed date/label prefix/suffix formats.
  2. For the one known-broken file (2019-12-15_Team_Sheets_Cleaned.csv)
     where extraction fails for all rows, recover names from the matching PDF.
  3. For ambiguous team names (Miami, Loyola, Saint Francis), use the
     matching PDF + row-position proximity to pick the correct variant.
  4. Write back just the clean team name (no date prefix) to the Team column.

Observed raw Team formats:
  Suffix:   "TEAM Of Final YYYY", "TEAM Of DOW, Month D, YYYY", ...
  Prefix:   "Of Final YYYY TEAM", "Of YYYY Final TEAM W-L SYS",
            "Of DOW, TEAM W-L SYS", "Of DOW TEAM W-L SYS",
            "Of Month D TEAM W-L SYS", "Of Month D, TEAM W-L SYS",
            "Of Month TEAM W-L SYS", "Of AbbrevMonth D, TEAM W-L SYS",
            "Of DD-MON-YY TEAM W-L SYS", "Of D1 MBB NET TEAM W-L NET",
            "Of Month D TEAM" (bare, no W-L)
  Clean:    "TEAM" or "TEAM W-L SYS"
  Skip:     "Of Selection Sunday", "NATIONAL...", "NITTY-GRITTY..."

Usage (from repo root):
  conda run -n tasty python feature_pipeline/scripts/resolve_ambiguous_teams.py
  conda run -n tasty python feature_pipeline/scripts/resolve_ambiguous_teams.py --write
  conda run -n tasty python feature_pipeline/scripts/resolve_ambiguous_teams.py --year 2020 --write
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import fitz
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from feature_pipeline.pdf_utils import load_pdf_team_names, load_pdf_team_names_for_disambiguation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_DIR  = "data/raw/test/rpi_archive"
PDF_DIR  = "data/raw/pdf/men/team_sheets/rpi_archive"
PROD_DIR = "data/team_sheets"

YEAR_CUTOFF = 2014  # ≤ this: use production CSV for disambiguation; > this: use PDF

# If extraction succeeds for fewer than this fraction of rows, recover names from PDF
PDF_RECOVER_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# Ambiguous names that need PDF lookup to pick the right variant
# ---------------------------------------------------------------------------
AMBIGUOUS = {"Miami", "Loyola"}

# Names with a single known correct resolution — no PDF lookup needed
FORCED_RESOLUTION: dict[str, str] = {
    "Saint Francis":  "Saint Francis (PA)",
    "SaintFrancis":   "Saint Francis (PA)",
    "St. Francis":    "Saint Francis (PA)",
    "St. Mary's":     "St. Mary's (CA)"
}

PAGE_WINDOW = 20

# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------
_DAYS          = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTHS        = (r"(?:January|February|March|April|May|June|July|August|"
                  r"September|October|November|December)")
_MONTHS_ABBREV = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?"

# Each tuple: (pattern, greedy_capture)
# greedy=True → capture is .+ (bare, apply _clean to strip any trailing record)
# greedy=False → capture is .+? with \d+-\d+\s+[A-Z]{2,}$ anchor (W-L required)
_PREFIX_PATTERNS = [
    (rf"^Final\s+\d{{4}}\s+(.+)$",                                          True),
    (rf"^\d{{4}}\s+Final\s+(.+)$",                                          True),
    (rf"^D1\s+MBB\s+\w+\s+(.+)$",                                          True),
    # DOW, Month D, YYYY TEAM
    (rf"^{_DAYS},\s+{_MONTHS}\.?\s+\d+,\s+\d{{4}}\s+(.+)$",                True),
    # Month D, YYYY TEAM
    (rf"^{_MONTHS}\s+\d+,\s+\d{{4}}\s+(.+)$",                              True),
    # DOW, TEAM W-L SYS
    (rf"^{_DAYS},\s+(.+?)\s+\d+-\d+\s+[A-Z]{{2,}}$",                       False),
    # DOW TEAM W-L SYS
    (rf"^{_DAYS}\s+(.+?)\s+\d+-\d+\s+[A-Z]{{2,}}$",                        False),
    # AbbrevMonth D, TEAM W-L SYS   e.g. "Dec. 25, Iowa 9-3 NET"
    (rf"^{_MONTHS_ABBREV}\s+\d+,\s+(.+?)\s+\d+-\d+\s+[A-Z]{{2,}}$",        False),
    # DD-MON-YY TEAM W-L SYS        e.g. "17-DEC-19 Villanova 8-2 NET"
    (rf"^\d{{1,2}}-[A-Z]{{3}}-\d{{2,4}}\s+(.+?)\s+\d+-\d+\s+[A-Z]{{2,}}$", False),
    # Month D, TEAM W-L SYS         e.g. "March 1, Kansas 25-3 NET"
    (rf"^{_MONTHS}\s+\d+,\s+(.+?)\s+\d+-\d+\s+[A-Z]{{2,}}$",               False),
    # Month D TEAM W-L SYS          e.g. "January 5 San Diego St. 14-0 NET"
    (rf"^{_MONTHS}\s+\d+\s+(.+?)\s+\d+-\d+\s+[A-Z]{{2,}}$",                False),
    # Month TEAM W-L SYS (no day)   e.g. "February Colorado 21-8 NET"
    (rf"^{_MONTHS}\s+(?!\d)(.+?)\s+\d+-\d+\s+[A-Z]{{2,}}$",                False),
    # Bare: Month D, TEAM or Month D TEAM (no W-L)
    (rf"^{_MONTHS}\s+\d+,?\s+(.+)$",                                        True),
    # Bare: Month TEAM (no day, no W-L)  e.g. "February Miami"
    (rf"^{_MONTHS}\s+(?!\d)(.+)$",                                          True),
    # Bare: DOW, TEAM (no W-L)
    (rf"^{_DAYS},\s+(.+)$",                                                 True),
    # Bare: DOW TEAM (no W-L)
    (rf"^{_DAYS}\s+(.+)$",                                                  True),
]

_RECORD_SUFFIX = re.compile(r"\s+\d+-\d+\s+[A-Za-z]{2,}$")
_D1_PREFIX     = re.compile(r"^D1\s+MBB\s+\w+\s+", re.IGNORECASE)
_STRAY_LETTER  = re.compile(r"^[A-Z]\s+")  # e.g. "D Minnesota" → "Minnesota"


def _strip_record(s: str) -> str:
    return _RECORD_SUFFIX.sub("", s).strip()


def _clean(s: str) -> str:
    s = _strip_record(s)
    s = _D1_PREFIX.sub("", s).strip()
    return s


def extract_team_name(raw: str) -> str | None:
    s = str(raw).strip()

    if s in ("Of Selection Sunday", "nan", ""):
        return None
    if re.fullmatch(r"Of Selection Sunday.*", s):
        return None
    if "NATIONALCOLLEGIATEATHLETICASSOCIATION" in s:
        return None
    if s.startswith("NITTY-GRITTY"):
        return None

    # Suffix: "TEAM Of ..."
    if " Of " in s and not s.startswith("Of "):
        return s.split(" Of ")[0].strip()

    # Prefix: "Of ..."
    if s.startswith("Of "):
        rest = s[3:]
        for pat, greedy in _PREFIX_PATTERNS:
            m = re.match(pat, rest, re.IGNORECASE)
            if m:
                result = m.group(1).strip()
                result = _clean(result) if greedy else _strip_record(result)
                result = _STRAY_LETTER.sub("", result).strip()
                return result if result else None
        return None

    # Embedded record suffix: "TEAM W-L SYS"
    stripped = _strip_record(s)
    if stripped != s:
        return stripped.strip()

    return s


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def pdf_for_csv(csv_path: str) -> str:
    rel = os.path.relpath(csv_path, CSV_DIR)
    return os.path.join(PDF_DIR, rel.replace("_Cleaned.csv", ".pdf"))


# team_name_from_page and load_pdf_pages replaced by load_pdf_team_names from pdf_utils


# ---------------------------------------------------------------------------
# Disambiguation helpers
# ---------------------------------------------------------------------------

def build_prod_map() -> dict[int, str]:
    result: dict[int, str] = {}
    for path in glob.glob(os.path.join(PROD_DIR, "*_Team_Sheets*.csv")):
        m = re.match(r"(\d{4})-", os.path.basename(path))
        if m:
            result[int(m.group(1))] = path
    return result


def resolve_via_prod(row_idx: int, clean: str, prod_df: pd.DataFrame) -> str | None:
    if row_idx >= len(prod_df):
        return None
    prod_name = str(prod_df.iloc[row_idx][prod_df.columns[0]]).strip()
    return None if prod_name.lower() == clean.lower() else prod_name


def resolve_via_pdf(row_idx: int, clean: str, page_names: dict[int, str]) -> str | None:
    expected = row_idx + 1
    lo = max(1, expected - PAGE_WINDOW)
    hi = min(max(page_names, default=expected + PAGE_WINDOW), expected + PAGE_WINDOW)
    norm = re.sub(r"\.", "", clean.lower()).strip()
    candidates = sorted(
        [(abs(pg - expected), pg, name)
         for pg, name in page_names.items()
         if lo <= pg <= hi
         and re.sub(r"\.", "", name.lower()).startswith(norm)]
    )
    if not candidates:
        return None
    resolved = candidates[0][2]
    return None if resolved.lower() == clean.lower() else resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LOG_FILE = "resolve_rows.log"

# Row statuses written to log
_OK              = "OK"
_MISSING         = "MISSING"          # PDF gave no extractable name
_SEL_SUNDAY      = "SELECTION_SUNDAY" # expected skip — not a team row
_AMBIG_RESOLVED  = "AMBIG_RESOLVED"
_AMBIG_UNRESOLVED = "AMBIG_UNRESOLVED"


def main(dry_run: bool = True, year_filter: int | None = None) -> None:
    yf = f"  year={year_filter}" if year_filter else ""
    print("=" * 72)
    print(f"{'DRY RUN' if dry_run else 'WRITE MODE'}  —  resolve_ambiguous_teams{yf}")
    print(f"  Row-level log → {LOG_FILE}")
    print("=" * 72)

    prod_map = build_prod_map()

    all_csvs = sorted(glob.glob(f"{CSV_DIR}/**/*.csv", recursive=True))
    all_csvs = [f for f in all_csvs if "(2)" not in os.path.basename(f)]
    if year_filter:
        all_csvs = [f for f in all_csvs
                    if int(os.path.basename(os.path.dirname(f))) == year_filter]

    stats = Counter()
    skipped_files: list[tuple[str, str]] = []
    pdf_recovered: list[tuple[str, float]] = []
    ambig_counter = Counter()
    changes_by_file: dict[str, pd.DataFrame] = {}
    # Accumulate rows that need attention across all files
    missing_rows:   list[tuple[str, int, str]] = []  # (rel, row_idx, raw_val)
    unresolved_ambig: list[tuple[str, int, str]] = []  # (rel, row_idx, name)

    log_fh = open(LOG_FILE, "w")
    log_fh.write("file,row_idx,pdf_page,raw_original,raw_pdf,cleaned,status\n")

    for csv_path in all_csvs:
        folder_year = int(os.path.basename(os.path.dirname(csv_path)))
        rel = os.path.relpath(csv_path, CSV_DIR)

        df = pd.read_csv(csv_path)
        if "Team" not in df.columns:
            skipped_files.append((rel, "no Team column"))
            continue

        data_cols = [c for c in df.columns if c != "Team"]
        df = df[df[data_cols].notna().any(axis=1)].reset_index(drop=True)
        if df.empty:
            skipped_files.append((rel, "all rows empty"))
            continue

        # ── Step 1: get team names ──────────────────────────────────────────
        pdf_path  = pdf_for_csv(csv_path)
        use_pdf   = True  # always use PDF — gives full unambiguous names for all years
        raw_pages: dict[int, str] = {}   # {1-based page: raw spatial name}

        if use_pdf:
            raw_pages = load_pdf_team_names(pdf_path)
            if not raw_pages:
                skipped_files.append((rel, f"year={folder_year}: PDF not found or empty"))
                continue
            # PDF header may include date cruft — strip with extract_team_name()
            extracted = [
                extract_team_name(raw_pages[i + 1]) if (i + 1) in raw_pages else None
                for i in range(len(df))
            ]
            pdf_recovered.append((rel, 0.0))
            stats["pdf_recovered_rows"] += len(extracted)
        else:
            # ≤2014: rpi_archive_copy already gave clean names
            success_rate = sum(
                1 for v in df["Team"].astype(str)
                if extract_team_name(v) is not None
            ) / len(df)
            if success_rate < PDF_RECOVER_THRESHOLD:
                raw_pages = load_pdf_team_names(pdf_path)
                if not raw_pages:
                    skipped_files.append((rel, f"PDF fallback needed ({success_rate:.0%}) but not found"))
                    continue
                extracted = [
                    extract_team_name(raw_pages[i + 1]) if (i + 1) in raw_pages else None
                    for i in range(len(df))
                ]
                pdf_recovered.append((rel, success_rate))
                stats["pdf_recovered_rows"] += len(extracted)
            else:
                extracted = [extract_team_name(str(v)) for v in df["Team"].astype(str)]

        # ── Step 2: apply forced resolutions, then disambiguate ────────────
        for i, name in enumerate(extracted):
            if name and name in FORCED_RESOLUTION:
                extracted[i] = FORCED_RESOLUTION[name]
                ambig_counter[FORCED_RESOLUTION[name]] += 1

        page_names: dict[int, str] = {}  # text-based, preserves (FL)/(OH)/(PA)
        ambig_indices = [i for i, n in enumerate(extracted) if n and n in AMBIGUOUS]
        if ambig_indices:
            page_names = load_pdf_team_names_for_disambiguation(pdf_path)

        first_valid = next((e for e in extracted if e), None)
        is_alpha = (
            folder_year <= YEAR_CUTOFF
            and first_valid is not None
            and first_valid[0].lower() <= "c"
        )
        prod_df = (
            pd.read_csv(prod_map[folder_year])
            if is_alpha and folder_year in prod_map
            else None
        )

        for i in ambig_indices:
            clean    = extracted[i]
            resolved = None
            if is_alpha and prod_df is not None:
                resolved = resolve_via_prod(i, clean, prod_df)
            elif page_names:
                resolved = resolve_via_pdf(i, clean, page_names)
            if resolved:
                resolved = re.sub(r'\s+\d+-\d+.*$', '', resolved).strip()
            if resolved:
                extracted[i] = resolved
                ambig_counter[resolved] += 1

        # ── Step 3: log every row ───────────────────────────────────────────
        original_vals = list(df["Team"].astype(str))
        new_values: list[str] = []

        for i in range(len(df)):
            orig      = original_vals[i]
            raw_pdf   = raw_pages.get(i + 1, "") if raw_pages else ""
            cleaned   = extracted[i]
            pdf_page  = i + 1  # 1-based assumption: row i ↔ PDF page i+1

            # Determine status
            if cleaned is None:
                # Check if original was a Selection Sunday row (expected skip)
                if re.fullmatch(r"Of Selection Sunday.*", orig, re.IGNORECASE):
                    status = _SEL_SUNDAY
                else:
                    status = _MISSING
                    if folder_year > YEAR_CUTOFF:
                        missing_rows.append((rel, i, orig))
                        stats["missing_post2014"] += 1
                    else:
                        stats["missing_pre2015"] += 1
            elif i in ambig_indices:
                if cleaned not in AMBIGUOUS:
                    status = f"{_AMBIG_RESOLVED}:{cleaned}"
                else:
                    status = _AMBIG_UNRESOLVED
                    unresolved_ambig.append((rel, i, cleaned))
                    stats["ambig_unresolved"] += 1
            else:
                status = _OK
                stats["rows_ok"] += 1

            # Write log line
            log_fh.write(
                f"{rel},{i},{pdf_page},"
                f"{orig!r},{raw_pdf!r},{cleaned!r},{status}\n"
            )

            # Determine final value to write to CSV
            if cleaned is not None:
                final_val = cleaned
            elif folder_year > YEAR_CUTOFF:
                # Post-2014: no filling — leave empty so integrate script skips it
                final_val = ""
            else:
                # ≤2014: fall back to original (should already be clean)
                final_val = orig

            new_values.append(final_val)

        stats["partial_none_rows"] += sum(1 for e in extracted if e is None)

        # ── Step 4: check if file needs updating ───────────────────────────
        if list(df["Team"].astype(str)) == new_values:
            continue

        df_out = df.copy()
        df_out["Team"] = new_values
        changes_by_file[csv_path] = df_out
        stats["files_to_patch"] += 1
        stats["rows_cleaned"] += sum(1 for e in extracted if e is not None)

    log_fh.close()

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print(f"  Files to patch            : {stats['files_to_patch']:,}")
    print(f"  Rows OK                   : {stats['rows_ok']:,}")
    print(f"  Rows from PDF recovery    : {stats['pdf_recovered_rows']:,}")
    print(f"  Rows MISSING (post-2014)  : {stats['missing_post2014']:,}")
    print(f"  Rows MISSING (≤2014)      : {stats['missing_pre2015']:,}")
    print(f"  Ambiguous unresolved      : {stats['ambig_unresolved']:,}")
    print()

    if ambig_counter:
        print("  Disambiguation breakdown:")
        for name, cnt in sorted(ambig_counter.items()):
            print(f"    {name!r:40s}: {cnt:,}")
        print()

    if missing_rows:
        print(f"  *** MISSING post-2014 rows (name not extractable from PDF) ***")
        for rel, row_idx, orig in missing_rows[:30]:
            print(f"    {rel}  row {row_idx}  orig={orig!r}")
        if len(missing_rows) > 30:
            print(f"    ... ({len(missing_rows) - 30} more — see {LOG_FILE})")
        print()

    if unresolved_ambig:
        print(f"  *** AMBIGUOUS unresolved ***")
        for rel, row_idx, name in unresolved_ambig[:20]:
            print(f"    {rel}  row {row_idx}  name={name!r}")
        print()

    if pdf_recovered:
        print(f"  Files using PDF extraction ({len(pdf_recovered)}):")
        for f, rate in pdf_recovered[:5]:
            print(f"    {f}")
        if len(pdf_recovered) > 5:
            print(f"    ... ({len(pdf_recovered) - 5} more)")
        print()

    print(f"  Files SKIPPED ({len(skipped_files)}):")
    for f, reason in skipped_files:
        print(f"    {f}  —  {reason}")

    print(f"\n  Full row log written to: {LOG_FILE}")

    if dry_run:
        print()
        print("DRY RUN — no files written. Re-run with --write.")
        return

    print()
    print("Writing...")
    for csv_path, df_out in changes_by_file.items():
        df_out.to_csv(csv_path, index=False)
    print(f"  Written {len(changes_by_file):,} files.")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    main(dry_run=not args.write, year_filter=args.year)
