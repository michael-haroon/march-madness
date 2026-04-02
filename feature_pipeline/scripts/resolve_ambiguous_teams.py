"""
resolve_ambiguous_teams.py

For each ambiguous team name in the rpi_archive CSVs (Miami, Loyola,
Saint Francis / St. Francis, St. Mary's), open the matching PDF, locate
the correct team page using row-position ≈ page-number heuristic, extract
the full disambiguated name from the PDF header, and patch the CSV.

PDF structure (confirmed from samples):
  Header line: "TEAM_NAME     (OFFICIAL) Through Games Of DATE"
  Each page = one team, pages in alphabetical order matching CSV row order.

Usage:
  conda run -n tasty python scripts/resolve_ambiguous_teams.py          # dry run
  conda run -n tasty python scripts/resolve_ambiguous_teams.py --write  # patch CSVs
"""

import argparse
import glob
import os
import re
import sys

import fitz          # PyMuPDF
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_DIR = "data/raw/test/rpi_archive"
PDF_DIR = "data/raw/pdf/men/team_sheets/rpi_archive"

# ---------------------------------------------------------------------------
# Ambiguous team names to resolve (as returned by extract_team_name)
# ---------------------------------------------------------------------------
AMBIGUOUS = {
    "Miami",
    "Loyola",
    "Saint Francis",
    "SaintFrancis",
    "St. Francis",
    "St. Mary's",
}

# Page search window around expected row position
PAGE_WINDOW = 8


# ---------------------------------------------------------------------------
# PDF path from CSV path
# ---------------------------------------------------------------------------

def pdf_for_csv(csv_path: str) -> str | None:
    """
    data/raw/test/rpi_archive/2013/2013-01-06_Team_Sheets_Cleaned.csv
    →  data/raw/pdf/men/team_sheets/rpi_archive/2013/2013-01-06_Team_Sheets.pdf
    """
    rel = os.path.relpath(csv_path, CSV_DIR)          # 2013/2013-01-06_Team_Sheets_Cleaned.csv
    pdf_rel = rel.replace("_Cleaned.csv", ".pdf")     # 2013/2013-01-06_Team_Sheets.pdf
    return os.path.join(PDF_DIR, pdf_rel)


# ---------------------------------------------------------------------------
# Team-name extraction from one PDF page using fitz
# ---------------------------------------------------------------------------

def team_name_from_page(page: fitz.Page) -> str | None:
    """
    Extract team name from a PDF team-sheet page.

    fitz extracts text starting with 'Page N\\nLoss,\\n...\\nOT\\nGmDte\\n'
    followed by the team name on its own line.  The team name is always
    the line immediately after 'GmDte\\n'.
    """
    text = page.get_text()
    if not text:
        return None

    # Primary: team name is the line right after "GmDte"
    m = re.search(r"GmDte\n(.+?)\n", text)
    if m:
        return m.group(1).strip()

    # Fallback: first non-empty, non-numeric, non-"Page N" line
    for line in text.splitlines():
        line = line.strip()
        if (line
                and not re.match(r"^(Page\s+\d+|Loss,|Non-Conf|\d)", line)
                and not line.startswith("(")):
            return line

    return None


# ---------------------------------------------------------------------------
# Load all page team-names from a PDF (cached dict: page_num → name)
# ---------------------------------------------------------------------------

def load_pdf_pages(pdf_path: str) -> dict[int, str]:
    """Return {1-based page number: team name} for every page in the PDF."""
    result: dict[int, str] = {}
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [WARN] cannot open PDF {pdf_path}: {e}")
        return result

    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]
        name = team_name_from_page(page)
        if name:
            result[page_num] = name
    doc.close()
    return result


# ---------------------------------------------------------------------------
# Extract clean team name from raw CSV Team cell (mirrors integrate script)
# ---------------------------------------------------------------------------

_RECORD_SUFFIX = re.compile(r"\s+\d+-\d+\s+[A-Z]+$")
_D1_PREFIX     = re.compile(r"^D1\s+MBB\s+\w+\s+", re.IGNORECASE)
_DAYS   = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"


def _clean_team(s: str) -> str:
    s = _RECORD_SUFFIX.sub("", s).strip()
    s = _D1_PREFIX.sub("", s).strip()
    return s


def extract_team_name(raw: str) -> str | None:
    s = str(raw).strip()
    if s in ("Of Selection Sunday", "nan"):
        return None
    if "NATIONALCOLLEGIATEATHLETICASSOCIATION" in s or s.startswith("NITTY-GRITTY"):
        return None
    if re.fullmatch(r"Of Selection Sunday.*", s):
        return None

    if " Of " in s and not s.startswith("Of "):
        return s.split(" Of ")[0].strip()

    if s.startswith("Of "):
        rest = s[3:]
        for pat in [
            r"^Final\s+\d{4}\s+(.+)$",
            r"^\d{4}\s+Final\s+(.+)$",
            r"^[A-Za-z]+,\s+[A-Za-z]+\.?\s+\d+,\s+\d{4}\s+(.+)$",
            r"^[A-Za-z]+\s+\d+,\s+\d{4}\s+(.+)$",
        ]:
            m = re.match(pat, rest)
            if m:
                return _clean_team(m.group(1))

        m = re.match(rf"^{_MONTHS}\s+\d+\s+(.+?)\s+\d+-\d+\s+[A-Z]+$", rest)
        if m:
            return m.group(1).strip()
        m = re.match(rf"^{_DAYS}\s+(.+?)\s+\d+-\d+\s+[A-Z]+$", rest)
        if m:
            return m.group(1).strip()
        return None

    stripped = _RECORD_SUFFIX.sub("", s)
    if stripped != s:
        return stripped.strip()
    return s


# ---------------------------------------------------------------------------
# Resolve a single ambiguous occurrence
# ---------------------------------------------------------------------------

def resolve_occurrence(
    ambiguous_name: str,
    row_idx: int,       # 0-based data row index (after dropping empty rows)
    page_names: dict[int, str],
) -> str | None:
    """
    Search PDF pages near row_idx for a page whose name starts with
    ambiguous_name.  Return the full page name (e.g. 'Miami (Fl.)').
    """
    expected_page = row_idx + 1   # 1-indexed
    lo = max(1, expected_page - PAGE_WINDOW)
    hi = min(max(page_names.keys()) if page_names else expected_page + PAGE_WINDOW,
             expected_page + PAGE_WINDOW)

    # Collect pages in window that match the ambiguous prefix
    ambig_lower = ambiguous_name.lower().rstrip(".")
    candidates: list[tuple[int, str]] = []
    for pg in range(lo, hi + 1):
        name = page_names.get(pg, "")
        if name.lower().startswith(ambig_lower):
            candidates.append((pg, name))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][1]

    # Multiple matches — pick the one closest to expected_page
    candidates.sort(key=lambda x: abs(x[0] - expected_page))
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = True) -> None:
    print("=" * 72)
    print(f"{'DRY RUN' if dry_run else 'WRITE MODE'}  —  resolve_ambiguous_teams")
    print("=" * 72)

    all_csvs = sorted(glob.glob(f"{CSV_DIR}/**/*.csv", recursive=True))
    all_csvs = [f for f in all_csvs if "(2)" not in os.path.basename(f)]

    total_resolved = total_no_pdf = total_no_match = 0
    all_changes: list[dict] = []

    for csv_path in all_csvs:
        pdf_path = pdf_for_csv(csv_path)
        df = pd.read_csv(csv_path)
        if "Team" not in df.columns:
            continue

        # Drop rows where all non-Team columns are empty
        data_cols = [c for c in df.columns if c != "Team"]
        df = df[df[data_cols].notna().any(axis=1)].reset_index(drop=True)

        # Find rows with ambiguous team names
        ambig_rows: list[tuple[int, str, str]] = []   # (row_idx, raw_value, clean_name)
        for idx, raw in enumerate(df["Team"].astype(str)):
            clean = extract_team_name(raw)
            if clean and clean in AMBIGUOUS:
                ambig_rows.append((idx, raw, clean))

        if not ambig_rows:
            continue

        # Load PDF pages (once per file)
        if not os.path.exists(pdf_path):
            total_no_pdf += len(ambig_rows)
            for _, raw, clean in ambig_rows:
                print(f"  [NO PDF] {os.path.relpath(csv_path, CSV_DIR)}: '{clean}' (PDF not found)")
            continue

        page_names = load_pdf_pages(pdf_path)
        if not page_names:
            total_no_pdf += len(ambig_rows)
            continue

        for row_idx, raw_value, clean_name in ambig_rows:
            resolved = resolve_occurrence(clean_name, row_idx, page_names)
            if resolved is None or resolved.lower() == clean_name.lower():
                total_no_match += 1
                print(f"  [UNRESOLVED] {os.path.relpath(csv_path, CSV_DIR)} "
                      f"row {row_idx}: '{clean_name}' → no PDF match nearby")
                continue

            # Build the patched raw Team value by replacing the ambiguous token
            new_raw = raw_value.replace(clean_name, resolved, 1)
            all_changes.append({
                "csv_path":   csv_path,
                "row_idx":    row_idx,
                "old_raw":    raw_value,
                "new_raw":    new_raw,
                "clean_name": clean_name,
                "resolved":   resolved,
            })
            total_resolved += 1

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  Resolved   : {total_resolved:,}")
    print(f"  No PDF     : {total_no_pdf:,}")
    print(f"  Unresolved : {total_no_match:,}")
    print()

    if all_changes:
        # Group by resolved name for a summary
        from collections import Counter
        summary = Counter(c["resolved"] for c in all_changes)
        print("  Resolution breakdown:")
        for name, count in sorted(summary.items()):
            print(f"    {name!r:40s}: {count:,} rows")

        print()
        print("  Sample changes (first 10):")
        for c in all_changes[:10]:
            rel = os.path.relpath(c["csv_path"], CSV_DIR)
            print(f"    {rel} row {c['row_idx']:3d}: "
                  f"{c['clean_name']!r} → {c['resolved']!r}")

    if dry_run:
        print()
        print("=" * 72)
        print("DRY RUN — no files written.")
        print("Re-run with --write to patch CSVs.")
        print("=" * 72)
        return

    # ── Apply changes ─────────────────────────────────────────────────────
    from collections import defaultdict
    changes_by_file: dict[str, list[dict]] = defaultdict(list)
    for c in all_changes:
        changes_by_file[c["csv_path"]].append(c)

    print("=" * 72)
    print("Patching CSVs...")
    patched_files = 0
    for csv_path, changes in changes_by_file.items():
        df = pd.read_csv(csv_path)
        data_cols = [c for c in df.columns if c != "Team"]
        df_work = df[df[data_cols].notna().any(axis=1)].reset_index(drop=True)

        changed = False
        for c in changes:
            idx = c["row_idx"]
            if idx < len(df_work) and df_work.at[idx, "Team"] == c["old_raw"]:
                df_work.at[idx, "Team"] = c["new_raw"]
                changed = True
            else:
                print(f"  [WARN] row mismatch in {os.path.relpath(csv_path, CSV_DIR)} "
                      f"row {idx}")

        if changed:
            df_work.to_csv(csv_path, index=False)
            patched_files += 1

    print(f"  Patched {patched_files:,} CSV files.")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Patch CSVs (default: dry run)")
    args = parser.parse_args()
    main(dry_run=not args.write)
