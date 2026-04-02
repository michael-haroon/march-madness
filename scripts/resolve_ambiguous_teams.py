"""
resolve_ambiguous_teams.py

Resolves ambiguous team names (Miami, Loyola, Saint Francis, St. Francis,
St. Mary's) in the rpi_archive CSVs by two strategies:

  ≤ 2014  (alphabetically ordered CSVs):
    1-to-1 row mapping against the clean production team sheets in
    data/team_sheets/.  Row i in archive == row i in production because
    both are alphabetically sorted and contain the same team set.

  > 2014  (rank-ordered CSVs):
    PDF parsing via PyMuPDF.  Row i in CSV ≈ page i+1 in the matching PDF
    (one page per team, alphabetical order within PDF).  The team name is
    extracted from the line immediately after "GmDte" in the page text.

Usage:
  conda run -n tasty python scripts/resolve_ambiguous_teams.py          # dry run
  conda run -n tasty python scripts/resolve_ambiguous_teams.py --write  # patch CSVs
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import fitz          # PyMuPDF
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_DIR  = "data/raw/test/rpi_archive"
PDF_DIR  = "data/raw/pdf/men/team_sheets/rpi_archive"
PROD_DIR = "data/team_sheets"

YEAR_CUTOFF = 2014   # ≤ this year: use production CSV; > this year: use PDF

# ---------------------------------------------------------------------------
# Ambiguous team names (as returned by extract_team_name)
# ---------------------------------------------------------------------------
# Note: "St. Mary's" removed — only Saint Mary's (CA) exists in D1 basketball;
# Mt. St. Mary's is a different name and won't collide.
AMBIGUOUS = {"Miami", "Loyola", "Saint Francis", "SaintFrancis", "St. Francis"}

PAGE_WINDOW = 20   # 2015+ PDFs are rank-ordered like CSVs; allow ±20 pages of drift

# ---------------------------------------------------------------------------
# Build year → production CSV path mapping
# ---------------------------------------------------------------------------

def build_prod_map(prod_dir: str) -> dict[int, str]:
    """Map season year → production team-sheet CSV path."""
    result: dict[int, str] = {}
    for path in glob.glob(os.path.join(prod_dir, "*_Team_Sheets*.csv")):
        fname = os.path.basename(path)
        m = re.match(r"(\d{4})-", fname)
        if m:
            result[int(m.group(1))] = path
    return result


def build_pdf_map(pdf_dir: str) -> dict[str, str]:
    """Map base filename (without _Cleaned.csv) → PDF path."""
    result: dict[str, str] = {}
    for path in glob.glob(os.path.join(pdf_dir, "**/*.pdf"), recursive=True):
        key = os.path.basename(path).replace(".pdf", "")
        result[key] = path
    return result


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def team_name_from_page(page: fitz.Page) -> str | None:
    """
    Extract team name from a PDF team-sheet page.

    Two observed formats:
      Old (≤ 2014): text begins 'Page N\\nLoss,\\n...\\nOT\\nGmDte\\nTEAM_NAME\\n'
      New (> 2014): stats summary line 'TEAM_NAME (NET: X KPI: X, ...)'
    """
    text = page.get_text()
    if not text:
        return None

    # Old format (≤ 2014): team name immediately follows "GmDte\n"
    # Validate it's not a record like "16- 2" (starts with a digit)
    m = re.search(r"GmDte\n(.+?)\n", text)
    if m:
        candidate = m.group(1).strip()
        if not re.match(r"^\d", candidate):
            return candidate

    # New format (2015+), top-ranked teams:
    # "TEAM_NAME (RPI: rank ...)" or "TEAM_NAME (NET: rank ...)"
    # Strict: parenthesis immediately before the system abbreviation + digit.
    # Correctly handles "Miami (FL) (RPI: 5 ...)" → "Miami (FL)"
    m = re.search(r"\n(.+?) \((?:NET|RPI): \d+", text)
    if m:
        return m.group(1).strip()

    # New format (2015+), lower-ranked teams:
    # "TEAM_NAME W-L  NET: rank ( KPI: rank)" — record on same line, no "(NET:" paren
    # Extract the team name by stopping before the W-L record.
    m = re.search(r"\n(.+?) \d+[-\u2013]\s*\d+\s+", text)
    if m:
        candidate = m.group(1).strip()
        # Sanity: must look like a team name (starts with a letter)
        if re.match(r"^[A-Za-z]", candidate):
            return candidate

    return None


def load_pdf_pages(pdf_path: str) -> dict[int, str]:
    result: dict[int, str] = {}
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    [WARN] cannot open PDF {pdf_path}: {e}")
        return result
    for pg in range(1, len(doc) + 1):
        name = team_name_from_page(doc[pg - 1])
        if name:
            result[pg] = name
    doc.close()
    return result


# ---------------------------------------------------------------------------
# Team name extraction from raw CSV cell (mirrors integrate script)
# ---------------------------------------------------------------------------

_RECORD_SUFFIX = re.compile(r"\s+\d+-\d+\s+[A-Z]+$")
_D1_PREFIX     = re.compile(r"^D1\s+MBB\s+\w+\s+", re.IGNORECASE)
_DAYS   = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTHS = (r"(?:January|February|March|April|May|June|July|August|"
           r"September|October|November|December)")


def _clean_team(s: str) -> str:
    s = _RECORD_SUFFIX.sub("", s).strip()
    return _D1_PREFIX.sub("", s).strip()


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
    return stripped.strip() if stripped != s else s


# ---------------------------------------------------------------------------
# Strategy A: production CSV row mapping (≤ 2014)
# ---------------------------------------------------------------------------

def resolve_via_prod_csv(
    ambig_rows: list[tuple[int, str, str]],   # (row_idx, raw, clean_name)
    prod_df: pd.DataFrame,
) -> dict[int, str]:
    """Return {row_idx: resolved_name} using production CSV 1-to-1 mapping."""
    team_col = prod_df.columns[0]   # always 'Team'
    resolved: dict[int, str] = {}
    for row_idx, raw, clean in ambig_rows:
        if row_idx < len(prod_df):
            prod_name = str(prod_df.iloc[row_idx][team_col]).strip()
            if prod_name and prod_name.lower() != clean.lower():
                resolved[row_idx] = prod_name
            else:
                # prod CSV has the same name → confirmed single occurrence this year,
                # not truly ambiguous; no patch needed (integration script handles it)
                resolved[row_idx] = "__SINGLE__"
        else:
            print(f"    [WARN] row_idx {row_idx} out of range for prod CSV "
                  f"({len(prod_df)} rows)")
    return resolved


# ---------------------------------------------------------------------------
# Strategy B: PDF page proximity (> 2014)
# ---------------------------------------------------------------------------

def resolve_via_pdf(
    ambig_rows: list[tuple[int, str, str]],
    page_names: dict[int, str],
) -> dict[int, str]:
    """Return {row_idx: resolved_name} using PDF page proximity."""
    resolved: dict[int, str] = {}
    used_pages: set[int] = set()

    for row_idx, raw, clean in ambig_rows:
        expected = row_idx + 1
        lo = max(1, expected - PAGE_WINDOW)
        hi = min(max(page_names) if page_names else expected + PAGE_WINDOW,
                 expected + PAGE_WINDOW)
        # Normalize ambiguous name for prefix matching:
        # 'SaintFrancis' → 'saint francis', 'St. Francis' → 'st francis'
        ambig_norm = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", clean).lower()
        ambig_norm = re.sub(r"\.", "", ambig_norm).strip()

        candidates = [
            (abs(pg - expected), pg, name)
            for pg, name in page_names.items()
            if lo <= pg <= hi
            and re.sub(r"\.", "", name.lower()).startswith(ambig_norm)
            and pg not in used_pages
        ]
        if not candidates:
            continue
        candidates.sort()
        _, best_pg, best_name = candidates[0]
        used_pages.add(best_pg)
        if best_name.lower() != clean.lower():
            resolved[row_idx] = best_name

    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = True) -> None:
    print("=" * 72)
    print(f"{'DRY RUN' if dry_run else 'WRITE MODE'}  —  resolve_ambiguous_teams")
    print("=" * 72)

    prod_map = build_prod_map(PROD_DIR)
    pdf_map  = build_pdf_map(PDF_DIR)
    print(f"  Production CSVs: {sorted(prod_map)}")
    print(f"  PDFs indexed   : {len(pdf_map):,}")
    print()

    all_csvs = sorted(glob.glob(f"{CSV_DIR}/**/*.csv", recursive=True))
    all_csvs = [f for f in all_csvs if "(2)" not in os.path.basename(f)]

    stats = {"resolved": 0, "no_source": 0, "unresolved": 0, "row_mismatch": 0}
    all_changes: list[dict] = []

    for csv_path in all_csvs:
        folder_year = int(os.path.basename(os.path.dirname(csv_path)))
        fname       = os.path.basename(csv_path)

        df = pd.read_csv(csv_path)
        if "Team" not in df.columns:
            continue
        data_cols = [c for c in df.columns if c != "Team"]
        df = df[df[data_cols].notna().any(axis=1)].reset_index(drop=True)

        # Find ambiguous rows
        ambig_rows: list[tuple[int, str, str]] = []
        for idx, raw in enumerate(df["Team"].astype(str)):
            clean = extract_team_name(raw)
            if clean and clean in AMBIGUOUS:
                ambig_rows.append((idx, raw, clean))
        if not ambig_rows:
            continue

        # ── Choose resolution strategy ────────────────────────────────────
        if folder_year <= YEAR_CUTOFF:
            # Strategy A: production CSV
            prod_path = prod_map.get(folder_year)
            if not prod_path:
                stats["no_source"] += len(ambig_rows)
                print(f"  [NO PROD CSV] {folder_year}: {fname} "
                      f"({len(ambig_rows)} ambiguous rows)")
                continue
            prod_df = pd.read_csv(prod_path)
            resolved = resolve_via_prod_csv(ambig_rows, prod_df)
            source = f"prod:{os.path.basename(prod_path)}"
        else:
            # Strategy B: PDF
            pdf_key = fname.replace("_Cleaned.csv", "")
            pdf_path = pdf_map.get(pdf_key)
            if not pdf_path:
                stats["no_source"] += len(ambig_rows)
                print(f"  [NO PDF] {folder_year}/{fname} "
                      f"({len(ambig_rows)} ambiguous rows, no matching PDF)")
                continue
            page_names = load_pdf_pages(pdf_path)
            resolved = resolve_via_pdf(ambig_rows, page_names)
            source = f"pdf:{os.path.basename(pdf_path)}"

        # ── Build changes ─────────────────────────────────────────────────
        for row_idx, raw_value, clean_name in ambig_rows:
            if row_idx not in resolved:
                stats["unresolved"] += 1
                print(f"  [UNRESOLVED] {folder_year}/{fname} "
                      f"row {row_idx}: {clean_name!r}")
                continue
            resolved_name = resolved[row_idx]
            if resolved_name == "__SINGLE__":
                # Confirmed only one school with this name this year — no patch needed
                continue
            new_raw = raw_value.replace(clean_name, resolved_name, 1)
            all_changes.append({
                "csv_path":    csv_path,
                "row_idx":     row_idx,
                "old_raw":     raw_value,
                "new_raw":     new_raw,
                "clean_name":  clean_name,
                "resolved":    resolved_name,
                "source":      source,
            })
            stats["resolved"] += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  Resolved       : {stats['resolved']:,}")
    print(f"  No source      : {stats['no_source']:,}  (no prod CSV or PDF)")
    print(f"  Unresolved     : {stats['unresolved']:,}")
    print()

    from collections import Counter
    if all_changes:
        summary = Counter(c["resolved"] for c in all_changes)
        print("  Resolution breakdown:")
        for name, count in sorted(summary.items()):
            print(f"    {name!r:40s}: {count:,} rows")
        print()
        print("  Sample changes (first 10):")
        for c in all_changes[:10]:
            rel = os.path.relpath(c["csv_path"], CSV_DIR)
            print(f"    {rel} row {c['row_idx']:3d}: "
                  f"{c['clean_name']!r} → {c['resolved']!r}  [{c['source']}]")

    if dry_run:
        print()
        print("DRY RUN — no files written. Re-run with --write to patch CSVs.")
        return

    # ── Apply changes ─────────────────────────────────────────────────────
    changes_by_file: dict[str, list[dict]] = defaultdict(list)
    for c in all_changes:
        changes_by_file[c["csv_path"]].append(c)

    print("=" * 72)
    print("Patching CSVs...")
    patched = 0
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
                stats["row_mismatch"] += 1
                print(f"  [WARN] mismatch "
                      f"{os.path.relpath(csv_path, CSV_DIR)} row {idx}")

        if changed:
            df_work.to_csv(csv_path, index=False)
            patched += 1

    print(f"  Patched {patched:,} CSV files.")
    if stats["row_mismatch"]:
        print(f"  Row mismatches: {stats['row_mismatch']:,}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    main(dry_run=not args.write)
