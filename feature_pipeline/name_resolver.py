"""
name_resolver.py
----------------
Bidirectional mapping between the three team naming systems in this project:

  1. Team sheet names  (e.g. "Connecticut", "Loyola (Ill.)")
  2. Kaggle TeamIDs    (e.g. 1163 for Connecticut)
  3. Team-stats names  (e.g. "Connecticut (Big East)")

The master source of truth is:
  - data/kaggle/MTeamSpellings.csv  →  1,178 name variants → TeamID
  - data/kaggle/MTeams.csv          →  381 canonical TeamID → TeamName
  - pipeline/config.TEAM_NAME_MAP   →  manual overrides for edge cases

Usage:
    from feature_pipeline.name_resolver import build_id_lookup, resolve_team_id, strip_conference

    lookup = build_id_lookup("data/kaggle")
    team_id = resolve_team_id("UConn", lookup)          # → 1163
    team_id = resolve_team_id("Connecticut (Big East)", lookup)  # → 1163
"""

import re
import os
import unicodedata
import pandas as pd
from difflib import SequenceMatcher


# ─────────────────────────────────────────────────────────────────────────────
#  String normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation noise."""
    if not isinstance(s, str):
        return ""
    # Unicode normalise → decompose accents
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().strip()
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def strip_conference(name: str) -> str:
    """
    Remove conference parenthetical from team-stats names.
    'Alabama (SEC)'       → 'Alabama'
    'Miami (OH) (MAC)'    → 'Miami (OH)'   ← strip LAST paren only
    'Connecticut (Big East)' → 'Connecticut'
    """
    if not isinstance(name, str):
        return name
    # Remove the last parenthetical group (always the conference)
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return stripped if stripped else name


# ─────────────────────────────────────────────────────────────────────────────
#  Build the lookup dictionary
# ─────────────────────────────────────────────────────────────────────────────

def build_id_lookup(kaggle_dir: str) -> dict:
    """
    Build a normalised-name → TeamID lookup from Kaggle reference files.

    Sources (in priority order):
      1. MTeamSpellings.csv  – 1,178 official alternative spellings
      2. MTeams.csv          – 381 canonical short names
      3. Extended variants   – computed from canonical (e.g. "st " → "saint ")

    Returns:
        dict mapping lowercase-normalised team name → int TeamID
    """
    lookup: dict[str, int] = {}

    # ── 1. MTeamSpellings ────────────────────────────────────────────────────
    spellings_path = os.path.join(kaggle_dir, "MTeamSpellings.csv")
    if os.path.exists(spellings_path):
        spellings = pd.read_csv(spellings_path, dtype={"TeamID": int})
        for _, row in spellings.iterrows():
            key = _normalise(str(row["TeamNameSpelling"]))
            if key:
                lookup[key] = int(row["TeamID"])

    # ── 2. MTeams canonical names ────────────────────────────────────────────
    teams_path = os.path.join(kaggle_dir, "MTeams.csv")
    teams_df = pd.DataFrame()
    if os.path.exists(teams_path):
        teams_df = pd.read_csv(teams_path, dtype={"TeamID": int})
        for _, row in teams_df.iterrows():
            key = _normalise(str(row["TeamName"]))
            if key:
                lookup.setdefault(key, int(row["TeamID"]))

    # ── 3. Computed variants ─────────────────────────────────────────────────
    # Add "saint X" → same as "st. X" / "st X" and vice versa
    extra: dict[str, int] = {}
    for name, tid in lookup.items():
        if name.startswith("st ") or name.startswith("st. "):
            saint_ver = re.sub(r"^st\.?\s+", "saint ", name)
            extra.setdefault(saint_ver, tid)
        if name.startswith("saint "):
            st_ver = re.sub(r"^saint\s+", "st ", name)
            extra.setdefault(st_ver, tid)
    lookup.update(extra)

    return lookup


def build_teams_df(kaggle_dir: str) -> pd.DataFrame:
    """Load MTeams.csv for reverse lookup (TeamID → canonical name)."""
    path = os.path.join(kaggle_dir, "MTeams.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"TeamID": int})
    return pd.DataFrame(columns=["TeamID", "TeamName"])


# ─────────────────────────────────────────────────────────────────────────────
#  Resolve a name → TeamID
# ─────────────────────────────────────────────────────────────────────────────

def resolve_team_id(name: str,
                    lookup: dict,
                    team_name_map: dict = None,
                    fuzzy_threshold: float = 0.82) -> int | None:
    """
    Resolve a team name string to its Kaggle TeamID.

    Resolution order:
      1. Apply manual TEAM_NAME_MAP overrides (e.g. "UConn" → "Connecticut")
      2. Exact normalised match in lookup
      3. Strip conference parenthetical, retry exact match
      4. Fuzzy match (SequenceMatcher ratio ≥ fuzzy_threshold)

    Returns TeamID (int) or None if unresolvable.
    """
    if not isinstance(name, str) or not name.strip():
        return None

    if team_name_map is None:
        from feature_pipeline.config import TEAM_NAME_MAP
        team_name_map = TEAM_NAME_MAP

    # Step 1: try original name as-is (before any remapping)
    key_orig = _normalise(name)
    if key_orig in lookup:
        return lookup[key_orig]

    # Step 1b: strip conference from original, try lookup
    key_stripped_orig = _normalise(strip_conference(name))
    if key_stripped_orig in lookup:
        return lookup[key_stripped_orig]

    # Step 2: apply manual override then retry
    canonical = team_name_map.get(name, name)
    canonical = team_name_map.get(canonical, canonical)  # two-pass for chains

    key = _normalise(canonical)
    if key in lookup:
        return lookup[key]

    # Step 3: strip conference paren from canonical and retry
    stripped = strip_conference(canonical)
    key2 = _normalise(stripped)
    if key2 in lookup:
        return lookup[key2]

    # Step 4: fuzzy match against all known spellings
    best_ratio = 0.0
    best_tid = None
    for known_name, tid in lookup.items():
        ratio = SequenceMatcher(None, key2 or key, known_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_tid = tid

    if best_ratio >= fuzzy_threshold:
        return best_tid

    return None


def resolve_team_name(team_id: int, teams_df: pd.DataFrame) -> str | None:
    """Reverse lookup: TeamID → canonical TeamName from MTeams.csv."""
    row = teams_df[teams_df["TeamID"] == team_id]
    if len(row) == 0:
        return None
    return row.iloc[0]["TeamName"]


# ─────────────────────────────────────────────────────────────────────────────
#  Batch resolution with miss logging
# ─────────────────────────────────────────────────────────────────────────────

def resolve_names_series(names: pd.Series,
                         lookup: dict,
                         team_name_map: dict = None,
                         fuzzy_threshold: float = 0.82) -> pd.Series:
    """
    Vectorised version: resolve a Series of team name strings → Series of TeamIDs.
    Returns int64 Series (NaN for unresolved).
    """
    return names.apply(
        lambda n: resolve_team_id(n, lookup, team_name_map, fuzzy_threshold)
    ).astype("Int64")  # nullable integer


# ─────────────────────────────────────────────────────────────────────────────
#  Verification / diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def verify_coverage(kaggle_dir: str,
                    min_season: int = 2003,
                    verbose: bool = True) -> dict:
    """
    Test resolve_team_id() against all teams that have ever appeared in
    MNCAATourneySeeds.csv for seasons >= min_season.

    Prints miss rate and returns a dict with:
      - miss_rate (float)
      - misses (list of team name strings that failed)
      - suggested_additions (dict of {name: best_fuzzy_match} for manual TEAM_NAME_MAP)
    """
    from feature_pipeline.config import TEAM_NAME_MAP

    lookup   = build_id_lookup(kaggle_dir)
    teams_df = build_teams_df(kaggle_dir)

    seeds_path = os.path.join(kaggle_dir, "MNCAATourneySeeds.csv")
    seeds = pd.read_csv(seeds_path)
    seeds = seeds[seeds["Season"] >= min_season]

    # Merge to get the canonical TeamName for each TeamID
    seeds = seeds.merge(teams_df[["TeamID", "TeamName"]], on="TeamID", how="left")
    unique_names = seeds["TeamName"].dropna().unique()

    misses = []
    for name in unique_names:
        tid = resolve_team_id(name, lookup, TEAM_NAME_MAP)
        if tid is None:
            misses.append(name)

    miss_rate = len(misses) / len(unique_names) if len(unique_names) > 0 else 0.0

    if verbose:
        print(f"\n=== Name Resolver Coverage ===")
        print(f"  Unique tournament team names (≥{min_season}): {len(unique_names)}")
        print(f"  Resolved: {len(unique_names) - len(misses)}")
        print(f"  Misses:   {len(misses)}  ({miss_rate:.1%})")
        if misses:
            print(f"  Unresolved names: {misses}")

    return {
        "miss_rate": miss_rate,
        "misses": misses,
        "total": len(unique_names),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point for quick verification
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    kaggle_dir = sys.argv[1] if len(sys.argv) > 1 else "data/kaggle"
    result = verify_coverage(kaggle_dir, verbose=True)
    if result["miss_rate"] > 0.02:
        print(f"\nWARNING: miss rate {result['miss_rate']:.1%} exceeds 2% threshold.")
        print("Add the following to TEAM_NAME_MAP in pipeline/config.py:")
        for m in result["misses"]:
            print(f'    "{m}": "<canonical name>",')
        sys.exit(1)
    else:
        print(f"\nPass: miss rate {result['miss_rate']:.1%} ≤ 2%")
