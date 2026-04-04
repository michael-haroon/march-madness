"""
Integrate RPI archive SOS data into MMasseyOrdinals.csv.

For each CSV in data/raw/test/rpi_archive/{year}/:
  1. Extract team name from the malformed Team column (13 different format patterns)
  2. Map team name → Kaggle TeamID via MTeamSpellings
  3. Compute Season + DayNum from file date and MSeasons DayZero
  4. Emit rows for SOS_D1, SOS_NonConf, Opp_SOS_D1, Opp_SOS_NonConf into MMasseyOrdinals

Usage:
  python scripts/integrate_rpi_archive_sos.py          # dry run (no writes)
  python scripts/integrate_rpi_archive_sos.py --write  # commit to MMasseyOrdinals.csv
"""

import argparse
import glob
import os
import re
import sys
import warnings
from datetime import datetime, date

import pandas as pd

from feature_pipeline.season_utils import build_season_table, get_season_and_daynum

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARCHIVE_DIR  = "data/raw/test/rpi_archive"
MSEASONS     = "data/kaggle/MSeasons.csv"
SPELLINGS    = "data/kaggle/MTeamSpellings.csv"
MASSEY       = "data/kaggle/MMasseyOrdinals.csv"

# Systems to extract from each file → MMasseyOrdinals SystemName
SOS_SYSTEMS = {
    "SOS_D1":            "SOS_D1",
    "SOS_NonConf":       "SOS_NC",
    "Opp_SOS_D1":        "OSOS_D1",
    "Opp_SOS_NonConf":   "OSOS_NC",
    "BPI":               "BPI",
    "POM":               "POM",
    "NET":               "NET",
    "KPI":               "KPI"
}

# ---------------------------------------------------------------------------
# Team name extraction
# ---------------------------------------------------------------------------

_RECORD_SUFFIX = re.compile(r"\s+\d+-\d+\s+[A-Z]+$")
_DAYS   = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"


_D1_PREFIX = re.compile(r"^D1\s+MBB\s+\w+\s+", re.IGNORECASE)


def _strip_record(s: str) -> str:
    """Remove embedded 'W-L SYSTEM' suffix (e.g. '16-0 NET') if present."""
    return _RECORD_SUFFIX.sub("", s).strip()


def _clean_team(s: str) -> str:
    """Strip record suffix then 'D1 MBB NET' sub-prefix if present."""
    s = _strip_record(s)
    s = _D1_PREFIX.sub("", s).strip()
    return s


def extract_team_name(raw: str) -> str | None:
    """
    Strip the date/label cruft from a malformed Team cell and return the
    clean team name, or None if the cell contains no usable team name.

    Handles 14 observed formats:
      Prefix:  'Of Final YYYY TEAM', 'Of YYYY Final TEAM',
               'Of DayOfWeek, Month D, YYYY TEAM [W-L SYS]',
               'Of Month D, YYYY TEAM [W-L SYS]',
               'Of DayOfWeek TEAM W-L SYS'
      Suffix:  'TEAM Of Final YYYY [Final|RPI]',
               'TEAM Of DayOfWeek, Month D, YYYY',
               'TEAM Of Month D, YYYY'
      Embedded record:  'TEAM W-L SYS'  (2019 clean format)
      Clean:   'TEAM'
      Skip:    'Of Selection Sunday', 'NATIONALCOLLEGIATEATHLETICASSOCIATION'
    """
    s = str(raw).strip()

    # ── Hard skip ─────────────────────────────────────────────────────────
    if s in ("Of Selection Sunday", "nan"):
        return None
    if "NATIONALCOLLEGIATEATHLETICASSOCIATION" in s:
        return None
    if re.fullmatch(r"Of Selection Sunday.*", s):
        return None
    if s.startswith("NITTY-GRITTY"):
        return None

    # ── Suffix pattern: "TEAM Of ..." ─────────────────────────────────────
    if " Of " in s and not s.startswith("Of "):
        return s.split(" Of ")[0].strip()

    # ── Prefix pattern: starts with "Of " ─────────────────────────────────
    if s.startswith("Of "):
        rest = s[3:]

        # "Final YYYY TEAM"
        m = re.match(r"^Final\s+\d{4}\s+(.+)$", rest)
        if m:
            return _clean_team(m.group(1))

        # "YYYY Final TEAM"
        m = re.match(r"^\d{4}\s+Final\s+(.+)$", rest)
        if m:
            return _clean_team(m.group(1))

        # "DayOfWeek, Month D, YYYY TEAM [W-L SYS]"
        m = re.match(
            r"^[A-Za-z]+,\s+[A-Za-z]+\.?\s+\d+,\s+\d{4}\s+(.+)$", rest
        )
        if m:
            return _clean_team(m.group(1))

        # "Month D, YYYY TEAM [W-L SYS]"
        m = re.match(r"^[A-Za-z]+\s+\d+,\s+\d{4}\s+(.+)$", rest)
        if m:
            return _clean_team(m.group(1))

        # 2020 "Month D TEAM W-L SYSTEM" (no comma, no year)
        # Must come BEFORE DayOfWeek pattern — both start with [A-Za-z]+
        # e.g. "January 5 San Diego St. 14-0 NET"
        m = re.match(rf"^{_MONTHS}\s+\d+\s+(.+?)\s+\d+-\d+\s+[A-Z]+$", rest)
        if m:
            return m.group(1).strip()

        # 2020 "DayOfWeek TEAM W-L SYSTEM"  e.g. "Wednesday San Diego St. 13-0 NET"
        m = re.match(rf"^{_DAYS}\s+(.+?)\s+\d+-\d+\s+[A-Z]+$", rest)
        if m:
            return m.group(1).strip()

        return None   # unrecognised prefix

    # ── Embedded record suffix without any "Of" cruft: "TEAM W-L SYS" ────
    # e.g. "Virginia 16-0 NET" (2019 Jan format)
    stripped = _strip_record(s)
    if stripped != s:
        return stripped

    # ── Already clean ─────────────────────────────────────────────────────
    return s


# build_season_table and get_season_and_daynum are imported from feature_pipeline.season_utils


def parse_file_date(filename: str, folder_year: int) -> date | None:
    """
    Extract the date from a filename.
    - YYYY-MM-DD_* → parse directly
    - Final_YYYY_* → return None (caller uses DayNum=154)
    """
    # Date-stamped file
    m = re.match(r"(\d{4}-\d{2}-\d{2})_", filename)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    # Final file — no date embedded
    return None


# ---------------------------------------------------------------------------
# Team name → TeamID
# ---------------------------------------------------------------------------

def build_spelling_lookup(spellings_path: str) -> dict[str, int]:
    df = pd.read_csv(spellings_path)
    return dict(zip(df["TeamNameSpelling"].str.lower(), df["TeamID"].astype(int)))


def _normalize_variants(raw: str) -> tuple[list[str], bool]:
    """
    Return (candidates, is_duplicate) where:
      - candidates: name strings to try against the lookup, in priority order
      - is_duplicate: True if raw ends with a digit, marking it as a second
        occurrence of a same-named team (e.g. 'Miami2' = Miami OH).
        These rows are skipped and deferred to manual PDF review.

    Pipeline applied to the raw name:
      1. Detect trailing digit → is_duplicate flag, strip it
      2. Compact spaced-out chars: 'W i c h i t a S t .' → 'WichitaSt.'
      3. CamelCase split (lower→UPPER): 'WichitaSt.' → 'Wichita St.'
      4. ALL-CAPS prefix split (UPPER→UPPER+lower): 'UCDavis' → 'UC Davis'
      5. Period stripping: 'Wichita St.' → 'Wichita St'
      All steps are applied in combination.
    """
    s = raw.strip()

    # ── Detect and strip trailing duplicate indicator ──────────────────────
    is_duplicate = bool(re.search(r"\d+$", s))
    base = re.sub(r"\d+$", "", s).strip()

    # ── Compact spaced-out characters ─────────────────────────────────────
    # Include apostrophe so "S t . J o h n ' s" → "St.John's"
    if re.match(r"^[A-Za-z&.'] [A-Za-z&.']", base):
        base = re.sub(r"(?<=[A-Za-z&.'\-]) (?=[A-Za-z&.'\-])", "", base).strip()

    # ── Build candidates with CamelCase splits and period-stripping ────────
    # Abbreviation overrides: maps cleaned lowercase name → canonical lookup key
    # Used for names that are too abbreviated to resolve automatically.
    ABBREV_MAP: dict[str, str] = {
        "ark-little rock":        "ark little rock",
        "ark little rock":        "ark little rock",
        "birmingham-so":          "birmingham so",
        "birmingham so":          "birmingham so",
        "cal st fullerton":       "cs fullerton",
        "cal st. fullerton":      "cs fullerton",
        "calst.fullerton":        "cs fullerton",
        "cal state fullerton":    "cs fullerton",
        "central conn st":        "central conn",
        "central conn. st":       "central conn",
        "centralconnst":          "central conn",
        "col of charleston":      "col charleston",
        "col. of charleston":     "col charleston",
        "colofcharleston":        "col charleston",
        "east caro":              "east carolina",
        "east caro.":             "east carolina",
        "east tenn. st.":         "etsu",
        "east tenn st":           "etsu",
        "easttennst":             "etsu",
        "fla. atlantic":          "florida atlantic",
        "fla atlantic":           "florida atlantic",
        "flaatlantic":            "florida atlantic",
        "fla. gulf coast":        "florida gulf coast",
        "fla gulf coast":         "florida gulf coast",
        "florida int'l":          "florida intl",
        "ga. southern":           "georgia southern",
        "gasouthern":             "georgia southern",
        "lmu":                    "loy marymount",
        "loyola":                 None,   # ambiguous — skip
        "md.-east. shore":        "md e shore",
        "md-east. shore":         "md e shore",
        "md e shore":             "md e shore",
        "miami":                  None,   # ambiguous FL/OH — skip
        "mcneese st":             "mcneese st",
        "mt. st. mary's":         "mt st mary's",
        "mt. st. marys":          "mt st mary's",
        "neb. omaha":             "omaha",
        "neb omaha":              "omaha",
        "north fla.":             "north florida",
        "north fla":              "north florida",
        "s.c. upstate":           "sc upstate",
        "s c upstate":            "sc upstate",
        "saint francis":          None,   # ambiguous NY/PA — skip
        "southwest mo. st.":      "missouri st",
        "southwest mo st":        "missouri st",
        "tex. a&m-corp. chris":   "a&m-corpus christi",
        "ul lafayette":           "louisiana",
        "william&mary":           "william & mary",
        # Disambiguated Miami names (from resolve_ambiguous_teams.py patches)
        "miami florida":          "miami fl",
        "miami ohio":             "miami oh",
        # Disambiguated St. Francis names
        "st. francis ny":         "st. francis (bkn)",
        "st. francis penn":       "saint francis (pa)",
        # saint francis no longer ambiguous after CSV patches
        "saint francis":          "saint francis (pa)",
        "st. francis":            "saint francis (pa)",
        # UC Santa Barbara
        "ucsantabarbara":         "uc santa barbara",
        "uc santabarbara":        "uc santa barbara",
        # Saint Mary's — only one D1 school (CA); Mt. St. Mary's is a different name
        "saint mary's":           "saint mary's (ca)",
        "st. mary's":             "saint mary's (ca)",
        "saintmarys":             "saint mary's (ca)",
        # Cal State schools
        "cal st. bakersfield":    "cs bakersfield",
        "cal st bakersfield":     "cs bakersfield",
        # Mc-prefix teams (CamelCase split inserts space, breaking the name)
        "mc neese st":            "mcneese st",
        "mc neese":               "mcneese",
        # N.C. A&T
        "n c a&t":                "nc a&t",
        "n. c. a&t":              "nc a&t",
        "nca&t":                  "nc a&t",
        # Apostrophe names
        "saint joseph's":         "saint joseph's",
        "st mary's":              "saint mary's (ca)",
        "st. john's":             "st. john's",
        "st john's":              "st. john's",
        "st. francis brooklyn":   "st. francis (bkn)",
        "stfrancisbrooklyn":      "st. francis (bkn)",
        # Southeast Missouri State
        "southeast mo. st.":      "southeast missouri state",
        "southeast mo st":        "southeast missouri state",
        "southeastmost":          "southeast missouri state",
        "southeast mo. st":       "southeast missouri state",
        # Stephen F. Austin
        "stephen f austin":       "stephen f. austin",
        "stephenf.austin":        "stephen f. austin",
        "stephen f. austin":      "stephen f. austin",
        # ── 8-char truncations from rpi_archive copy (≤2014 CSVs) ────────────
        # These are safety-net entries; resolve_ambiguous_teams now uses PDF
        # for all years, so most will already be full names after resolve --write.
        "abilene c":              "abilene christian",
        "alabama a":              "alabama a&m",
        "alabama s":              "alabama st",
        "appalach":               "appalachian st",
        "arizona s":              "arizona st",
        "ark.-pin":               "ark pine bluff",
        "ark-pin":                "ark pine bluff",
        "army west":              "army",
        "austin pe":              "austin peay",
        "bakersfi":               "cs bakersfield",
        "csu baker":              "cs bakersfield",
        "cal st fu":              "cs fullerton",
        "bethune-":               "bethune cookman",
        "binghamt":               "binghamton",
        "boston co":              "boston college",
        "bowling g":              "bowling green",
        "californ":               "california",
        "central a":              "central arkansas",
        "central c":              "central conn",
        "central m":              "central michigan",
        "charlest":               "charleston so",
        "charlott":               "charlotte",
        "chattano":               "chattanooga",
        "chicago s":              "chicago st",
        "cincinna":               "cincinnati",
        "clevelan":               "cleveland st",
        "coastal c":              "coastal carolina",
        "col.of ch":              "col charleston",
        "colof ch":               "col charleston",
        "creighto":               "creighton",
        "dartmout":               "dartmouth",
        "de paul":                "depaul",
        "detroit m":              "detroit mercy",
        "east tenn":              "etsu",
        "eastern i":              "eastern illinois",
        "eastern k":              "eastern kentucky",
        "eastern m":              "eastern michigan",
        "eastern w":              "eastern washington",
        "evansvil":               "evansville",
        "fairfiel":               "fairfield",
        "fairleig":               "fairleigh dickinson",
        "fla atla":               "florida atlantic",
        "flaatla":                "florida atlantic",
        "florida a":              "florida a&m",
        "florida s":              "florida st",
        "ga south":               "georgia southern",
        "gasouth":                "georgia southern",
        "gardner-":               "gardner webb",
        "george ma":              "george mason",
        "george wa":              "george washington",
        "georgeto":               "georgetown",
        "georgia s":              "georgia st",
        "georgia t":              "georgia tech",
        "gramblin":               "grambling",
        "grand can":              "grand canyon",
        "high poin":              "high point",
        "holy cros":              "holy cross",
        "houston b":              "houston baptist",
        "ill.-chi":               "illinois chicago",
        "ill-chi":                "illinois chicago",
        "incarnat":               "incarnate word",
        "indiana s":              "indiana st",
        "jackson s":              "jackson st",
        "james mad":              "james madison",
        "liu brook":              "liu brooklyn",
        "la.-lafa":               "louisiana",
        "la-lafa":                "louisiana",
        "la.-monr":               "louisiana monroe",
        "la-monr":                "louisiana monroe",
        "lafayett":               "lafayette",
        "lamar uni":              "lamar",
        "lamaruni":               "lamar",
        "little ro":              "ark little rock",
        "long beac":              "long beach st",
        "louisian":               "louisiana tech",
        "louisvil":               "louisville",
        "loyola ch":              "loyola chicago",
        "loyola ma":              "loyola maryland",
        "manhatta":               "manhattan",
        "marquett":               "marquette",
        "massachu":               "massachusetts",
        "mc neese s":             "mcneese st",
        "middle te":              "middle tennessee",
        "milwauke":               "milwaukee",
        "minnesot":               "minnesota",
        "montana s":              "montana st",
        "morehead":               "morehead st",
        "mt st ma":               "mt st mary's",
        "mtstma":                 "mt st mary's",
        "n. c. central":          "nc central",
        "n.c.cent":               "nc central",
        "nccent":                 "nc central",
        "nc cent":                "nc central",
        "new hamps":              "new hampshire",
        "new orlea":              "new orleans",
        "norfolk s":              "norfolk st",
        "north ala":              "north alabama",
        "north car":              "north carolina",
        "northcar":               "north carolina",
        "north flo":              "north florida",
        "north tex":              "north texas",
        "northeas":               "northeastern",
        "northwes":               "northwestern",
        "notre dam":              "notre dame",
        "old domin":              "old dominion",
        "oral robe":              "oral roberts",
        "pepperdi":               "pepperdine",
        "pittsbur":               "pittsburgh",
        "prairie v":              "prairie view",
        "presbyte":               "presbyterian",
        "princeto":               "princeton",
        "providen":               "providence",
        "purdue fo":              "purdue fort wayne",
        "quinnipi":               "quinnipiac",
        "rhode isl":              "rhode island",
        "robert mo":              "robert morris",
        "sacramen":               "sacramento st",
        "sacred he":              "sacred heart",
        "saint fra":              "saint francis (pa)",
        "saint jos":              "saint joseph's",
        "saint lou":              "saint louis",
        "saint mar":              "saint mary's (ca)",
        "saint pet":              "saint peter's",
        "sam houst":              "sam houston st",
        "san franc":              "san francisco",
        "san jose s":             "san jose st",
        "santa cla":              "santa clara",
        "savannah":               "savannah st",
        "seton hal":              "seton hall",
        "south ala":              "south alabama",
        "south car":              "south carolina",
        "south fla":              "south florida",
        "southeas":               "se louisiana",
        "st bonav":               "st bonaventure",
        "st franc":               "st. francis (bkn)",
        "st john'":               "st. john's",
        "st mary'":               "saint mary's (ca)",
        "stephen f":              "stephen f. austin",
        "stony bro":              "stony brook",
        "tennesse":               "tennessee",
        "tex.-pan":               "texas pan american",
        "tex-pan":                "texas pan american",
        "texas-ar":               "ut arlington",
        "texas sou":              "texas southern",
        "texas tec":              "texas tech",
        "the citad":              "the citadel",
        "uc rivers":              "uc riverside",
        "ucrivers":               "uc riverside",
        "uc santa b":             "uc santa barbara",
        "u mass low":             "umass lowell",
        "unc ashev":              "unc asheville",
        "uncashev":               "unc asheville",
        "unc green":              "unc greensboro",
        "uncgreen":               "unc greensboro",
        "usc upsta":              "usc upstate",
        "uscupsta":               "usc upstate",
        "ut arling":              "ut arlington",
        "utarling":               "ut arlington",
        "utah vall":              "utah valley",
        "valparai":               "valparaiso",
        "vanderbi":               "vanderbilt",
        "villanov":               "villanova",
        "wake fore":              "wake forest",
        "west virg":              "west virginia",
        "western c":              "western carolina",
        "western i":              "western illinois",
        "western k":              "western kentucky",
        "western m":              "western michigan",
        "wichita s":              "wichita st",
        "william&":               "william & mary",
        "wisconsi":               "wisconsin",
        "youngsto":               "youngstown st",
    }

    seen: set[str] = set()
    result: list[str] = []

    def add(v: str) -> None:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)

    def _strip_p(v: str) -> str:
        """Strip periods — including before a letter (e.g. 'St.F' → 'StF')."""
        return re.sub(r"\.", "", v).strip()

    def _camel(v: str) -> str:
        """Insert space at lower→UPPER boundaries."""
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", v)

    def _allcaps(v: str) -> str:
        """Insert space at UPPER→UPPER+lower boundaries (e.g. 'UCDavis' → 'UC Davis')."""
        return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", v)

    def _try_abbrev(v: str) -> bool:
        """Check abbreviation map; add mapped value or __SKIP__ sentinel. Return True if found."""
        mapped = ABBREV_MAP.get(v.lower())
        if v.lower() not in ABBREV_MAP:
            return False
        if mapped is None:
            add("__SKIP__")
        else:
            add(mapped)
        return True

    def expand(v: str) -> None:
        add(v)
        _try_abbrev(v)

        # Build all variants: CamelCase splits in combinations
        variants: list[str] = [v]

        c1 = _camel(v)              # lowercase→UPPER
        if c1 != v:
            add(c1)
            _try_abbrev(c1)
            variants.append(c1)

        c2 = _allcaps(v)            # UPPER→UPPER+lower
        if c2 != v:
            add(c2)
            variants.append(c2)
            # Chain: also apply lowercase→UPPER to the ALLCAPS-split result
            # so 'UCSantaBarbara' → 'UC SantaBarbara' → 'UC Santa Barbara'
            c2c = _camel(c2)
            if c2c != c2:
                add(c2c)
                _try_abbrev(c2c)
                variants.append(c2c)

        # Period-strip every variant, then re-apply both splits
        for vv in variants:
            sp = _strip_p(vv)
            if sp == vv:
                continue
            add(sp)
            _try_abbrev(sp)
            sp_c = _camel(sp)
            if sp_c != sp:
                add(sp_c)
                _try_abbrev(sp_c)
            sp_u = _allcaps(sp)
            if sp_u != sp:
                add(sp_u)
                sp_uc = _camel(sp_u)
                if sp_uc != sp_u:
                    add(sp_uc)
                    _try_abbrev(sp_uc)

    expand(s)      # original (pre-strip) — highest priority
    if base != s:
        expand(base)   # digit-stripped + compacted

    return result, is_duplicate


def resolve_id(team_name: str, lookup: dict[str, int]) -> tuple[int | None, bool]:
    """
    Returns (TeamID, is_duplicate).
    is_duplicate=True means the row is a second occurrence of a same-named team
    and should be deferred to manual PDF review.
    Returns (None, False) for genuinely ambiguous names (Miami, Loyola, etc.)
    that are explicitly mapped to None in ABBREV_MAP.
    """
    candidates, is_duplicate = _normalize_variants(team_name)
    for candidate in candidates:
        # Explicit None in ABBREV_MAP means "known ambiguous, skip"
        if candidate == "__SKIP__":
            return None, is_duplicate
        tid = lookup.get(candidate.lower())
        if tid is not None:
            return tid, is_duplicate
    return None, is_duplicate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = True) -> None:
    print("=" * 72)
    print("STEP 1  Load reference tables")
    print("=" * 72)
    season_table = build_season_table(MSEASONS)
    spelling_lookup = build_spelling_lookup(SPELLINGS)
    print(f"  Seasons available: {len(season_table)}")
    print(f"  Spelling entries : {len(spelling_lookup):,}")

    print()
    print("=" * 72)
    print("STEP 2  Discover and parse archive files")
    print("=" * 72)

    all_files = sorted(glob.glob(f"{ARCHIVE_DIR}/**/*.csv", recursive=True))
    # Skip duplicate "(2)" files
    all_files = [f for f in all_files if "(2)" not in os.path.basename(f)]
    print(f"  Files to process (excl. duplicates): {len(all_files)}")

    rows_out     = []
    skipped_files = []
    unmapped_teams = set()
    file_stats = []

    for fpath in all_files:
        folder_year = int(os.path.basename(os.path.dirname(fpath)))
        fname       = os.path.basename(fpath)
        is_final    = fname.startswith("Final_")

        # ── Parse date / DayNum ───────────────────────────────────────────
        if is_final:
            # Final = championship day
            season  = folder_year
            daynum  = 154
        else:
            file_date = parse_file_date(fname, folder_year)
            if file_date is None:
                skipped_files.append((fname, "cannot parse date"))
                continue
            season, daynum = get_season_and_daynum(file_date, season_table)
            if season is None:
                skipped_files.append((fname, "date out of range for any season"))
                continue

        # ── Load CSV ──────────────────────────────────────────────────────
        df = pd.read_csv(fpath)
        if "Team" not in df.columns:
            skipped_files.append((fname, "no Team column"))
            continue

        # Quick skip: all rows have no usable team name
        sample_team = extract_team_name(str(df["Team"].iloc[0]))
        if sample_team is None:
            skipped_files.append((fname, "no usable team names (e.g. Selection Sunday)"))
            continue

        # ── Drop rows where all columns except Team are empty ─────────────
        data_cols = [c for c in df.columns if c != "Team"]
        row_has_data = df[data_cols].notna().any(axis=1)
        n_empty = (~row_has_data).sum()
        if n_empty:
            df = df[row_has_data].reset_index(drop=True)

        # ── Process rows ──────────────────────────────────────────────────
        file_ok = file_skip = file_unmapped = file_dupes = 0
        for _, row in df.iterrows():
            team_name = extract_team_name(str(row["Team"]))
            if team_name is None:
                file_skip += 1
                continue

            team_id, is_duplicate = resolve_id(team_name, spelling_lookup)

            if is_duplicate:
                # Second occurrence of a same-named team — defer to PDF review
                file_dupes += 1
                continue

            if team_id is None:
                unmapped_teams.add(team_name)
                file_unmapped += 1
                continue

            for col, sys_name in SOS_SYSTEMS.items():
                if col not in df.columns:
                    continue
                val = row[col]
                # Skip blanks, dashes, non-numeric
                try:
                    rank = int(float(val))
                except (ValueError, TypeError):
                    continue
                if rank <= 0:
                    continue
                rows_out.append({
                    "Season":        season,
                    "RankingDayNum": daynum,
                    "SystemName":    sys_name,
                    "TeamID":        team_id,
                    "OrdinalRank":   rank,
                })
            file_ok += 1

        file_stats.append((fname, season, daynum, file_ok, file_skip, file_unmapped, file_dupes))

    print(f"  Files skipped entirely: {len(skipped_files)}")
    for fname, reason in skipped_files[:10]:
        print(f"    {fname}: {reason}")
    if len(skipped_files) > 10:
        print(f"    ... ({len(skipped_files) - 10} more)")

    problem_files = [(f, s, d, ok, sk, un, du) for f, s, d, ok, sk, un, du in file_stats
                     if sk > 0 or un > 0 or du > 0]
    if problem_files:
        print(f"\n  Files with row-level issues ({len(problem_files)}):")
        for fname, season, daynum, ok, sk, un, du in problem_files:
            print(f"    {fname}  [Season={season}, Day={daynum}]: "
                  f"OK={ok}, Skipped={sk}, Unmapped={un}, Dupes={du}")

    print()
    print("=" * 72)
    print("STEP 3  Mapping summary")
    print("=" * 72)
    total_rows  = sum(s[3] for s in file_stats)
    total_skip  = sum(s[4] for s in file_stats)
    total_unmap = sum(s[5] for s in file_stats)
    total_dupes = sum(s[6] for s in file_stats)
    print(f"  Team rows mapped OK       : {total_rows:,}")
    print(f"  Team rows skipped (no name): {total_skip:,}")
    print(f"  Team rows deferred (dupes) : {total_dupes:,}  ← trailing-digit rows, needs PDF review")
    print(f"  Team rows unmapped        : {total_unmap:,}")

    if unmapped_teams:
        print(f"\n  Unmapped team names ({len(unmapped_teams)}):")
        for t in sorted(unmapped_teams):
            print(f"    '{t}'")

    print()
    print("=" * 72)
    print("STEP 4  Build output DataFrame")
    print("=" * 72)
    new_df = pd.DataFrame(rows_out)
    print(f"  Total rows to add    : {len(new_df):,}")
    print(f"  Systems              : {sorted(new_df['SystemName'].unique())}")
    print(f"  Season range         : {new_df['Season'].min()}–{new_df['Season'].max()}")
    print(f"  DayNum range         : {new_df['RankingDayNum'].min()}–{new_df['RankingDayNum'].max()}")
    print()
    print("  Sample (10 rows):")
    print(new_df.sample(10, random_state=42).sort_values(["Season","RankingDayNum"]).to_string(index=False))

    print()
    print("=" * 72)
    print("STEP 5  Check against existing MMasseyOrdinals")
    print("=" * 72)
    massey = pd.read_csv(MASSEY)
    print(f"  Existing rows: {len(massey):,}")

    for sys_name in SOS_SYSTEMS.values():
        existing = massey[massey["SystemName"] == sys_name]
        print(f"  Existing {sys_name} rows: {len(existing):,}")

    # Dedup: remove rows already present
    existing_keys = massey[
        massey["SystemName"].isin(SOS_SYSTEMS.values())
    ][["Season", "RankingDayNum", "SystemName", "TeamID"]].drop_duplicates()

    before = len(new_df)
    new_df = new_df.merge(
        existing_keys,
        on=["Season", "RankingDayNum", "SystemName", "TeamID"],
        how="left",
        indicator=True,
    )
    new_df = new_df[new_df["_merge"] == "left_only"].drop(columns="_merge")
    print(f"\n  Rows after dedup: {len(new_df):,} (removed {before - len(new_df):,} duplicates)")

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
    combined = pd.concat([massey, new_df], ignore_index=True)
    combined.to_csv(MASSEY, index=False)
    verify = pd.read_csv(MASSEY)
    print(f"  Written: {len(combined):,} total rows")
    for sys_name in SOS_SYSTEMS.values():
        n = len(verify[verify["SystemName"] == sys_name])
        print(f"  Verified {sys_name}: {n:,} rows in file")
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write to MMasseyOrdinals.csv (default: dry run)")
    args = parser.parse_args()
    main(dry_run=not args.write)
