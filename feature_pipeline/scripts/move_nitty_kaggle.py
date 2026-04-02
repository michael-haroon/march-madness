from __future__ import annotations

import csv
import glob
import os
import re
import argparse
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple, Set

# --- Configuration & Constants ---
DATE_RE = re.compile(r'(\d{1,2})_(\d{1,2})_(\d{4})')
CONFIDENCE_THRESHOLD = 0.88
SECONDARY_GAP = 0.03

# Add your ALIAS_MAP and REPLACEMENTS here (omitted for brevity in this snippet)
# ... [Keep your existing ALIAS_MAP, STOPWORDS, and REPLACEMENTS] ...
ALIAS_MAP = {
    'abilene christian': 'abilene chr',
    'alcorn state': 'alcorn st',
    'appalachian state': 'app state',
    'arizona state': 'arizona st',
    'arkansas little rock': 'ark little rock',
    'arkansas pine bluff': 'ark pine bluff',
    'ball state': 'ball st',
    'bethune cookman': 'bethune cookman',
    'boise state': 'boise st',
    'boston college': 'boston col',
    'bowling green': 'bowling green',
    'brigham young': 'byu',
    'cal state bakersfield': 'cs bakersfield',
    'cal state fullerton': 'cs fullerton',
    'central connecticut state': 'central conn',
    'central michigan': 'central mich',
    'charleston southern': 'charleston so',
    'chicago state': 'chicago st',
    'cleveland state': 'cleveland st',
    'coastal carolina': 'coastal car',
    'college of charleston': 'col charleston',
    'california baptist': 'cal baptist',
    'east tennessee state': 'etsu',
    'eastern illinois': 'e illinois',
    'eastern kentucky': 'e kentucky',
    'eastern michigan': 'e michigan',
    'florida atlantic': 'fl atlantic',
    'florida gulf coast': 'fl gulf coast',
    'florida international': 'fiu',
    'georgia southern': 'ga southern',
    'georgia state': 'ga state',
    'grambling state': 'grambling',
    'illinois chicago': 'uic',
    'jacksonville state': 'jacksonville st',
    'kansas city': 'missouri kc',
    'kennesaw state': 'kennesaw',
    'little rock': 'ark little rock',
    'loyola marymount': 'loy marymount',
    'louisiana': 'louisiana',
    'louisiana monroe': 'ul monroe',
    'louisiana lafayette': 'louisiana',
    'maryland baltimore county': 'umbc',
    'massachusetts lowell': 'ma lowell',
    'mcneese state': 'mcneese st',
    'miami ohio': 'miami oh',
    'mississippi valley state': 'ms valley st',
    'missouri state': 'missouri st',
    'middle tennessee': 'mid tenn st',
    'middle tennessee state': 'mid tenn st',
    'mount saint marys': 'mt st marys',
    'mount st marys': 'mt st marys',
    'north carolina asheville': 'unc asheville',
    'north carolina central': 'nc central',
    'north dakota state': 'north dakota st',
    'northwestern state': 'nw state',
    'ole miss': 'mississippi',
    'omaha': 'nebraska omaha',
    'purdue fort wayne': 'pfw',
    'saint bonaventure': 'st bonaventure',
    'saint francis pennsylvania': 'st francis pa',
    'saint johns': 'st johns',
    'saint josephs': 'st josephs',
    'saint louis': 'st louis',
    'saint marys': 'st marys ca',
    'saint peters': 'st peters',
    'sam houston': 'sam houston st',
    'southeast missouri state': 'se missouri st',
    'southeastern louisiana': 'se louisiana',
    'southern illinois': 's illinois',
    'southern mississippi': 'southern miss',
    'st thomas minnesota': 'st thomas mn',
    'texas am corpus christi': 'tam c christi',
    'texas rio grande valley': 'ut rio grande valley',
    'texas southern': 'tx southern',
    'uc davis': 'uc davis',
    'uc irvine': 'uc irvine',
    'uc riverside': 'uc riverside',
    'uc san diego': 'uc san diego',
    'uc santa barbara': 'uc santa barb',
    'unc greensboro': 'unc greensboro',
    'unc wilmington': 'unc wilmington',
    'unlv': 'nv las vegas',
    'southern california': 'usc',
    'ut arlington': 'tx arlington',
    'ut martin': 'tn martin',
    'utah valley': 'utah valley',
    'virginia commonwealth': 'vcu',
    'western carolina': 'western car',
    'western illinois': 'western ill',
    'western kentucky': 'w kentucky',
    'western michigan': 'w michigan',
    'wichita state': 'wichita st',
    'william mary': 'william mary',
    'youngstown state': 'youngstown st',
}

STOPWORDS = {'university', 'college', 'the'}
REPLACEMENTS = {
    '&': ' and ',
    '-': ' ',
    '.': '',
    ',': '',
    "'": '',
    'st.': 'saint',
    'st ': 'saint ',
    'state': 'st',
    'mount': 'mt',
    'fort': 'ft',
}


@dataclass(frozen=True)
class MatchResult:
    team_id: int
    kaggle_name: str
    score: float
    method: str

class TeamMatcher:
    def __init__(self, teams: Dict[int, dict], by_variant: Dict[str, List[int]]):
        self.teams = teams
        self.by_variant = by_variant
        self._cache: Dict[Tuple[int, str], Optional[MatchResult]] = {}

    def get_match(self, raw_name: str, season: int) -> Optional[MatchResult]:
        cache_key = (season, raw_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._perform_match(raw_name, season)
        self._cache[cache_key] = result
        return result

    def _perform_match(self, raw_name: str, season: int) -> Optional[MatchResult]:
        raw_variants = variants(raw_name)
        
        # 1. Exact Match Check
        for raw_v in raw_variants:
            if raw_v in self.by_variant:
                for team_id in self.by_variant[raw_v]:
                    row = self.teams[team_id]
                    if self._is_active(row, season):
                        return MatchResult(team_id, row['TeamName'], 1.0, 'exact_variant')

        # 2. Filter active teams for fuzzy search (Speed Improvement)
        active_teams = [
            (tid, variants(row['TeamName'])) 
            for tid, row in self.teams.items() 
            if self._is_active(row, season)
        ]

        # 3. Fuzzy Match
        candidates: List[Tuple[float, int, str]] = []
        for team_id, team_vars in active_teams:
            best = max(similarity(rv, tv) for rv in raw_variants for tv in team_vars)
            candidates.append((best, team_id, 'fuzzy'))

        candidates.sort(reverse=True)
        if not candidates:
            return None

        best_score, best_team_id, method = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0

        if best_score >= CONFIDENCE_THRESHOLD and (best_score - second_score) >= SECONDARY_GAP:
            return MatchResult(best_team_id, self.teams[best_team_id]['TeamName'], best_score, method)

        return None

    @staticmethod
    def _is_active(team_row: dict, season: int) -> bool:
        first = int(team_row.get('FirstD1Season') or 0)
        last = int(team_row.get('LastD1Season') or 9999)
        return (not first or season >= first) and (not last or season <= last)

# --- Helper Functions ---

def normalize_name(name: str) -> str:
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii').lower().strip()
    for old, new in REPLACEMENTS.items():
        s = s.replace(old, new)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    parts = [p for p in s.split() if p and p not in STOPWORDS]
    return ' '.join(parts)

def variants(name: str) -> List[str]:
    base = normalize_name(name)
    vals = {base}
    if base in ALIAS_MAP:
        vals.add(normalize_name(ALIAS_MAP[base]))
    # Pre-calculated inverse alias mapping would be faster if ALIAS_MAP is huge
    return sorted(v for v in vals if v)

def similarity(a: str, b: str) -> float:
    if a == b: return 1.0
    # Quick length check to skip obviously different names
    if abs(len(a) - len(b)) > 5: return 0.0 
    return SequenceMatcher(None, a, b).ratio()

def build_team_index(mteams_path: str) -> Tuple[Dict[int, dict], Dict[str, List[int]]]:
    teams = {}
    by_variant = defaultdict(list)
    with open(mteams_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            tid = int(row['TeamID'])
            teams[tid] = row
            for v in variants(row['TeamName']):
                by_variant[v].append(tid)
    return teams, by_variant

# --- Processing Logic ---

def load_nitty_data(nitty_dir: str, matcher: TeamMatcher, dayzero_map: Dict[int, datetime]):
    values = {}
    m_logs, u_logs = [], []
    files = sorted(glob.glob(os.path.join(nitty_dir, '**', '*.csv'), recursive=True))

    for path in files:
        # Season logic
        season = None
        for part in os.path.normpath(path).split(os.sep):
            if re.fullmatch(r'\d{4}', part):
                season = int(part)
                break
        
        if season is None:
            file_year = parse_snapshot_date(path).year
            season = file_year if file_year in dayzero_map else file_year - 1

        if season not in dayzero_map:
            u_logs.append(f"SKIP\t{path}\tNo dayzero mapping")
            continue

        daynum = (parse_snapshot_date(path) - dayzero_map[season]).days

        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            cols = {c.strip(): c for c in reader.fieldnames or []}
            t_col, s_col, nc_col = cols.get('Team'), cols.get('NETSOS'), cols.get('NETNonConfSOS')

            if not all([t_col, s_col, nc_col]):
                continue

            for row in reader:
                raw_team = (row.get(t_col) or '').strip()
                if not raw_team: continue

                match = matcher.get_match(raw_team, season)
                if not match:
                    u_logs.append(f"UNMATCHED\t{season}\t{daynum}\t{raw_team}")
                    continue

                values[(season, daynum, match.team_id)] = {
                    'NETSOS': (row.get(s_col) or '').strip(),
                    'NETNonConfSOS': (row.get(nc_col) or '').strip(),
                }
                m_logs.append(f"MATCH\t{season}\t{raw_team} -> {match.kaggle_name} ({match.score:.2f})")

    return values, m_logs, u_logs

def augment_massey(mmassey_path: str, output_path: str, nitty_values: Dict[Tuple[int, int, int], Dict[str, str]]):
    with open(mmassey_path, newline='', encoding='utf-8-sig') as src, open(output_path, 'w', newline='', encoding='utf-8') as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        for col in ['NETSOS', 'NETNonConfSOS']:
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            key = (int(row['Season']), int(row['RankingDayNum']), int(row['TeamID']))
            extra = nitty_values.get(key, {})
            row['NETSOS'] = extra.get('NETSOS', '')
            row['NETNonConfSOS'] = extra.get('NETNonConfSOS', '')
            writer.writerow(row)

def load_dayzero_map(mseasons_path: str) -> Dict[int, datetime]:
    out = {}
    with open(mseasons_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        dayzero_col = next((c for c in reader.fieldnames if c.replace(' ', '').lower() == 'dayzero'), None)
        if not dayzero_col:
            raise ValueError('Could not find DayZero column in MSeasons.csv')
        for row in reader:
            out[int(row['Season'])] = datetime.strptime(row[dayzero_col], '%m/%d/%Y')
    return out

def parse_snapshot_date(path: str) -> Optional[datetime]:
    filename = os.path.basename(path)
    m = DATE_RE.search(filename)
    if not m:
        # Instead of 'initial', return None or print a warning
        print(f"Warning: Could not find date in filename '{filename}'. Skipping.")
        return None
    
    month, day, year = map(int, m.groups())
    return datetime(year, month, day)

# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description="Augment Massey Ordinals with NET SOS data.")
    parser.add_argument("--execute", action="store_true", help="Actually write files (default is Dry Run)")
    parser.add_argument("--mmassey", default='/Users/michaelharoon/Projects/tasty/march-madness/data/kaggle/MMasseyOrdinals.csv')
    parser.add_argument("--nitty", default='data/raw/nitty_gritty/csvs')
    parser.add_argument("--mteams", default='/Users/michaelharoon/Projects/tasty/march-madness/data/kaggle/MTeams.csv')
    parser.add_argument("--mseasons", default='/Users/michaelharoon/Projects/tasty/march-madness/data/kaggle/MSeasons.csv')
    parser.add_argument("--output", default='/Users/michaelharoon/Projects/tasty/march-madness/data/test/MMasseyOrdinals_with_NETSOS.csv')
    args = parser.parse_args()

    print(f"--- {'EXECUTION' if args.execute else 'DRY RUN'} MODE ---")
    
    # 1. Load Indexes
    teams, by_variant = build_team_index(args.mteams)
    dayzero_map = load_dayzero_map(args.mseasons)
    matcher = TeamMatcher(teams, by_variant)

    # 2. Process Nitty Gritty
    print("Processing Nitty Gritty files...")
    nitty_values, m_logs, u_logs = load_nitty_data(args.nitty, matcher, dayzero_map)
    
    print(f"Matches found: {len(nitty_values)}")
    print(f"Unmatched entries: {len(u_logs)}")

    if args.execute:
        print(f"Writing output to {args.output}...")
        augment_massey(args.mmassey, args.output, nitty_values)
        # write_lines(match_log, m_logs) etc...
    else:
        print("Dry run complete. No files were written. Use --execute to save results.")

if __name__ == "__main__":
    main()