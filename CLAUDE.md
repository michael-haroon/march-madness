# CLAUDE.md — March Madness ML Project

Instructions and context for Claude Code working in this repository.

Always use conda env `tasty`. it has pandas and python.

---

## Project Goal

Build an ML model to predict the probability of winning the NCAA championship for Final Four teams. The label is who wins the championship (or finish rank). Features come from three domains: team sheet metrics, Kaggle game data, and Kalshi market microstructure.

The **2026 Final Four is live now** (Illinois, UConn, Michigan, Arizona) — treat 2026 as test/prediction data, not training data.

---

## Data Sources — Quick Reference

### `data/team_sheets/`
Per-team committee metrics for Final Four teams, 2005–2026 (no 2020). Two formats:
- `{year}_Team_Sheets_Selection.csv` — snapshot at Selection Sunday (preferred, no lookahead)
- `{year}_Team_Sheet_Final.csv` — slight lookahead bias from conf tournament games

**Schema changed in 2021**: pre-2021 uses `Record`, `Avg_RPI_Win/Loss`, `SOS_D1/NonConf`; post-2021 uses `NET_Rank`, `Overall_Record`, `Conference_Record`, `Road_Record`, `NET_SOS`, `NET_NonConf_SOS`, `Avg_NET_Wins/Losses`.

### `data/kaggle/`
See `data/kaggle/DATA.md` for full documentation. Key files:
- `MRegularSeasonDetailedResults.csv` — game-by-game box scores, 2003–2026
- `MRegularSeasonCompactResults.csv` — game results back to 1985
- `MNCAATourneySeeds.csv` — seeds 1985–2026
- `MMasseyOrdinals.csv` — 200+ ranking systems, 2003–2026
- `MTeamSpellings.csv` — maps external name spellings to canonical `TeamID`
- `MTeamCoaches.csv` — coaching history
- `MConferenceTourneyGames.csv` — conf tournament game identification

**DayNum reference**: Selection Sunday = 132, Final Four = 152, Championship = 154. All regular season features must use DayNum ≤ 132 to avoid lookahead.

### `data/market_data_store/`
Kalshi futures trade data, 2025–2026 tournaments only.
- `historical-endpoint/year=2025/ticker=KXMARMAD-25-{ABBREV}/` — 2025 season historical trades
- `markets-endpoint/year=2025/` and `year=2026/` — current market trades
- `kalshi_name_maping.csv` — maps Kalshi tickers to team names
- Schema: `trade_id, yes_price_dollars, no_price_dollars, count_fp, taker_side, created_time`
- 696,167 total trades; covers all ~68 tournament teams per year, not just Final Four
- `taker_side` = direction of aggressor (yes/no buyer)

### `data/yearlys/`
- `yearly_champions.csv` — champion + 2nd/3rd/4th place, 1939–2025 (use as label)
- `yearly_award_winners.csv` — USBWA_Team is safe pre-tournament feature; Naismith/Wooden/AP/BT are post-tournament, drop them
- `yearly_sporting_news_player.csv` — pre-tournament, safe
- `yearly_championship_location.csv` — redundant with Kaggle `MGameCities.csv`

### `data/{year}-team-stats/`
Per-team season aggregate stats, 2003–2026. Multiple redundant files per year; see README for which files to skip. Use 2023+ as reference for clean schema. Note: no 2020.

---

## Key Data Facts to Know

### What you have that Kaggle does NOT

| Metric | Your coverage | Kaggle gap |
|---|---|---|
| SOR | 2005–2026 | Not in Kaggle at all |
| BPI | 2021–2026 | Kaggle only has 2009–2013 |
| KPI | 2005–2026 | Kaggle missing 2021–2023 |
| NET | 2005–2026 | Kaggle only has 2019–2025 |
| SAG | 2005–2026 | Kaggle only through 2023 |
| All SOS splits | 2005–2026 | Not in Kaggle |
| Road_Record, conf/nonconf records | 2021–2026 | Not in Kaggle |
| RB_WAB, PM_T-Rank | 2025–2026 | Not in Kaggle |
| Kalshi market microstructure | 2025–2026 | Not in Kaggle |

### NET column 2005–2018 is suspicious
NET was introduced in the 2018–2019 season. The `NET` column in pre-2019 team sheets is likely retroactive estimates, not the official committee metric. Do not treat it as equivalent to 2019+ NET. Use a `NET_era_flag` or just exclude pre-2019 NET values.

### No 2020 data
The 2020 tournament was cancelled (COVID). All year-indexed loops must skip 2020.

### Kalshi data is tick-level, not a price lookup
Do not treat it as a single `yes_price` per team. Compute derived features: VWAP, price momentum, order flow imbalance (yes taker volume / total volume), trade count, price volatility.

---

## Team Name Mapping

**Always join through `TeamID`, never through team name strings.**

Use `data/kaggle/MTeamSpellings.csv` as the primary name → TeamID mapper. It has 1,178 entries but does not cover all edge cases (e.g., "Loyola Chicago" vs "Loyola (Ill.)"). Use fuzzy matching as a fallback for unmatched names.

---

## Lookahead Rules

- All game-derived features: DayNum ≤ 132 only
- Team sheets: prefer `_Selection.csv` files; flag `_Final.csv` as slightly biased
- Award features: USBWA and Sporting News are safe; drop Naismith, Wooden, AP, BT
- Market features: use last price/trade before championship tip-off (DayNum 154)
- Never use tournament game outcomes as feature inputs

---

## Modeling Notes

- **Sample size**: ~84 rows if Final Four only (21 seasons × 4 teams). Very small.
- **CV scheme**: Leave-one-year-out (21 folds). With 84 rows this gives ~4 training samples per fold — treat feature importance results as directional only.
- **Feature selection order**: SFI first (no collinearity concerns), then MDI + Clustered MDA on survivors
- **Label**: champion flag (binary 1/4) or finish rank (1–4). Define before building feature matrix.
- **Market as primary model**: Kalshi implied probability (VWAP of yes_price) can serve as the "white box" primary model in a meta-labeling setup. Your team/game features are the secondary model identifying when the market is wrong.
- **Do not use MDI/MDA on Kalshi features directly** — only 2 years of Final Four data (8 rows). Use SFI only, or treat market probability as a single composite feature.

---

## Warnings

- `{year}-team-stats/` has many redundant files per year — see README for the skip list
- 2022 and earlier team-stats have cross-file redundancy; use 2023+ schema as reference
- `yearly_champions.csv` does not yet have 2026 results (tournament in progress)
- The `markets-endpoint` has both 2025 and 2026 tickers; `historical-endpoint` only has `year=2025` directory but contains tickers for both seasons
- 2026 Kaggle `MNCAATourneyCompactResults.csv` has NO tournament results yet — Kaggle hasn't updated. Tier 1 labels (survival) only go 2003–2025. The 2026 Final Four is set manually: Illinois, Connecticut, Michigan, Arizona.
- Kalshi maps UConn as `"UConn"` (not `"Connecticut"`). `normalise_team()` from `data_loader.py` handles this, but always verify market features join correctly via `mkt['team'].isin(ff_teams)` check.
- The 2021 tournament had unusual DayNum scheduling (bubble in Indianapolis). E8 was DayNum 147–148 instead of 145–146. The `load_tournament_labels()` function handles this by deriving `made_ff` directly from DayNum=152 participants rather than DayNum ranges.

---

## Pipeline Architecture (updated 2026-04-03)

All pipeline code lives in `feature_pipeline/`. The module was previously called `pipeline/`.

### Active pipeline: `run_v2.py` + `game_model.py`

**Entry point:** `python -m feature_pipeline.run_v2`

Data flow:
1. `game_model.build_team_season_features(data_dir)` — builds one feature vector per (Season, TeamID) using **only DayNum ≤ 132** (pre-tournament). Reads Kaggle CSVs directly; does NOT call `load_all`.
2. `data_loader.load_all(include_kaggle=False)` via `enrich_with_existing_features` — merges team sheet metrics, award flags, quadrant features. No Kaggle game stats here so no DayNum issue.
3. `game_model.build_game_pairs(team_df, tourney_results)` — constructs one row per tournament game (R1–Championship) with diff features. Label = did team A win THIS game.
4. `game_model.train_game_model(pairs_df)` — LightGBM with leave-one-year-out CV on ~1,400 tournament games (2003–2025).
5. `game_model.predict_final_four(model, team_df_2026, ff_ids, n_sims=50000)` — Monte Carlo bracket simulation to derive championship probabilities.
6. `market_features.blend_with_market(preds, mkt_2026, market_weight=0.3)` — blends model probs with Kalshi VWAP.

Outputs to `output_v2/`: `team_season_features.csv`, `game_pairs.csv`, `cv_results.csv`, `feature_importance.csv`, `2026_predictions.csv`.

### Legacy pipeline: `run.py` (not actively used)

`run.py` is a standalone Final-Four-only pipeline (84 rows, de Prado MDI/MDA/SFI on champion_flag). It uses `data_loader.load_all(include_kaggle=True)` and a Bradley-Terry pairwise model. **Do not confuse with run_v2.py** — they share `data_loader.py` and `feature_engineering.py` but are otherwise independent.

### Key files
- `feature_pipeline/game_model.py` — core of the active pipeline. `build_team_season_features` is where DayNum ≤ 132 filtering lives for the run_v2 path.
- `feature_pipeline/data_loader.py` — `load_all()` accepts `include_kaggle`, `include_team_stats`, `include_market` flags. Used by both pipelines for team sheet / enrichment loading.
- `feature_pipeline/feature_engineering.py` — `build_features()`, feature group constants (`TIER1_FEATURES`, `KAGGLE_FEATURES`, `TOURNEY_PATH_FEATURES`, etc.), cross-source reconciliation, PCA.
- `feature_pipeline/name_resolver.py` — bidirectional team name ↔ Kaggle TeamID. 0% miss rate on all 279 tournament teams (2003–2026).
- `feature_pipeline/market_features.py` — Kalshi microstructure loader. `load_kalshi_trades` + `compute_market_features` → VWAP, OFI, momentum, volatility, trade count.

### Cross-source reconciliation findings
`reconcile_cross_source()` always drops ts_ in favor of kg_ — no correlation gate. Kaggle covers all years with consistent game-level aggregation; ts_ is spot data for a subset of years. Correlation is printed as a diagnostic only.
- **All ts_ equivalents dropped**: `ts_fg_pct`, `ts_fg_pct_def`, `ts_ft_pct`, `ts_three_pct`, `ts_scoring_margin`, `ts_efg_pct`, `ts_off_reb_pg`, `ts_def_reb_pg`, `ts_assists_pg`, `ts_steals_pg`, `ts_blocks_pg`
- Note: `kg_fg_pct` vs `ts_fg_pct` correlation is low (r≈0.50) because ts_ includes tournament games through E8 while kg_ is regular season only — but we still prefer kg_ for consistency.

### Team-stats PCA findings
7 components explain 93.1% variance of the 11 retained ts_* features:
- `ts_pc1`: rebound margin, 3pt% defense, fouls per game
- `ts_pc2`: FG%, fouls, offensive rebounds
- `ts_pc3`: FG%, fouls, rebound margin
- Interpretation: PC1 ≈ defensive toughness/physical play, PC2 ≈ offensive efficiency

### 2026 Final Four market snapshot (as of March 30, 2026)
| Team | VWAP | Trade Count | Implied Prob (normalized) |
|---|---|---|---|
| Arizona | 0.2186 | 93,584 | ~38% |
| Michigan | 0.2169 | 78,847 | ~37% |
| Illinois | 0.0843 | 16,286 | ~15% |
| Connecticut | 0.0579 | 31,689 | ~10% |

Arizona and Michigan are nearly tied as market favorites (~37-38% each). Illinois is a moderate underdog (15%). Connecticut is the longest shot at ~10%. All four teams now resolve correctly via `normalise_team()` — Kalshi's "UConn" maps to "Connecticut".
