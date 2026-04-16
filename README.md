# March Madness ML

Predict NCAA championship win probabilities for Final Four teams using game data, team metrics, and Kalshi prediction market microstructure.

**2026 Final Four: Illinois, UConn, Michigan, Arizona — tournament in progress (as of 2026-04-05)**

---

## 2026 Predictions

| Team | Seed | Model P(Champion) |
|------|------|-------------------|
| Michigan | 1 | 35.9% |
| Arizona | 1 | 33.2% |
| Connecticut | 2 | 18.8% |
| Illinois | 3 | 12.1% |

| Team | Seed | Model P(Semi) |
|------|------|---------------|
| Michigan | 1 | 51.4% |
| Arizona | 1 | 48.6% |
| Connecticut | 2 | 56.3% |
| Illinois | 3 | 43.7% |

---

## How to Run

```bash
conda activate tasty
python -m feature_pipeline.run_v2 --data-dir data/ --output-dir output_v2/
```

Outputs to `output_v2/`: `team_season_features.csv`, `game_pairs.csv`, `feature_importance_catalog.csv`, `filtered/feature_list.txt`.

---

## Pipeline

The active pipeline is `feature_pipeline/run_v2.py`. It frames the problem as pairwise game prediction — "who wins this game?" — across all ~1,400 historical tournament games (2003–2025), then runs de Prado feature importance analysis to identify the most predictive signals.

```
MRegularSeasonDetailedResults.csv ──► build_team_season_features()
MMasseyOrdinals.csv                     (Season, TeamID) → 30+ features
MNCAATourneySeeds.csv                   DayNum ≤ 132 only (no lookahead)
                                              │
                               enrich_with_existing_features()
                                 team sheets, SOS, awards
                                              │
MNCAATourneyCompactResults.csv ──► build_game_pairs()
                                    ~1,400 rows (2003–2025)
                                    diff features + binary label
                                              │
                                    run_all_importance()
                                    de Prado: ONC → MDI → MDA → SFI
                                    feature survivors → filtered/feature_list.txt
```

Kalshi market microstructure features (VWAP, OFI, momentum, volatility, trade count) are computed from ~696k tick-level trades and included in the feature matrix for importance analysis. They are treated as features on equal footing with game and team-sheet metrics — not as a separate model or blending signal.

### Key files

| File | Purpose |
|------|---------|
| `feature_pipeline/run_v2.py` | Active entry point |
| `feature_pipeline/game_model.py` | `build_team_season_features`, `build_game_pairs` |
| `feature_pipeline/data_loader.py` | `load_all()` — team sheets, Massey, market loading |
| `feature_pipeline/feature_engineering.py` | `build_features()`, feature group constants, PCA |
| `feature_pipeline/market_features.py` | Kalshi VWAP, OFI, momentum, volatility — from raw tick data |
| `feature_pipeline/feature_importance.py` | MDI, MDA, SFI, CFI, PurgedYearKFold |
| `feature_pipeline/name_resolver.py` | Team name ↔ Kaggle TeamID (0% miss rate on all 279 teams) |
| `strategy/` | Legacy Final-Four-only pipeline (not actively used) |

---

## Data Sources

| Source | Location | What it has |
|--------|----------|-------------|
| Kaggle game data | `data/kaggle/` | Game-by-game box scores 2003–2026, seeds, 200+ ranking systems |
| Team sheets | `data/team_sheets/` | NET, KPI, SOR, BPI, POM, SAG, SOS splits — Final Four teams 2005–2026 |
| Team stats | `data/{year}-team-stats/` | Season aggregate stats 2003–2026 |
| Kalshi markets | `data/market_data_store/` | Tick-level futures trades 2025–2026 (~696k trades, all ~68 tournament teams) |
| Labels | `data/yearlys/yearly_champions.csv` | Champions + Final Four finish 1939–2025 |

See `data/kaggle/KAGGLE.md` for Kaggle file schemas.

---

## Model Design

The active pipeline (`run_v2.py`) frames the problem as **pairwise game prediction** rather than rank-4 modeling — more training data (~1,400 game rows vs ~84 Final Four rows) and better generalization.

Key design choices:
- **Pairwise over rank-4**: avoids the small-sample problem of Final-Four-only models
- **Leave-one-year-out CV**: non-overlapping folds, no data leakage
- **Feature importance**: de Prado suite — MDI, MDA, SFI with ONC clustering. Kalshi features evaluated on equal footing via SFI (limited to 2 years of data, so MDI/MDA have wide error bars)
- **Missing data**: binary missingness indicators alongside imputed values
- **Market microstructure**: Kalshi tick data processed into VWAP, OFI, price momentum, and volatility features. These enter the feature matrix as inputs — not as a separate blending model

See `FEATURES.md` for the complete feature catalog.

---

## Repository Structure

```
march-madness/
├── feature_pipeline/        # Active ML pipeline
│   ├── run_v2.py            # Entry point
│   ├── game_model.py        # Feature construction
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── feature_importance.py
│   ├── market_features.py
│   ├── name_resolver.py
│   ├── season_utils.py
│   └── config.py
├── strategy/                # Legacy Final-Four-only pipeline
├── data/
│   ├── kaggle/              # Kaggle competition data
│   ├── team_sheets/         # Committee metrics (NET, KPI, SOR, BPI...)
│   ├── {year}-team-stats/   # Season aggregate stats (2003–2026, no 2020)
│   ├── market_data_store/   # Kalshi prediction market trades
│   └── yearlys/             # Champions, awards, locations
├── output_v2/               # Latest pipeline outputs
├── README.md
├── FEATURES.md              # Feature catalog
├── REVIEW.md                # Data source audit and lookahead analysis
└── CLAUDE.md                # Claude Code instructions
```

---

## Key Constraints

- **No 2020**: Tournament cancelled (COVID). All loops skip 2020.
- **DayNum ≤ 132**: All pre-tournament features must use only regular season games.
- **2026 is test data only**: Do not use 2026 tournament results as training labels.
- **Treat 2026 as test**: `yearly_champions.csv` does not yet have 2026 results.
