# March Madness ML

Predict NCAA championship win probabilities for Final Four teams using game data, team metrics, and Kalshi prediction market microstructure.

**2026 Final Four: Illinois, UConn, Michigan, Arizona — tournament in progress (as of 2026-04-05)**

---

## 2026 Predictions

| Team | Seed | Model P(Champion) | Market Implied |
|------|------|-------------------|---------------|
| Michigan | 1 | 35.9% | 37.6% |
| Arizona | 1 | 33.2% | 37.8% |  
| Connecticut | 2 | 18.8% | 10.0% | 
| Illinois | 3 | 12.1% | 14.6% | 

| Team | Seed | Model P(Semi) | Market Implied |
|------|------|-------------------|---------------|
| Michigan | 1 | 51.4% | --% |
| Arizona | 1 | 48.6% | --% |  
| Connecticut | 2 | 56.3% | 47% | 
| Illinois | 3 | 43.7% | 53% | 


Model and market disagree most on Connecticut (model favors strongly) and Arizona (market favors).

---

## How to Run

```bash
conda activate tasty
python -m feature_pipeline.run_v2 --data-dir data/ --output-dir output_v2/
```

Outputs to `output_v2/`: `team_season_features.csv`, `game_pairs.csv`, `cv_results.csv`, `feature_importance.csv`, `2026_predictions.csv`.

---

## Pipeline

The active pipeline is `feature_pipeline/run_v2.py`. It predicts "who wins this game?" across all ~1,400 historical tournament games (not just Final Four), then simulates the 2026 bracket via Monte Carlo.

```
MRegularSeasonDetailedResults.csv ──► build_team_season_features()
MMasseyOrdinals.csv                     (Season, TeamID) → 30+ features
MNCAATourneySeeds.csv                   DayNum ≤ 132 only (no lookahead)
                                              │
                               enrich_with_existing_features()
                                 team sheets, SOS, awards, market
                                              │
MNCAATourneyCompactResults.csv ──► build_game_pairs()
                                    ~1,400 rows (2003–2025)
                                    diff features + binary label
                                              │
                                    train_game_model()
                                    LightGBM, leave-one-year-out CV
                                    CV AUC: 0.773 | Log-Loss: 0.608
                                              │
                                    predict_final_four()
                                    Monte Carlo bracket simulation
                                              │
Kalshi trades ──────────────────► blend_with_market()
  (2025–2026)                       blended probs + model_edge
```

### Key files

| File | Purpose |
|------|---------|
| `feature_pipeline/run_v2.py` | Active entry point |
| `feature_pipeline/game_model.py` | `build_team_season_features`, `build_game_pairs`, `train_game_model`, `predict_final_four` |
| `feature_pipeline/data_loader.py` | `load_all()` — team sheets, Massey, market loading |
| `feature_pipeline/feature_engineering.py` | `build_features()`, feature group constants, PCA |
| `feature_pipeline/market_features.py` | Kalshi VWAP, OFI, momentum, volatility |
| `feature_pipeline/name_resolver.py` | Team name ↔ Kaggle TeamID (0% miss rate on all 279 teams) |
| `feature_pipeline/feature_importance.py` | MDI, MDA, SFI, CFI, PurgedYearKFold |
| `strategy/` | Legacy Final-Four-only pipeline (not actively used) |

---

## Data Sources

| Source | Location | What it has |
|--------|----------|-------------|
| Kaggle game data | `data/kaggle/` | Game-by-game box scores 2003–2026, seeds, 200+ ranking systems |
| Team sheets | `data/team_sheets/` | NET, KPI, SOR, BPI, POM, SAG, SOS splits — Final Four teams 2005–2026 |
| Team stats | `data/{year}-team-stats/` | Season aggregate stats 2003–2026 |
| Kalshi markets | `data/market_data_store/` | Tick-level futures trades 2025–2026 (~696k trades) |
| Labels | `data/yearlys/yearly_champions.csv` | Champions + Final Four finish 1939–2025 |

See `REVIEW.md` for a detailed breakdown of what your sources have that Kaggle does not.  
See `data/kaggle/KAGGLE.md` for Kaggle file schemas.

---

## Model Design

See `MODEL.md` for the full architecture, including:
- Why pairwise (Bradley-Terry style) over rank-4 modeling
- De Prado feature importance suite (MDI, MDA, SFI, CFI)
- Missing data strategy (binary missingness indicators)
- Meta-labeling with Kalshi as primary model

See `FEATURES.md` for the complete feature catalog.

---

## Repository Structure

```
march-madness/
├── feature_pipeline/        # Active ML pipeline
│   ├── run_v2.py            # Entry point
│   ├── game_model.py        # Core model logic
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
├── MODEL.md                 # ML architecture and theory
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
