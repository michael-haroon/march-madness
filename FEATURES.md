# FEATURES.md — March Madness ML Feature Reference

## 1. ML Architecture Overview

### The Three-Tier Label Design

```
Tier 1 (feature selection)
  Input:  ~1,400 rows — all 68 seeded tournament teams × 21 years
  Label:  made_ff (binary: did this team make the Final Four?)
  Use:    SFI to identify which features have standalone signal at all

Tier 2 (prediction)
  Input:  ~252 rows — 12 ordered pairs × 21 years (4 FF teams → 6 pairs × 2 orderings)
  Label:  a_wins (binary: did team A finish better than team B?)
  Model:  LightGBM binary classifier on feature DIFFERENCES (A_metric - B_metric)
  Use:    Championship win probability

Tier 3 (confidence)
  Input:  4 rows/year — Final Four teams only
  Label:  meta_label (1 if primary model's pick actually won)
  Primary model: Kalshi VWAP for 2025–2026, consensus_rank for 2005–2024
  Use:    "When does the market/ranking favorite actually win?"
```

### Why Pairwise?

The Final Four is a 4-team ranking problem (1st–4th place). Pairwise conversion:
- Turns 4 rows/year into 12 ordered pairs/year (both orderings of each pair)
- Makes the target unambiguous binary: did A finish better than B?
- Features become *differences*: `diff_net_rank = net_rank_A - net_rank_B`
- Multiplies training data by 3× without adding noise

### The Base Model

**Random Forest** (`pipeline/feature_importance.py: build_rf()`):
- `DecisionTreeClassifier(max_features=1, criterion="entropy")`
- Wrapped in `BaggingClassifier(n_estimators=1000)`
- `max_features=1` is de Prado's key parameter — every split considers exactly one random feature, giving all features equal opportunity to be selected

**LightGBM** (`pipeline/model.py`) for the final pairwise prediction:
- `objective="binary"`, `max_depth=4` (shallow to prevent overfit on 252 rows)
- `subsample=0.8, colsample_bytree=0.8` (further regularization)
- `class_weight="balanced"` (champions are rare — 1/4 = 25% positive rate)

### Cross-Validation

**PurgedYearKFold** (`pipeline/feature_importance.py: PurgedYearKFold`):
- Leave-one-year-out: train on 20 years, test on 1 year
- No temporal leakage — each row is a full season, no game-by-game overlap to purge
- Applied in MDI (in-sample only), MDA, SFI, CFI, and LightGBM pairwise CV

---

## 2. Missing Data — Does the Model "Know" When a Feature is Missing?

**Yes, explicitly.** The strategy is from de Prado AFML Ch.19 — binary missingness indicators.

In `pipeline/feature_engineering.py: handle_missing()`:

```python
if method == "indicator":
    df[f"{col}_missing"] = is_null.astype(int)   # NEW column: 1 = was missing
    fill_val = df[col].median()
    df[col] = df[col].fillna(fill_val)            # fill with median
```

Every feature marked `"indicator"` in `MISSING_STRATEGY` (`pipeline/config.py`) gets a paired `{col}_missing` column added. So for `net_rank`:
- `net_rank` → filled with median (keeps the column numeric)
- `net_rank_missing` → 1 for 2005–2018 (pre-NET era), 0 for 2019+

**What this means for the model:**
The tree can now learn "if `net_rank_missing=1` AND `kpi=3`, this team is likely strong (committee would have given a good NET if it existed)." Missing itself becomes signal. In particular:
- Pre-2021 teams have `bpi_missing=1` — the model can learn this means "older era"
- Teams without Kalshi market data have `mkt_vwap_missing=1` — the model can learn the market's absence is informative (pre-2025)
- A team with `q1_win_pct_missing=1` didn't have quadrant data — probably an older year

**What NOT to do:** Simply dropping missing rows or filling with 0 would discard this information. Filling with median + adding a binary flag preserves both the distribution of non-missing values and the information that the value was absent.

---

## 3. Collinearity — How It's Handled

Collinearity is the central challenge in this dataset. NET rank, POM, SAG, BPI, KPI, and SOR are all trying to measure "how good is this team" — they're highly correlated.

### Three mechanisms address it:

**a) Cross-source reconciliation** (`pipeline/feature_engineering.py: reconcile_cross_source()`)

Before importance analysis, pairs of features from different sources (Kaggle vs team-stats) are compared. If Pearson r > 0.95, the redundant one is dropped. From our run:
- `kg_opp_fg_pct` vs `ts_fg_pct_def`: r=0.985 → dropped ts_ version
- `kg_fg_pct` vs `ts_fg_pct`: r=0.501 → KEPT BOTH (different because ts_ includes tournament games)

This is structural deduplication — not statistical, just logical.

**b) ONC clustering + Clustered Feature Importance (CFI)** (`pipeline/feature_importance.py: onc_cluster(), feat_imp_cfi_mda()`)

ONC (Optimal Number of Clusters) groups features by their correlation structure using silhouette t-stat as the quality criterion. Then CFI shuffles an *entire cluster at once* during MDA permutation.

Why this matters: if POM and SAG are in the same cluster (r=0.95), shuffling only POM barely hurts accuracy because SAG compensates. The model thinks POM is useless. But if you shuffle POM + SAG simultaneously, neither can compensate for the other — you get the true combined importance of the "predictive ranking" cluster.

From our last run, ONC found 3 clusters:
- **Cluster 1 (rankings)**: Massey systems + consensus_rank + pred_consensus + rank_spread — all measuring "what do the analytical models think of this team?"
- **Cluster 2 (performance)**: scoring margin, FG%, tournament path, rebounding, records — what the team actually did on the court
- **Cluster 0 (efficiency/pace)**: turnovers, steals, blocks, individual box stats, NET/KPI/SOR

**c) SFI as collinearity-free baseline** (`pipeline/feature_importance.py: feat_imp_sfi()`)

Single Feature Importance trains the model on ONE feature at a time. Zero collinearity is possible. If a feature scores high on SFI but low on MDA, the interpretation is: "this feature has standalone signal, but it's already captured by something else in the feature set." That's useful information — it tells you features are substitutable.

A feature that scores high on BOTH SFI and MDA is genuinely unique — nothing else captures what it captures.

---

## 4. Feature Combinations — Interaction Effects

The model learns combinations implicitly through the tree structure. A decision tree split like:

```
if consensus_rank < 5:
    if kg_tourney_avg_margin > 10:
        predict champion  ← high probability
    else:
        predict runner-up
```

...captures the interaction: "top-ranked team that dominated tournament games." Neither feature alone predicts this.

**What we do and don't explicitly test:**
- MDI captures interactions because trees are hierarchical — feature B may only matter when feature A splits left
- MDA does NOT test interactions — it shuffles one feature (or cluster) at a time
- Friedman's H-statistic for explicit pairwise interaction testing is NOT implemented — with 52 features that's 1,326 pairs, computationally expensive and statistically noisy at n=78

The tree depth (`max_depth=4` for LightGBM) limits interaction depth to 4 levels. This is intentional — with 78 Final Four rows, deep trees overfit.

---

## 5. Synthetic Validation — How It Works

(`pipeline/feature_importance.py: synthetic_validation()`)

Before running importance analysis on real data, we verify the MDI/SFI methods can recover known signal. Here's exactly what it does:

**Step 1: Generate data with known ground truth**
```python
X_raw, y = make_classification(
    n_samples=500, n_features=n_informative + n_noise,
    n_informative=3, n_redundant=0, shuffle=False
)
```
- `INFO_0`, `INFO_1`, `INFO_2` — these three features actually cause the label
- `NOISE_0`, `NOISE_1` — random, zero relationship to the label

**Step 2: Add synthetic redundant features**
```python
noisy = X_raw[:, src] + rng.normal(0, 0.5, n_samples)
```
- `REDUND_0`, `REDUND_1`, ... — noisy copies of INFO features (like POM ≈ SAG in real data)

**Step 3: Run MDI and SFI on the synthetic data**
Train the Random Forest on all features, compute MDI. Run SFI separately.

**Step 4: Check if INFO features rank above NOISE features**
```python
info_rank  = mdi_result.index.get_indexer([f"INFO_{i}" for i in range(n_informative)])
noise_rank = mdi_result.index.get_indexer([f"NOISE_{i}" for i in range(n_noise)])
mdi_pass   = max(info_rank) < min(noise_rank)
```

If all 3 INFO features rank above both NOISE features → `mdi_pass = True`. This validates that our importance methods are correctly identifying signal vs noise before we trust their output on real basketball data.

**Why this matters:** Feature importance methods can produce misleading results if implemented incorrectly (biased toward high-cardinality features, etc.). Running on synthetic data first is a sanity check — "does our MDI actually find known signal?"

---

## 6. Feature Catalog

### Feature Availability by Source and Era

| Era | Years | Sources Available |
|---|---|---|
| Pre-NET | 2005–2015 | RPI, SOS splits, records, Kaggle game stats, team-stats core |
| KPI/SOR era | 2016–2019 | + KPI, SOR, team-stats extended |
| NET era | 2021+ | + NET, BPI, quadrant records |
| Market era | 2025–2026 | + Kalshi microstructure |
| Tournament path | 2003–2025 (Kaggle) | Requires tournament results to be in Kaggle |

---

### TIER 1 FEATURES (regular season only, safe for all-round survival prediction)

#### Team Sheet Rankings (`CORE_RANK_FEATURES`)

| Feature | Description | Years | Missing strategy |
|---|---|---|---|
| `net_rank` | NCAA Evaluation Tool rank (official committee metric) | 2019–2026 | indicator |
| `kpi` | Kevin Pauga Index — win-probability-based ranking | 2016–2026 | indicator |
| `sor` | Strength of Record — probability of replicating your record vs average team | 2005–2026 | indicator |
| `bpi` | ESPN Basketball Power Index — predictive | 2021–2026 (Kaggle: 2009–2013 only) | indicator |
| `pom` | Ken Pomeroy's efficiency-based rating | 2005–2026 | indicator |
| `sag` | Jeff Sagarin rating | 2005–2026 (gap: 2024 missing) | indicator |
| `consensus_rank` | Mean of available predictive metrics (BPI, POM, SAG) | derived | zero |
| `pred_consensus` | Mean of predictive-only metrics | derived | zero |
| `rank_spread` | Std of all metrics — disagreement = uncertainty | derived | zero |

**Note on NET 2005–2018**: The team sheets have a `NET` column going back to 2005, but NET wasn't introduced until 2019. Those values are retroactive estimates — treat with caution. `net_rank_missing=1` for those years.

#### Strength of Schedule (`SOS_FEATURES`)

| Feature | Description | Years |
|---|---|---|
| `net_sos` | NET-based SOS score | 2021–2026 |
| `net_nc_sos` | NET non-conference SOS | 2021–2026 |
| `rpi_sos` | RPI-based overall SOS (or pre-2021 `SOS_D1`) | 2005–2026 |
| `rpi_nc_sos` | RPI non-conference SOS | 2005–2026 |
| `avg_net_wins` | Average NET rank of your wins (quality of wins) | 2021–2026 |
| `avg_net_losses` | Average NET rank of your losses (quality of losses) | 2021–2026 |

#### Record Breakdowns (`RECORD_FEATURES`)

| Feature | Description | Years |
|---|---|---|
| `overall_win_pct` | W / (W+L) | all |
| `nc_win_pct` | Non-conference win % | all |
| `road_win_pct` | Road win % | all |
| `conf_win_pct` | Conference win % | 2021–2026 |
| `q1_win_pct` | Win % vs Quadrant 1 opponents (hardest games) | 2021–2026 |
| `resume_score` | Q1W×4 + Q2W×2 + Q3W×1 − Q1L×3 − Q2L×2 − Q3L×1 − Q4L×0.5 | 2021–2026 |

#### de Prado Financial Analogs (`DEPRADO_FEATURES`)

| Feature | Financial analog | Description |
|---|---|---|
| `win_entropy` | Shannon entropy of price series | Disorder in win/loss sequence. Pure winning streak → 0. Alternating → 1. Measures consistency. |
| `cusum_peak` | CUSUM filter (de Prado Ch.2) | Largest detected momentum shift in a team's win-rate time series across years. High = team that surged or collapsed. |
| `net_rank_yoy` | Fracdiff (de Prado Ch.5) | Year-over-year NET rank improvement (−rank_diff, so positive = got better). Captures trajectory, not just position. |
| `kpi_yoy`, `sor_yoy`, `bpi_yoy`, `pom_yoy` | Same | YoY improvement for each metric |

#### Awards (`AWARD_FEATURES`)

| Feature | Description | Lookahead safe? |
|---|---|---|
| `total_awards` | Sum of USBWA + Sporting News player of year flags | Yes — both pre-tournament |
| `has_usbwa_award` | Binary: team had the USBWA Player of Year | Yes |
| `has_sporting_news_award` | Binary | Yes |
| ~~`has_naismith_award`~~ | ~~Naismith Trophy~~ | ~~NO — post-tournament~~ |
| ~~`has_wooden_award`~~ | ~~Wooden Award~~ | ~~NO — post-tournament~~ |

#### Kaggle Regular Season Stats (`KAGGLE_FEATURES`, DayNum ≤ 132)

| Feature | Description | Available |
|---|---|---|
| `kg_fg_pct` | Field goal % (regular season only) | 2003–2026 |
| `kg_fg3_pct` | 3-point field goal % | 2003–2026 |
| `kg_ft_pct` | Free throw % | 2003–2026 |
| `kg_efg_pct` | Effective FG% = (FGM + 0.5×FGM3) / FGA | 2003–2026 |
| `kg_off_reb_pg` | Offensive rebounds per game | 2003–2026 |
| `kg_def_reb_pg` | Defensive rebounds per game | 2003–2026 |
| `kg_ast_pg` | Assists per game | 2003–2026 |
| `kg_to_pg` | Turnovers per game | 2003–2026 |
| `kg_stl_pg` | Steals per game | 2003–2026 |
| `kg_blk_pg` | Blocks per game | 2003–2026 |
| `kg_scoring_margin` | Mean (own score − opp score) across all games | 2003–2026 |
| `kg_recent_margin` | Scoring margin in last 10 games (DayNum 120–132) | 2003–2026 |
| `kg_road_win_pct` | Road win % (WLoc='A') | 2003–2026 |
| `kg_opp_fg_pct` | Opponent FG% allowed (defensive efficiency) | 2003–2026 |
| `kg_opp_efg_pct` | Opponent effective FG% allowed | 2003–2026 |
| `kg_ast_to_ratio` | Assist-to-turnover ratio | 2003–2026 |

#### Massey Ordinal Rankings (`MASSEY_FEATURES`, RankingDayNum=133)

Pre-tournament rankings from the Massey database — supplements team sheets for years/teams without team sheet data.

| Feature | System | Available |
|---|---|---|
| `massey_POM` | Pomeroy | 2003–2026 |
| `massey_SAG` | Sagarin | 2003–2023 |
| `massey_RPI` | RPI | 2003–2026 |
| `massey_MOR` | Morin | 2003–2026 |
| `massey_WLK` | Wolker | varies |
| `massey_DOL` | Dolphin | varies |
| `massey_COL` | Colley | varies |

---

### TIER 2-ONLY FEATURES (Final Four prediction only — include tournament games)

These use post-Selection Sunday data. Safe ONLY when you're already conditioning on teams that made the Final Four (since those games already happened by the time you predict the championship).

#### Tournament Path Features (`TOURNEY_PATH_FEATURES`, DayNum 134–146)

| Feature | Description | Available |
|---|---|---|
| `kg_tourney_avg_margin` | Mean scoring margin across R1–E8 games | 2003–2025 (2026 not yet in Kaggle) |
| `kg_tourney_worst_margin` | Minimum margin — closest game (stress-test) | 2003–2025 |
| `kg_tourney_fg_pct` | FG% during tournament games through E8 | 2003–2025 |
| `kg_tourney_opp_fg_pct` | Defensive FG% allowed in tournament | 2003–2025 |
| `kg_rounds_survived` | Games won (1–4, where 4 = won E8 = made FF) | 2003–2025 |

**Interpretation**: `kg_tourney_avg_margin` is the single strongest new signal from our last importance run (top 5 by MDI+MDA). Teams that dominate their path to the Final Four win the championship at a higher rate than teams that barely survive.

#### Team-Stats PCA Components (`TEAM_STATS_PC_FEATURES`)

Derived from 11 retained team-stats features after cross-source reconciliation dropped 8 redundant ones. 7 components explain 93.1% of variance. These are full-season aggregates (including tournament through E8).

| Component | Top loadings | Basketball interpretation |
|---|---|---|
| `ts_pc1` | rebound_margin, three_pct_def, fouls_pg | Defensive toughness / physical play |
| `ts_pc2` | fg_pct, fouls_pg, off_reb_pg | Offensive efficiency |
| `ts_pc3` | fg_pct, fouls_pg, rebound_margin | Combined efficiency |
| `ts_pc4` | three_pct_def, turnover_margin, fg_pct | Ball-control defense |
| `ts_pc5` | turnover_margin, three_pct_def, to_forced_pg | Pressure defense |
| `ts_pc6` | turnover_margin, to_forced_pg, def_reb_pg | Defensive rebounding |
| `ts_pc7` | off_reb_pg, def_reb_pg, rebound_margin | Rebounding balance |

**Raw ts_* features kept after reconciliation** (r < 0.95 with Kaggle equivalent):
- `ts_fg_pct` (r=0.50 vs `kg_fg_pct` — diverge because ts_ includes tournament games)
- `ts_off_reb_pg` (r=0.90)
- `ts_def_reb_pg` (r=0.90)
- Plus all features with no Kaggle equivalent: `ts_rebound_margin`, `ts_three_pct_def`, `ts_turnover_margin`, `ts_fouls_pg`, `ts_reb_pg`, `ts_bench_pts_pg`, `ts_fastbreak_pts_pg`, `ts_to_forced_pg`

---

### MARKET FEATURES (2025–2026 only, Kalshi prediction markets)

| Feature | Description | Notes |
|---|---|---|
| `mkt_vwap` | Volume-weighted average yes_price | Best estimate of market's implied win probability |
| `mkt_last_price` | Last yes_price before championship | Current market price |
| `mkt_ofi` | Order flow imbalance = (yes_vol − no_vol) / total_vol | Directional pressure — positive = buyers dominating |
| `mkt_momentum` | (last 24h VWAP − prior 7d VWAP) / prior 7d VWAP | Price trend direction |
| `mkt_trade_count` | Total trades | Liquidity proxy — high count = more confident market |
| `mkt_volatility` | Std(yes_price) over last 7 days | Price uncertainty |
| `mkt_price_range` | Max − min yes_price over last 7 days | Full range of market belief |

**2026 Final Four snapshot (March 30):**
| Team | VWAP | Implied % |
|---|---|---|
| Arizona | 0.2186 | 37.8% |
| Michigan | 0.2169 | 37.5% |
| Illinois | 0.0843 | 14.6% |
| Connecticut | 0.0579 | 10.0% |

**Do NOT run MDI/MDA on market features** — only 2 years of Final Four data (8 rows). Use SFI only, or use `mkt_vwap` as the primary model in the meta-labeling framework.

---

## 7. Feature Selection Pipeline

```
Raw features (~78 total after loading)
        │
        ▼
Cross-source reconciliation
  Compare kg_* vs ts_* pairs by Pearson r
  Drop ts_* if r > 0.95 (captured by Kaggle)
  Keep both if r < 0.95 (genuinely different signal)
  → drops 8 redundant ts_* columns
        │
        ▼
PCA on remaining ts_* features
  Standardize within year (removes era effects)
  Retain components explaining 90% variance
  → 11 raw ts_* → 7 PCA components
        │
        ▼
SFI on Tier 1 (~1,400 seeded team rows)
  Train on one feature at a time, leave-one-year-out CV
  Label: made_ff (binary)
  Features above log-loss baseline → SFI survivors
  → collinearity-immune standalone importance filter
        │
        ▼
ONC clustering on correlation matrix
  Groups correlated features into clusters
  CFI shuffles clusters together (immune to substitution bias)
  → 3 clusters found (rankings / performance / efficiency)
        │
        ▼
MDI + MDA on Final Four frame (Tier 2, ~78 rows)
  MDI: in-sample, fast, captures interactions
  MDA: out-of-sample via purged year-CV, marginal importance
  Clustered MDA: shuffle clusters simultaneously
  → final importance ranking
```

---

## 8. What the Output Numbers Mean

| Method | Measures | Bias | When to trust |
|---|---|---|---|
| **SFI** | Standalone predictive power, no other features present | None (collinearity-free) | First filter — if a feature can't do anything alone, it's noise |
| **MDI** | Average information gain contribution across all trees | High-cardinality features get more splits | In-sample: identifies features the model actually uses |
| **MDA** | Accuracy drop when this feature is shuffled (others intact) | Correlated features understate importance | OOS: identifies uniquely important features |
| **CFI** | MDA but shuffle the entire correlated cluster | None (cluster handles substitution) | Most reliable for correlated groups |

A feature that ranks well across ALL four methods is unambiguously important.
A feature high in SFI but low in MDA = has signal, but is redundant with something else in the set.
A feature low in SFI = has no standalone signal → drop.
