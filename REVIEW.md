# Data Review — March Madness ML Project

Last updated: 2026-03-31

---

## What You Have vs. What Kaggle Has

### Ranking metrics — gaps or completely absent in Kaggle

| Column | Your years | Kaggle (MMasseyOrdinals) coverage | Notes |
|---|---|---|---|
| **SOR** | 2005–2026 | Not present at all | 100% exclusive to your team sheets |
| **BPI** | 2021–2026 | 2009–2013 only | Your 2014–2026 is exclusive |
| **KPI** | 2005–2026 | 2016–2019, 2024–2025 (missing 2021–2023) | Your 2005–2015 and 2021–2023 are exclusive |
| **NET** | 2005–2026 | 2019–2025 only | Your 2026 is exclusive; 2005–2018 is exclusive but flagged below |
| **SAG** | 2005–2026 | 2003–2023 only | Your 2024–2026 are exclusive |
| **POM** | 2005–2026 | 2003–2026 | Fully covered by Kaggle — use Kaggle for full 23-year history |
| **RPI** | 2005–2026 | 2003–2026 | Fully covered by Kaggle |

**Critical data quality flag — NET 2005–2018**: The NET ranking system was not introduced until the 2018–2019 season. Your team sheets have a `NET` column going back to 2005. Those values are likely retroactively computed estimates or mislabeled. Kaggle correctly only includes NET from 2019. Treat 2005–2018 NET values with suspicion; do not use them as equivalent to the official committee metric.

---

### Strength-of-schedule breakdowns — not in Kaggle at all

Kaggle has ordinal rankings but no sub-split SOS data. These are exclusively yours:

**2005–2019:**
- `SOS_D1`, `SOS_NonConf` — overall and non-conference SOS
- `Opp_SOS_D1`, `Opp_SOS_NonConf` — opponents' SOS
- `Avg_RPI_Win`, `Avg_RPI_Loss` — average RPI of wins and losses (quality-adjusted record)
- `RPI_Rank_D1`, `RPI_Rank_NonConf`

**2021–2026:**
- `NET_SOS`, `NET_NonConf_SOS` — NET-based SOS splits
- `Avg_NET_Wins`, `Avg_NET_Losses` — average NET rank of wins and losses
- `RPI_Rank_D1`, `RPI_Rank_NonConf`

These are high-value features: `NET_NonConf_SOS` and `Avg_NET_Wins` directly reflect how teams performed against quality opponents, which is what the committee evaluates.

---

### Record breakdowns — not in Kaggle at all

Available in your team sheets **2021–2026**: `Conference_Record`, `Overall_Record`, `NonConf_Record`, `Road_Record`

Kaggle's game logs let you compute these from scratch, but you have them pre-computed in committee format.

---

### New-era committee metrics (2025–2026 only)

- **`RB_WAB`** (Résumé-Based Wins Above Bubble) — not anywhere in Kaggle
- **`PM_T-Rank`** — not in Kaggle

Only 2 years of data so very limited standalone signal, but they are the metrics the committee currently emphasizes. Include them with missing-value flags.

---

### Kalshi market data — entirely absent from Kaggle

- **696,167 trade records** across 2025 and 2026 tournaments
- **~139 tickers** (historical endpoint) + **~165 tickers** (markets endpoint), covering all ~68 tournament teams — not just Final Four
- Schema: `trade_id, yes_price_dollars, no_price_dollars, count_fp, taker_side, created_time`
- Date range: Jan 2025 – Mar 31, 2026 (live through current tournament)
- `taker_side` (yes/no) enables order flow imbalance features — de Prado's microstructure signal

This is not a "one price per team" lookup. It is tick-by-tick trade history from which you can derive: VWAP, price momentum, order flow imbalance, volatility, and contract liquidity per team.

---

### Yearlys

| File | Exclusive to you? | Notes |
|---|---|---|
| `yearly_champions.csv` | Derivable from Kaggle | Has Third/Fourth Place pre-labeled — use as your label source |
| `yearly_award_winners.csv` — USBWA_Team | **Exclusive** | Awarded before tournament, safe to use as feature |
| `yearly_award_winners.csv` — Naismith, Wooden, AP, BT | **Exclusive but drop** | Awarded after tournament, lookahead bias |
| `yearly_sporting_news_player.csv` | **Exclusive** | Pre-tournament, safe to use |
| `yearly_championship_location.csv` | Overlaps with Kaggle `MGameCities.csv` | Redundant |

---

## Lookahead Audit

| Source | Safe? | Note |
|---|---|---|
| `*_Selection.csv` team sheets | Safe | Snapshot at selection, before tournament |
| `*_Final.csv` team sheets | Slight bias | Includes conf tournament games after selection. Values don't change much but technically do |
| Regular season game stats (DayNum ≤ 132) | Safe | All pre-tournament |
| USBWA award | Safe | Awarded before tournament |
| Sporting News Player of Year | Safe | Pre-tournament |
| Naismith, Wooden, AP, BT awards | **Drop** | Awarded after tournament |
| Kalshi prices | Safe if pre-tip snapshot | Use last price before championship tip-off |
| `*_Final.csv` + Kaggle tournament results mixed | **Drop** | Do not compute features using tournament game outcomes as inputs |

---

## Feature Matrix Priority

1. **Use yours, Kaggle cannot substitute**: SOR, all SOS splits, NET_SOS/NonConf, Avg_NET_Wins/Losses, Road_Record, RB_WAB, USBWA flag, Kalshi microstructure
2. **Use yours to fill Kaggle's gaps**: BPI (2014–2026), KPI (2021–2023), SAG (2024–2026), NET (2026)
3. **Prefer Kaggle for full history**: POM, RPI, SAG (2003–2023)
4. **Verify before use**: NET column in 2005–2018 team sheets — likely not the official NET ranking

---

## Sample Size Reality Check

- Final Four only: ~21 years × 4 teams = **84 rows** (no 2020, tournament not held)
- If expanding to all 68 tournament teams: ~21 years × 68 = ~1,400 rows (but team sheet data only covers Final Four)
- Kaggle regular season game data goes back to 1985 for compact results, 2003 for detailed box scores

With 84 rows, MDI and MDA will overfit. Run SFI first as a filter. Use leave-one-year-out CV (21 folds, ~4 training samples per fold). Treat the feature selection results as directional, not definitive.

The Kalshi data expands your universe to all 68 teams for 2025–2026, which meaningfully helps market feature development.

---

## 2026 Live Prediction Target

The 2026 Final Four is: **Illinois, UConn, Michigan, Arizona** (confirmed).
The championship game has not been played yet as of 2026-04-05.
Market data runs through the current date — Kalshi prices up to tip-off are available as features.
Treat 2026 as test data, not training data.
