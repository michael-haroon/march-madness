update readme: archived data are data that have been cleaned and added to kaggle data set; wanted improvements include a modulated name normalizer

we can generalize this repo to all basketball associattions
---
# Potential Features for March Madness as tournament comes to a close

Labels determine the task that the algorithm is going to learn. Examples:
- triple barrier labels (binary classification)
- meta labels (binary classification)

We are analyzing binary futures contracts that expire soon. So we have two labeling options:
1. Label winners vs losers (ie. only one winner for the whoel tourney, so rest are labeled losers)

## Microstructure
- [ ] de Prado features
- [ ] what else...
- [ ] volatility
- [ ] cross market behaviors (not to be confused with cross exchange behaviors)
- [ ] cross exchange behaviors
- [ ] chance of Yes compared to others

## Team Stats 
- WARNING: the following have overlap so how might use it in ML feature discovery?
- Note that my team stats might be more recent than kaggles
    - [ ] A KEY feature is scoring margin in recent games and their difficulty!
    - [ ] average stats per game from `/{year]}team-stats`
        - the issue with this data is that, although it is theoretically unbiased, it is not a good fit for some distributions. so we have to find the distribution of games. Maybe the ML can do that when finding features.
    - [x] total wins so far. In Team sheets
    - [x] wins-losses ratio. Team sheets
    - [ ] win streak 
    - [ ] recent point scored
    - [ ] average points per game
    - [ ] average age of team
    - [ ] team movement (needs recordings of games, can skip this one)
    - [ ] performance/winrate against a particular team (pairwaise comparison)
    - [ ] player chemistry (how do we measure this? ideally with video performance and interaction, online interaction, total assists per game in a team)
    - [ ] coach
    - [ ] historical ranking
    - [ ] number of all stars
    - [ ] which team holds or made the most records that season
    - [-] Historical NET, KPI, SOR, BPI, POM, SAG, Strngth of schedule (SOS). Some are missing

## Player Stats
- [ ] highest scoring player
- [ ] tallest player
- [ ] average free throw % of team
- [ ] average fp% of team members
- [ ] distribution of points or rebounds or steals or turnovers

## Game
- [ ] climate and weather
- [-] home vs away. calculated in most RPIs. can be found in team sheets if need more
- [-] specific stadium/location. We have it for finals, but not for every game played.
- [ ] morning, night, or evening performance
- [x] final four contestestants. Found in `yearly_champions.csv`. 2026 includes Illinois, UConn, Michigan, and Arizona

---

---

Remember that historical data is just one random walk--you need more.Sample historical tournaments for backtesting and simulations. Make synthetic data for absolute "what the fuck" moments to see how your model reacts; it does not need historical data--it's just risk management

Get historical data for each year, and the features will target who won that year (row wise, not autoregressive; but maybe the only autoregressive features are team/player data from the current and the previous season). For example, find all records, recent stats, etc. for each team in a YEAR up until before the championship game, and let them be features to determine who wins. See if they determine target (who wins? or at least who ends up top 3?)

## Data
Note: Data still needs normalization! We need a reference point since some data different abbreviations for the same teams
- `yearlys/yearly_champions.csv`
    Columns: `Year,Champion,Score,Runner-Up,Third Place,Fourth Place,Overtime,OT_Count,Champion_Vacated`
    - only final games have overtime
    - contains all final vour contestants except for 2026.

- `team_sheets/{year}_Team_Sheets_[Final|Selection].csv`
    Columns (note that some csvs don't have them!): `Avg_NET_Losses,Avg_NET_Wins,Avg_RPI_Loss,Avg_RPI_Win,BPI,Conference_Record,KPI,NET,NET_NonConf_SOS,NET_Rank,NET_SOS,NonConf_Record,Opp_SOS_D1,Opp_SOS_NonConf,Overall_Record,PM_T-Rank,POM,RB_WAB,RPI_Rank_D1,RPI_Rank_NonConf,Record,Road_Record,SAG,SOR,SOS_D1,SOS_NonConf,Team`
    - NET, prediction, and resume measurements
    - There is slight lookahead bias here for files that end in Final. I say slight because the Team Sheets don't change much from Final Four to Championship.
    - Also note that names are inconsistent. For example, one csv has "Loyola Chicago" and the other has "Loyola (Ill.)" or even just "Loyola (Ill)"
        - there is also some inconcsisteny for hte ones still in test. the naems are duped or missing a part of the ful name

- `yearly/yearly_award_winners.csv`
    Columns: `Year,BT_Player,BT_Team,AP_Player,AP_Team,USBWA_Player,USBWA_Team,Wooden_Player,Wooden_Team,NABC_Player,NABC_Team,Naismith_Player,Naismith_Team`
    - key feature of interest is USBWA_Team since it is awarded before the finals. The rest pose lookahead bias and don't have an awardee for 2026. Anyway, the winner of that award is from a team that did not make the final four so it might not be a good feature. Let ML figure that out

- `yearly/yearly_sporting_news_player.csv`
    Columns: `Year,Player,School`
    - consists of award winners and their team name for Sporting News Player of the Year. It is awarded before the finals

- `yearly/yearly_champion_location.csv`
    Columns: `Year,City,State,Arena`
    - might be redundant to Kaggle

- `{year}-team-stats/*.csv`
    Columns: (some dont have what others have!) `3FG,3FG%,3FGA,3PG,APG,AST,Avg,BKPG,BLKS,Bench,DQ,DRebs,FB pts,FG%,FGA,FGM,FT,FT%,FTA,Fouls,G,GM,L,OPP FG,OPP FG%,OPP FGA,OPP PPG,OPP PTS,OPP REB,OPP RPG,ORebs,Opp 3FG,Opp 3FGA,Opp TO,PFPG,PPG,PTS,Pct,REB,REB MAR,RPG,Rank,Ratio,SCR MAR,ST,STPG,TO,TOPG,Team,W,W-L`
    - might be redundant to kaggle, but we cover recent years that kaggle does not!
    - Note that the year from 2022 and before have redundancy across sheets despite some distinction (eg. assists vs assit turnove ratio have all the same cols exccept assists has AVG per game and the ratio one has a ratio). Use 2023+ for reference of how clean data should be
    - total rebounds per game has data from total rebounds and we can jsut sum off and def reboudns to get total rebounds. 
        - but def and off rebound per game files are distinct
    - Then there is fewest fouls in <2023 and just fouls per game in alter fiels
    - You only need Free Throw Percentage file since the other two free throw files are absolutely redundant
    - Fewest fouls is also not needed bc personal fouls per game covers it
    - same with fewest turnovers: turnover margine covers it
    - So skip these files:
        - 3-pt field goal attempts
        - Assist Turnover ratio (just get total assists from assists per game and TO from any turnover file and calculate the ratio assts/turnovers)
        - Fewest fouls
        - fewest turnovers
        - Final Opp Points (can just get Opp 3FG and OPP FG and OPP FT and OPP PTS from other files)
        - Free THrow Attempts
        - Free THrows Made
        - Scoring Defense
        - Scoring Offense
        - total 3 point FGM
        - Three point field goals per game (just take 3FG/GM)
        - Total Assists
        - Total Blocks
        - Total Rebounds
        - Total Rebounds per Game
        - Total Steals
        - Turn over per game
        - forced turnovers (take turnovers per game and turnovers forces and do TpG-TF or divide somehow depending on how to weight it for ML; these two files have per game data so don't get rid of them... or we can calculate them ourselves via TO/num games and the likes)
            NOTE: NO forced turnovers for the years 2015 and before
            NOTE: Turn over margin includes opp TO and TO, so TO/G is just doing the math right there-- same with forced turn. Now only downloading Turnover Margin from 2015 and before
        - won-lost percentage (it's in every other file in that year just take W-L column)
    - IN FACT: we can get ALL TOTAL and MARGIN FILES and just calculate per game stats from there. We will also need the Defense files (not defensive) and the Percetange files. Try this for 13
        - no, it makes us do an extra step since we still download but then have to calculate ourselves
    - 2013 and before has less data in general. so 2015 loses the forced turnover file. 2014 i use turn over margin, then 13 just has less files, then 08 has even less
    - If a team is not in three point field goal percentage file, then they made less that 5 shts per game

- `market_data_store/year={2025|2026}/ticker={ticker}/*.parquet`
    - Columns: `trade_id (string/UUID), ticker (string), yes_price_dollars (float64), no_price_dollars (float64), count_fp (int64), created_time (datetime64[ns]), taker_side (string)`
    - Flattened market data from Kalshi for only the past two years

- `kaggle/`
    - READ THE kaggle/DATA.md!


    
### Warnings on Data
Keep an eye out for lookahead bias. We don't want that.

# Metrics
Firstly, we identify what we want to find. We want to find the probability of winning for the Final Four contestents of each year. Ideally, the team with the highest probabilty of winning from our model is the one who truly wins, and the rest are truly losers ranked accordingly to their order of probability of winning. So how do we measure that given out features? Also, can we measure confidence? How? The number of times it got right?



# What if the game is rigged or a champ vacates?
- How do we manage this risk?

# Current project structure
```
march-madness
├── 2003-team-stats
├── 2004-team-stats
├── 2005-team-stats
├── 2006-team-stats
├── 2007-team-stats
├── 2008-team-stats
├── 2009-team-stats
├── 2010-team-stats
├── 2011-team-stats
├── 2012-team-stats
├── 2013-team-stats
├── 2014-team-stats
├── 2015-team-stats
├── 2016-team-stats
├── 2017-team-stats
├── 2018-team-stats
├── 2019-team-stats
├── 2021-team-stats
├── 2022-team-stats
├── 2023-team-stats
├── 2024-team-stats
├── 2025-team-stats
├── 2026-team-stats
├── kaggle
├── market_data_store
│   ├── kalshi_name_mapping.csv
│   ├── historical-endpoint
│   │   └── year=2025
│   │       └── ticker=KXMARMAD-25-* 
│   └── markets-endpoint
│       ├── year=2025
│       │   └── ticker=KXMARMAD-25-*
│       └── year=2026
│           └── ticker=KXMARMAD-26-*
├── team_sheets
└── yearlys
├── feature_pipeline/         ← all ML code lives here
│   ├── run_v2.py             ← ACTIVE entry point (game-level model)
│   ├── run.py                ← legacy Final Four pipeline (not actively used)
│   ├── game_model.py         ← build_team_season_features, build_game_pairs, train/predict
│   ├── data_loader.py        ← load_all(), team sheets + Kaggle + market loading
│   ├── feature_engineering.py← build_features(), feature group constants, PCA
│   ├── feature_importance.py ← MDI, MDA, SFI, CFI, PurgedYearKFold
│   ├── model.py              ← Bradley-Terry pairwise model (used by legacy run.py)
│   ├── market_features.py    ← Kalshi microstructure VWAP/OFI/momentum features
│   ├── name_resolver.py      ← team name ↔ Kaggle TeamID bidirectional lookup
│   ├── season_utils.py       ← calendar date → (Season, DayNum) helpers
│   └── config.py             ← feature lists, LGBM params, bracket config
├── scripts/                  ← scrapers and data integrators (not part of ML pipeline)
└── README.md
```