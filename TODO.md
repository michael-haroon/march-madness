# TODO

## In Progress
- 2026 tournament is live — update predictions after each game result

## Next Up

### Phase 1B: Tournament Path Features
Add cumulative path stats (avg margin, opponent seed quality, OT games) as features for later-round predictions. The game model currently treats all tournament rounds identically; path features give it "current form" signal.
- Implement `compute_path_features(team_id, season, games_so_far)` in `game_model.py`
- Modify `build_game_pairs(include_path=True)` — add `diff_path_*` columns
- Update `predict_final_four` to pass real E8 path data for the 2026 bracket
- Ablation: compare CV AUC with vs without path features per round

### Data Gaps to Fill
- **BPI**: missing 2021–2025 (have 2021–2026 from team sheets; Kaggle only has 2009–2013). Fill from ESPN.
- **SOR**: not in Kaggle at all — only have it from team sheets for Final Four teams. Expand to all 68 teams via ESPN or Warren Nolan.
- **KPI**: missing 2021–2023 in Kaggle. Fill from Warren Nolan.
- **Recent tournament game stats for 2026**: need to add E8 results once Kaggle updates (or ingest directly).

### Features Not Yet in Pipeline
- Away game performance weighted stats (relevant since Finals are neutral-site)
- Coach tenure / historical Final Four experience (from `MTeamCoaches.csv`)

## Done
- [x] Game-level pairwise model (Phase 1A) — `feature_pipeline/run_v2.py`
- [x] Monte Carlo bracket simulation for 2026 Final Four
- [x] Kalshi market blend (VWAP, OFI, momentum, trade count)
- [x] Name resolver — 0% miss rate on all 279 tournament teams (2003–2026)
- [x] Cross-source reconciliation (kg_ vs ts_ features)
- [x] Binary missingness indicators for pre-NET / pre-market eras
- [x] PurgedYearKFold leave-one-year-out CV
- [x] De Prado feature importance suite (MDI, MDA, SFI, CFI)
- [x] UConn name mapping fix (Kalshi "UConn" → "Connecticut")
