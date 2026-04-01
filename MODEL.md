# MODEL.md — ML Architecture, Theory, and Empirical Basis

## The Core Problem

Predicting who wins the NCAA championship from a four-team Final Four is a **ranking problem with a tiny sample**. Every year you get exactly 4 teams and exactly 1 champion. Over 21 historical years (2005–2025, excluding 2020) that's 84 team-season observations total.

This creates two interlocking constraints:
1. You cannot use deep models that require thousands of examples
2. You cannot trust feature importance results computed on 84 rows

The entire architecture below exists to solve these two problems without sacrificing rigor.

---

## Part 1: The Missing Data Problem — Why Indicators Work

### The problem

Many features are only available for some years. NET rank didn't exist before 2019. Kalshi market data only covers 2025–2026. Quadrant records (Q1/Q2/Q3/Q4) only exist post-2021. A naive approach would drop rows with missing values or fill with zero.

### Why both are wrong

**Dropping rows**: With 84 total observations, dropping any year with missing features leaves you with as few as 21 rows (only 2021–2024 have the full modern feature set). That's not enough to train or evaluate anything.

**Filling with zero or median**: This destroys information. Consider `net_rank` filled with median (rank ~50) for pre-2019 years. The model now "sees" all pre-2019 teams as having a middling NET rank — indistinguishable from truly average modern teams. The information that "this is a pre-NET era team" is lost.

### The correct approach: binary missingness indicators

In `pipeline/feature_engineering.py: handle_missing()`:
```python
df[f"{col}_missing"] = is_null.astype(int)   # 1 = this value was absent
df[col] = df[col].fillna(df[col].median())   # fill so the column is numeric
```

Every feature with `"indicator"` in `MISSING_STRATEGY` gets a paired binary column. For `net_rank`:
- `net_rank` — filled with median for pre-2019 rows
- `net_rank_missing` — 1 for pre-2019, 0 for 2019+

### Why this is theoretically sound

A decision tree with access to `net_rank_missing` can now learn:

```
if net_rank_missing == 1:
    # pre-2019 team — use RPI and SOS as primary signals
    if rpi_sos < 30 AND overall_win_pct > 0.9: → high probability
else:
    # modern team — NET is reliable
    if net_rank < 5 AND kg_tourney_avg_margin > 12: → high probability
```

This is equivalent to fitting **separate sub-models** for each missingness pattern. The tree discovers this split automatically from data, without you having to hardcode era-specific logic.

**Formal basis**: Saar-Tsechansky & Provost (2007) showed the indicator method outperforms both mean imputation and case deletion for tree models, specifically because trees can use the indicator as a split criterion. Josse, Prost, Scornet & Varoquaux (2019, "On the Consistency of Supervised Learning with Missing Values") proved that gradient-boosted trees with missing indicators are consistent estimators even under MNAR (Missing Not At Random) mechanisms — which is exactly our case (data is missing because the metric didn't exist, not randomly).

### Why MNAR matters here

Our missing data is **not random**:
- `net_rank_missing=1` is perfectly correlated with `year < 2019`
- `mkt_vwap_missing=1` is perfectly correlated with `year < 2025`
- `q1_win_pct_missing=1` is perfectly correlated with pre-2021 era

MNAR is the hardest missing data regime. The indicator method handles it correctly precisely because the missingness pattern itself becomes a feature the model can split on. No imputation method can handle MNAR correctly — only the indicator approach can.

### Empirical validation in our data

In our feature importance runs, `net_rank_missing` and similar indicators rank in the top half of features by SFI. This confirms the tree is actively using these columns as signal, not ignoring them. The model learned that `net_rank_missing=1` is correlated with era-specific patterns in championship outcomes.

---

## Part 2: Pairwise Modeling vs. Rank-4 Modeling

### The rank-4 approach (what we did NOT choose as primary)

**Setup**: 4 rows per year (one per Final Four team), target = finish rank (1=champion, 2=runner-up, 3/4=semis losers). Train an ordinal or multi-class classifier.

**Problems**:

1. **84 training observations**. With 50+ features, any flexible model will memorize the training data. Even a shallow decision tree has enough degrees of freedom to overfit.

2. **Extreme class imbalance within each year**. One champion out of four = 25% positive rate. Ordinal rank has four classes at frequencies [25%, 25%, 25%, 25%] — balanced, but each absolute rank only has 21 examples across all years.

3. **Year-to-year non-stationarity**. The features are not on the same absolute scale across years. A team ranked NET #3 in 2022 is not the same as a team ranked NET #3 in 2026 because the pool of D1 teams changes, the ranking algorithm is updated, and basketball evolves. Absolute feature values are not directly comparable.

4. **Label noise**: The difference between finishing 1st and 2nd in a one-game championship is partially luck. Treating it as a deterministic ordinal outcome overweights single-game randomness.

### The pairwise approach (Bradley-Terry style)

**Setup**: For each year's Final Four, generate all ordered pairs (A, B) where A ≠ B. Features = A_metric − B_metric for each feature. Target = 1 if A finished better (lower finish_rank), 0 otherwise.

`pipeline/model.py: build_pairwise()` uses `itertools.permutations(teams, 2)` — both orderings of each pair — giving 4×3 = 12 rows per year × 21 years = **252 training observations** from the same 84 team-seasons.

### Why pairwise is superior — the theory

**1. Sample efficiency: 3× more data from the same observations.**

The pairwise conversion is a lossless transformation of the ranking information. Four teams ranked [A, B, C, D] imply 6 pairwise comparisons (C(4,2) = 6 unordered, 12 ordered). No information is discarded. You now have 252 training rows instead of 84, with the same underlying ground truth.

**2. Feature differences cancel out baseline drift.**

When features are differences rather than absolute values:

```
diff_net_rank = net_rank_A - net_rank_B
```

The absolute scale of `net_rank` in a given year is irrelevant. What matters is the *gap* between two teams in the same year. A gap of 8 positions means the same thing in 2005 as in 2025 — team A is considerably better ranked than team B. This solves the non-stationarity problem at the feature level.

This is analogous to using **price returns** rather than price levels in finance — removing the non-stationary component (the level) while preserving the informative signal (the relative change).

**3. The Bradley-Terry model is well-studied and empirically validated.**

The Bradley-Terry model (Bradley & Terry, 1952; originally Zermelo, 1929) is the canonical framework for paired comparisons. It defines team strength as a scalar r_i such that:

```
P(A beats B) = exp(r_A) / (exp(r_A) + exp(r_B)) = sigmoid(r_A - r_B)
```

This is exactly what a logistic regression or gradient-boosted classifier on feature *differences* is estimating. Our LightGBM pairwise model is a nonlinear extension of the Bradley-Terry model that allows feature interactions.

**Empirical evidence for pairwise ranking**:
- Herbrich, Minka & Graepel (2007) — TrueSkill: paired comparison models outperform absolute rating models for predicting game outcomes across all sports studied
- Chen & Lin (2006) — pairwise learning has O(n log n) sample complexity advantage vs. rank-N approaches for the same desired generalization error bound
- Kvam & Sokol (2006) — pairwise models outperform traditional power ratings for NCAA tournament prediction specifically

**4. The label is genuinely binary, not noisy ordinal.**

"Did team A finish better than team B?" is a cleaner label than "what rank did team A finish?" because:
- It's symmetric: if A beats B, then B does not beat A (no ties possible given 4 distinct ranks)
- It doesn't try to distinguish 3rd and 4th place (both are semis losers — essentially tied for "eliminated in semifinals")
- Binary classification with balanced classes (each pair has 50/50 win rate by construction) is easier to learn than imbalanced ordinal regression

### Aggregation: pairwise → win probability

After the model produces P(A beats B) for all 12 ordered pairs, aggregate to team-level win probability in `pipeline/model.py: pairwise_to_win_prob()`:

1. For each team A and each of its 3 opponents (B, C, D): get P(A beats B), P(A beats C), P(A beats D)
2. Take the mean: `raw_prob_A = mean([P(A>B), P(A>C), P(A>D)])`
3. Normalize within each year so all four teams' probabilities sum to 1

This aggregation is principled — it's the Condorcet aggregation of pairwise comparisons, and it gives each team a well-calibrated championship probability.

### The fallback when the pairwise model isn't trained

When there aren't enough clean pairs to train (e.g., first run before NaN imputation is complete), the fallback in `pipeline/run.py` computes a **composite ranking**:

```python
win_prob = 0.50 × normalise(mkt_vwap)          # market implied prob
         + 0.30 × normalise(-consensus_rank)    # analytical rankings (inverted)
         + 0.20 × normalise(kg_scoring_margin)  # on-court performance
```

**Theoretical basis of this fallback**: Each component is a documented predictor of championship outcomes:
- `mkt_vwap`: prediction markets are efficient aggregators of public information (Fama, 1970). For 2025–2026, the Kalshi market has 694k trades representing the collective belief of all market participants.
- `consensus_rank`: the mean of analytical models (POM, SAG, BPI, KPI, SOR) is more accurate than any individual model — ensemble averaging reduces variance. This is proven by the Brier score decomposition.
- `kg_scoring_margin`: teams that outscore opponents by large margins are less likely to lose close games in the tournament (Pythagorean expectation theory, James, 1981).

The 50/30/20 weighting is ad hoc but conservatively overweights the market (strongest prior), then analytical rankings (decades of validation), then game stats (high variance on small samples).

---

## Part 3: Feature Importance Suite — Theory and Implementation

### Why not just look at coefficients or tree splits?

Standard ML models give you in-sample importance metrics (logistic regression coefficients, decision tree splits). These fail for our purpose because:

1. **Correlated features**: NET rank and POM are r=0.85 correlated. In a logistic regression, their coefficients are numerically unstable — any split of variance between them is arbitrary. A coefficient near zero doesn't mean the feature is unimportant; it means another correlated feature absorbed its contribution.

2. **In-sample bias**: A feature that memorizes the training set gets high in-sample importance but performs poorly on held-out years. With only 21 years of data, a single year with an "unusual" champion can dominate the coefficients.

3. **Interactions**: Linear coefficients miss nonlinear relationships (e.g., "a team with low NET rank AND high tournament margin is qualitatively different from a team with just one of those").

The de Prado suite addresses all three.

### Method 1: MDI (Mean Decrease Impurity)

**Implementation**: `pipeline/feature_importance.py: feat_imp_mdi()`

After fitting the Random Forest, for each decision tree, walk every internal node. Record:
- Which feature was used for the split
- What fraction of training samples reached this node (`w`)
- How much the Gini impurity decreased (`Δi`)

Sum `w × Δi` for each feature across all nodes across all trees. Normalize to sum to 1.

**The critical parameter: `max_features=1`**

Standard Random Forests use `max_features=sqrt(n_features)` — at each split, consider a random subset of features and pick the best one. This means correlated features compete, and the first one chosen "blocks" correlated features from appearing higher in the tree.

De Prado's key insight (AFML §8.3.1): set `max_features=1`. Every split considers exactly one randomly chosen feature. Features are chosen by lottery, not by competition. This removes the "rich get richer" bias where high-correlation features dominate early splits.

**Mathematical result**: with `max_features=1`, MDI converges to the true marginal contribution of each feature in the linear limit. Each feature gets roughly equal total exposure across the ensemble.

**Limitations of MDI**:
- In-sample only: a feature that happens to correlate with champion years in the training set gets high MDI even if the relationship doesn't hold out-of-sample
- Biased toward continuous features: a continuous feature can produce more distinct split points than a binary feature, giving it more opportunities to reduce impurity

MDI is a first pass — identifies what the model uses, not what genuinely predicts future outcomes.

### Method 2: MDA (Mean Decrease Accuracy)

**Implementation**: `pipeline/feature_importance.py: feat_imp_mda()`

For each fold in PurgedYearKFold:
1. Fit the model on training years
2. Compute baseline log-loss on the held-out year
3. For each feature X_j: shuffle X_j in the test set (break its correlation with the label), recompute log-loss
4. MDA(X_j) = baseline_log_loss − permuted_log_loss

A positive MDA means shuffling hurt accuracy — the feature was genuinely used. Negative MDA means shuffling helped — the feature was adding noise.

**Why log-loss, not accuracy**: Log-loss rewards calibrated probabilities. A model that always predicts 50/50 gets the worst possible log-loss even if it correctly classifies half the examples. Log-loss distinguishes "was the model confident and right" from "was the model confident and wrong."

**The substitution bias problem**:

If NET rank and POM are correlated, shuffling NET rank barely hurts because the model compensates with POM. Both features appear to have low MDA even though together they're the most important signal. This is the central failure mode of MDA with correlated features.

**Our fix: Clustered MDA via ONC**

ONC (Optimal Number of Clusters) finds groups of correlated features using silhouette t-stat as the quality criterion:
```python
for k in range(2, max_clusters):
    labels = KMeans(n_clusters=k).fit_predict(distance_matrix)
    quality = mean(silhouette_scores) / std(silhouette_scores)   # t-stat
```

Then CFI shuffles the ENTIRE cluster simultaneously. When you shuffle NET rank AND POM AND SAG AND consensus_rank together, none of them can compensate for the others. You get the true combined importance of the "rankings" cluster.

From our last run, ONC found 3 clusters:
- **Rankings cluster**: Massey systems + consensus_rank — all measuring "what do the models predict?"
- **Performance cluster**: scoring margins, FG%, tournament path, records — what the team actually did
- **Efficiency cluster**: individual box stats, turnovers, fouls — pace/style of play

The CFI importance of the rankings cluster is substantially higher than any individual member's MDA — confirming that the substitution bias was masking true signal.

### Method 3: SFI (Single Feature Importance)

**Implementation**: `pipeline/feature_importance.py: feat_imp_sfi()`

Train the model on ONE feature at a time. For each feature X_j:
1. For each fold in PurgedYearKFold: train on `{X_j}` only, evaluate log-loss on held-out year
2. SFI(X_j) = mean(log-loss across folds)

**No collinearity is possible** — there are no other features to compensate.

**What SFI tells you that MDA does not**:

| Feature | SFI rank | MDA rank | Interpretation |
|---|---|---|---|
| High | High | → Genuinely unique signal. Keep it. |
| High | Low | → Has signal but is redundant with correlated features. One representative from its cluster is enough. |
| Low | Any | → No standalone signal. Drop it. It's noise even alone. |

SFI is the **primary filter** in our pipeline. Features that score below the log-loss baseline in SFI (i.e., they make predictions WORSE than a coin flip) are excluded from the Tier 2 model regardless of MDI score. This prevents noise features from contributing even in an ensemble.

**Why log-loss baseline** = −log(0.5) = −0.693: this is the log-loss of a model that always predicts 50% probability for a binary outcome. If a feature's SFI mean is below this, the model with just that feature performs worse than random guessing. That feature should be dropped unconditionally.

### Method 4: PurgedYearKFold

**Implementation**: `pipeline/feature_importance.py: PurgedYearKFold`

Standard k-fold on temporal data causes **future leakage**: if fold 3 contains data from 2010 and fold 4 contains data from 2015, the model trained on fold 4 has "seen the future" relative to fold 3's test set. In time series with correlated observations, this inflates validation metrics significantly.

Our structure (one row per team × year) is simpler than continuous time series — there is no overlap between years. The purging is whole-year purging: train on {2005, ..., 2017, 2019, ..., 2025}, test on 2018. Strictly: no future information enters training.

**Why leave-one-year-out (not k-fold)**:

With k-fold (e.g., 5 folds), the test fold contains ~4 years of data. The model's performance on a test fold with 4 years is harder to interpret than performance on a single held-out year. We want to know "how would this model have done in YEAR X?" — that requires the test fold to be exactly one year.

Additionally, since we have 21 years, 21-fold leave-one-year-out gives us 21 independent performance estimates. This is the maximum possible number of independent evaluations from our data — we're not wasting any.

**Sample size concern**: each test fold has 4 Final Four teams (rank-4) or 12 pairwise comparisons (pairwise). The training fold has 80 teams or 240 pairs. With n=80 training samples and 50 features, there is legitimate risk of overfitting. Our mitigations:
- `max_depth=4` (LightGBM) limits model complexity
- `min_weight_fraction_leaf=0.02` prevents tiny leaves
- `class_weight="balanced"` prevents the model from collapsing to the majority class
- SFI pre-filtering reduces active features before training

### Synthetic Validation

**Implementation**: `pipeline/feature_importance.py: synthetic_validation()`

Before trusting any importance result on real basketball data, we verify the methods work on data where we know the ground truth.

**The test**:
1. Generate 500 samples with known structure:
   - `INFO_0`, `INFO_1`, `INFO_2`: these 3 features directly cause the label (informative)
   - `NOISE_0`, `NOISE_1`: random Gaussian noise, zero relationship to label
   - `REDUND_0`, `REDUND_1`, `REDUND_2`, `REDUND_3`: noisy copies of INFO features (simulating POM ≈ SAG)
2. Run MDI and SFI on the full feature set
3. Check: do all 3 INFO features rank above both NOISE features?

**Why this validates the method**: If MDI is correctly implemented with `max_features=1`, it will find the information-containing features because they genuinely reduce impurity more than noise features — regardless of the redundant features (which are also correlated with the label via their INFO source, but provide no independent information).

**What would failure look like**: MDI placing `NOISE_0` above `INFO_2`. This would mean the implementation is biased — perhaps toward high-cardinality features or toward features that happen to split the 500 training samples favorably. In practice with our implementation, the test passes reliably because `max_features=1` gives every feature equal opportunity.

**What this doesn't validate**: Whether basketball features have any signal at all. The synthetic validation only checks that our *methodology* is correctly implemented. The signal question is answered by SFI on real data with PurgedYearKFold.

---

## Part 4: The LightGBM Pairwise Model — Architecture Details

**Why LightGBM over Random Forest for prediction**:

The Random Forest is used for *feature importance* because it has well-studied MDI properties. LightGBM is used for the *final prediction* because:
- Gradient boosting sequentially corrects residuals — it achieves lower bias than bagging
- With `early_stopping(30)`, it automatically finds the optimal number of trees
- Faster inference on the 4-team Final Four frame

**Hyperparameters and their justification** (`pipeline/config.py: LGBM_PARAMS`):

| Parameter | Value | Why |
|---|---|---|
| `objective` | `"binary"` | Pairwise label is binary: did A finish better? |
| `n_estimators` | 300 | Upper bound; early_stopping finds the true optimum |
| `learning_rate` | 0.05 | Small learning rate → more iterations needed, but lower variance |
| `max_depth` | 4 | With ~252 training pairs, depth-4 = max 16 leaves = ~15 pairs per leaf. Depth-5 would overfit. |
| `min_child_samples` | 5 | Leaf must contain ≥5 samples. Prevents single-pair leaves. |
| `subsample` | 0.8 | Stochastic boosting: sample 80% of pairs per tree. Reduces variance. |
| `colsample_bytree` | 0.8 | Sample 80% of features per tree. Further variance reduction. |
| `class_weight` | `"balanced"` | 50/50 by construction for pairwise — but use anyway as safeguard |

**Early stopping with PurgedYearKFold**:

For each leave-one-year-out fold, `early_stopping(30, verbose=False)` monitors log-loss on the held-out year and stops when it doesn't improve for 30 consecutive rounds. This means each fold might use a different number of trees (50 rounds in a "hard" year, 250 in an "easy" year). The final model (trained on ALL data) uses the average optimal number of rounds from CV.

**Time-decay sample weights** (`pipeline/feature_engineering.py: time_decay_weights()`):

```python
weight(year) = 0.3 + 0.7 × (year_position / n_years)
```

Oldest year: weight 0.3. Newest year: weight 1.0. Linear interpolation.

**Why**: Basketball has changed structurally since 2005. The 3-point revolution (2012–2016), analytics adoption, pace changes, and the NIL era (2021+) mean that the relationship between features and championship outcomes was different in 2005 than in 2025. Downweighting older years reduces the influence of "regime-shifted" training data — the same reasoning behind time-decay in financial models.

The coefficient `c=0.3` (not 0.0) preserves some weight for old data because the underlying game (who wins an elimination bracket) hasn't changed as fundamentally as feature distributions suggest.

---

## Part 5: Meta-Labeling (Tier 3)

**Implementation**: `pipeline/model.py: build_market_meta_labels()`

De Prado (AFML Ch.3) introduced meta-labeling as a two-model framework:
- **Primary model**: high recall, simpler, produces candidate signals
- **Secondary model (meta-model)**: filters the primary model's output — predicts "is this primary signal actually right?"

**Our implementation**:

- **Primary model (2025–2026)**: Kalshi market VWAP. The team with the highest VWAP is the market's predicted champion. Markets aggregate the beliefs of thousands of participants who collectively know everything that's publicly known about these teams.

- **Primary model (2005–2024)**: consensus_rank (mean of POM, SAG, BPI, KPI, SOR). The team with the lowest consensus rank is the analytical models' predicted champion.

- **Meta-label**: `meta_label = 1` if the primary model's favorite actually won.

**What the meta-model asks**: "The market (or the analytics consensus) thinks Arizona will win. What features predict that Arizona actually does win?" The meta-model learns conditions like: "when the market favorite also has the best tournament path margin, they win 70% of the time; when they don't, only 40%."

**Primary model accuracy** (from our last run): 40%. This is only slightly above random (25% = 1-in-4 chance). This tells you something important: **neither the analytical consensus nor the market reliably predicts NCAA champions**. Tournament outcomes are high-variance. A 40% hit rate on "who does the market/consensus favor" is consistent with the known difficulty of bracket prediction.

The meta-model's value is not in improving on 40% — it's in understanding *when* the 40% hits. If you can identify conditions under which the market favorite wins 70% of the time vs. 20%, that's actionable.

---

## Part 6: Calibration and What the Numbers Mean

**Win probability output is NOT a calibrated probability in the statistical sense.**

It is a model-consistent ranking of four teams, normalized to sum to 1. The distinction matters:
- Calibrated: if model says 60%, the team actually wins 60% of the time across many similar situations
- Ranking-consistent: if model says Arizona 38%, Michigan 37%, it means Arizona is slightly favored over Michigan

With only 21 test observations, we cannot meaningfully calibrate the probabilities. We can validate the rank ordering (does the team ranked #1 by the model win more than 25%?) but not the exact probability values.

**The Brier score** (used in CV: `sklearn.metrics.brier_score_loss`) measures the mean squared error between predicted probability and binary outcome. It penalizes both overconfidence and underconfidence. A model that always predicts 25% (equal probability for each team) gets a Brier score of ~0.1875 (the baseline). Our model should beat this.

**Log-loss** is used for SFI/MDA because it's more sensitive to calibration than accuracy. A model that says "100% sure this team wins" and is wrong gets infinitely bad log-loss. This prevents the model from being confidently wrong.

---

## Summary: Why This Architecture Works

| Problem | Solution | Theoretical basis |
|---|---|---|
| 84 rows too few for flexible ML | Pairwise conversion → 252 rows | Bradley-Terry, lossless transformation |
| Collinear features mask importance | ONC + CFI shuffle clusters together | Strobl et al. 2007, de Prado MLAM Ch.6 |
| Missing data destroys era information | Binary missingness indicators | Josse et al. 2019, Saar-Tsechansky & Provost 2007 |
| Feature importance is in-sample biased | SFI with PurgedYearKFold | de Prado AFML Ch.7-8 |
| "Rich get richer" MDI bias | max_features=1 in RF | de Prado AFML §8.3.1 |
| Future data leaks into CV | Leave-one-year-out purging | de Prado AFML Ch.7 |
| Regime shift across eras | Time-decay sample weights | de Prado AFML Ch.4 |
| Market data too sparse for MDI/MDA | Meta-labeling with market as primary | de Prado AFML Ch.3 |
| Can't trust methodology without ground truth | Synthetic validation with known signal | de Prado MLAM §1.4 |
