"""
filter_features.py
------------------
Applies de Prado's MDI / MDA / SFI filtering criteria to raw importance CSVs.
No normality assumption. Uses CLT for MDA (valid at ~21 folds), 
Wilcoxon against a known null for MDI/SFI, bootstrap CI for robustness.

Usage:
    python filter_features.py

Outputs:
    surviving_features.csv   - features that pass ALL three filters
    feature_report.csv       - full per-feature breakdown with pass/fail per criterion
    feature_report.html      - visual summary (open in browser)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

BASE = '/Users/michaelharoon/Projects/tasty/march-madness/output_v2/features/'
FILES = {
    "mdi": os.path.join(BASE, 'importance_mdi_raw.csv'),
    "mda": os.path.join(BASE, 'importance_mda_raw.csv'),
    "sfi": os.path.join(BASE, 'importance_sfi_raw.csv'),
}

# de Prado MDA CLT threshold: how many std-errors above 0 is "significant"?
# With 21 folds, CLT holds well. Use 1.0 (weak) or 1.65 (one-sided 5%) or 2.0 (stricter)
MDA_Z_THRESHOLD = 1.0        # mean / (std / sqrt(n_folds)) > this → significant
                              # intentionally weak because n=21 is small

# SFI baseline: log-loss of a coin-flip classifier
# If your SFI scores are log-loss: higher (less negative) = better
# -log(0.5) ≈ -0.693 is the random baseline for binary log-loss
# Adjust to 0.5 if SFI scores are accuracy or AUC
SFI_LOG_LOSS_BASELINE = -np.log(0.5)   # ≈ -0.693

# Bootstrap config for MDI confidence intervals
N_BOOTSTRAP = 2000
BOOTSTRAP_CI = 0.95

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_importance_file(path: str, name: str) -> pd.DataFrame:
    """
    Load a CSV where:
      - First column = Fold/Iteration index (to be ignored)
      - Subsequent columns = Feature names
      - Rows = Values for those features
    """
    if not os.path.exists(path):
        print(f"  ⚠  {name} file not found: {path}")
        return pd.DataFrame()
    
    # 1. Load the CSV
    df = pd.read_csv(path)
    
    # 2. Drop the first column (the index/fold label)
    df = df.iloc[:, 1:]
    
    # 3. Transpose so Features become the INDEX
    # This makes the DataFrame: Rows = Features, Columns = Folds
    df = df.T
    print(df.columns,df.head())
    
    print(f"  {name}: {df.shape[0]} features detected across {df.shape[1]} folds.")
    return df

def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, ci: float = BOOTSTRAP_CI):
    """
    Bootstrap confidence interval for the mean. No normality assumption.
    Returns (mean, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    return values.mean(), np.percentile(boot_means, 100 * alpha / 2), np.percentile(boot_means, 100 * (1 - alpha / 2))


def wilcoxon_vs_null(values: np.ndarray, null_value: float):
    """
    Wilcoxon signed-rank test: is the median of 'values' significantly 
    different from null_value?
    
    This is appropriate when:
    - You don't know the population mean
    - You don't want to assume normality
    - You DO have a known null hypothesis value (e.g., 1/F for MDI, 0 for MDA)
    
    Returns p-value (one-sided: values > null_value).
    """
    diffs = values - null_value
    diffs = diffs[diffs != 0]  # Wilcoxon requires no ties at zero
    if len(diffs) < 4:
        return np.nan  # not enough data for the test
    
    # One-sided: testing if values are systematically ABOVE null_value
    stat, p_two = stats.wilcoxon(diffs, alternative='greater')
    return p_two  # already one-sided with alternative='greater'


# ─── STEP 1: Load files ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1: Loading importance files")
print("="*60)

raw = {}
for name, path in FILES.items():
    raw[name] = load_importance_file(path, name.upper())

# Identify the feature universe (intersection is safest; union loses some methods)
all_features = set()
for df in raw.values():
    if not df.empty:
        all_features.update(df.index.tolist())
print(f"\n  Total unique features across all files: {len(all_features)}")

# ─── STEP 2: Compute per-feature statistics ───────────────────────────────────

print("\n" + "="*60)
print("STEP 2: Computing per-feature statistics")
print("="*60)

report_rows = []

for feat in sorted(all_features):
    row = {"feature": feat}

    # ── MDI ──────────────────────────────────────────────────────────────────
    if not raw["mdi"].empty and feat in raw["mdi"].index:
        mdi_vals = raw["mdi"].loc[feat].dropna().values.astype(float)
        F = len(all_features)
        threshold_1_over_F = 1.0 / F
        
        if len(mdi_vals) > 0:
            mdi_mean = mdi_vals.mean()
            mdi_std  = mdi_vals.std()
            n_mdi    = len(mdi_vals)
            
            # Bootstrap CI (no normality assumption)
            if n_mdi >= 4:
                _, mdi_ci_lo, mdi_ci_hi = bootstrap_ci(mdi_vals)
            else:
                mdi_ci_lo = mdi_ci_hi = mdi_mean
            
            # Wilcoxon against 1/F null (non-parametric test)
            p_mdi = wilcoxon_vs_null(mdi_vals, threshold_1_over_F)
            
            row.update({
                "mdi_mean":      round(mdi_mean, 6),
                "mdi_std":       round(mdi_std, 6),
                "mdi_n_folds":   n_mdi,
                "mdi_1_over_F":  round(threshold_1_over_F, 6),
                "mdi_ci_lo":     round(mdi_ci_lo, 6),
                "mdi_ci_hi":     round(mdi_ci_hi, 6),
                "mdi_p_vs_1F":   round(p_mdi, 4) if not np.isnan(p_mdi) else np.nan,
                # PASS if: mean > 1/F AND (CI lower bound > 1/F OR p < 0.10)
                "mdi_passes":    (
                    mdi_mean > threshold_1_over_F and
                    (mdi_ci_lo > threshold_1_over_F or (not np.isnan(p_mdi) and p_mdi < 0.10))
                ),
            })
        else:
            row.update({k: np.nan for k in ["mdi_mean","mdi_std","mdi_n_folds",
                                              "mdi_1_over_F","mdi_ci_lo","mdi_ci_hi",
                                              "mdi_p_vs_1F"]})
            row["mdi_passes"] = False
    else:
        row.update({k: np.nan for k in ["mdi_mean","mdi_std","mdi_n_folds",
                                          "mdi_1_over_F","mdi_ci_lo","mdi_ci_hi",
                                          "mdi_p_vs_1F","mdi_passes"]})

    # ── MDA ──────────────────────────────────────────────────────────────────
    if not raw["mda"].empty and feat in raw["mda"].index:
        mda_vals = raw["mda"].loc[feat].dropna().values.astype(float)
        
        if len(mda_vals) > 0:
            mda_mean  = mda_vals.mean()
            mda_std   = mda_vals.std()
            n_mda     = len(mda_vals)
            
            # CLT-based z-score: valid at n≈21 folds even without normality
            # (CLT says the mean of n observations → Normal as n grows)
            mda_se    = mda_std / np.sqrt(n_mda) if n_mda > 1 else np.inf
            mda_z     = mda_mean / mda_se if mda_se > 0 else 0
            
            # Also do Wilcoxon vs 0 (non-parametric complement)
            p_mda_wilcoxon = wilcoxon_vs_null(mda_vals, 0.0)
            
            # Bootstrap CI
            if n_mda >= 4:
                _, mda_ci_lo, mda_ci_hi = bootstrap_ci(mda_vals)
            else:
                mda_ci_lo = mda_ci_hi = mda_mean
            
            row.update({
                "mda_mean":         round(mda_mean, 6),
                "mda_std":          round(mda_std, 6),
                "mda_n_folds":      n_mda,
                "mda_se":           round(mda_se, 6),
                "mda_z":            round(mda_z, 3),
                "mda_ci_lo":        round(mda_ci_lo, 6),
                "mda_ci_hi":        round(mda_ci_hi, 6),
                "mda_p_wilcoxon":   round(p_mda_wilcoxon, 4) if not np.isnan(p_mda_wilcoxon) else np.nan,
                # HARD FAIL: negative mean = shuffling HELPED = feature is noise/harmful
                "mda_detrimental":  mda_mean < 0,
                # PASS if: mean > 0 AND z > threshold AND not detrimental
                "mda_passes":       (
                    mda_mean > 0 and
                    mda_z >= MDA_Z_THRESHOLD and
                    not (mda_mean < 0)
                ),
            })
        else:
            row.update({k: np.nan for k in ["mda_mean","mda_std","mda_n_folds",
                                              "mda_se","mda_z","mda_ci_lo","mda_ci_hi",
                                              "mda_p_wilcoxon"]})
            row["mda_detrimental"] = False
            row["mda_passes"] = False
    else:
        row.update({k: np.nan for k in ["mda_mean","mda_std","mda_n_folds",
                                          "mda_se","mda_z","mda_ci_lo","mda_ci_hi",
                                          "mda_p_wilcoxon","mda_detrimental","mda_passes"]})

    # ── SFI ──────────────────────────────────────────────────────────────────
    if not raw["sfi"].empty and feat in raw["sfi"].index:
        sfi_vals = raw["sfi"].loc[feat].dropna().values.astype(float)
        
        if len(sfi_vals) > 0:
            sfi_mean = sfi_vals.mean()
            sfi_std  = sfi_vals.std()
            n_sfi    = len(sfi_vals)
            
            # Bootstrap CI
            if n_sfi >= 4:
                _, sfi_ci_lo, sfi_ci_hi = bootstrap_ci(sfi_vals)
            else:
                sfi_ci_lo = sfi_ci_hi = sfi_mean
            
            # Wilcoxon against the log-loss baseline
            p_sfi = wilcoxon_vs_null(sfi_vals, SFI_LOG_LOSS_BASELINE)
            
            # PASS: mean is above the baseline (i.e., better than a coin flip)
            # For log-loss: more negative = worse; above baseline means less negative
            # But in de Prado's framework, SFI scores can be stored as the 
            # IMPROVEMENT over baseline, or as the raw score. 
            # We handle both cases:
            sfi_above_baseline = sfi_mean > SFI_LOG_LOSS_BASELINE
            
            row.update({
                "sfi_mean":           round(sfi_mean, 6),
                "sfi_std":            round(sfi_std, 6),
                "sfi_n_folds":        n_sfi,
                "sfi_ci_lo":          round(sfi_ci_lo, 6),
                "sfi_ci_hi":          round(sfi_ci_hi, 6),
                "sfi_baseline":       round(SFI_LOG_LOSS_BASELINE, 6),
                "sfi_p_vs_baseline":  round(p_sfi, 4) if not np.isnan(p_sfi) else np.nan,
                "sfi_above_baseline": sfi_above_baseline,
                # PASS if: mean is above baseline AND CI lower > baseline
                "sfi_passes":         (
                    sfi_above_baseline and
                    sfi_ci_lo > SFI_LOG_LOSS_BASELINE
                ),
            })
        else:
            row.update({k: np.nan for k in ["sfi_mean","sfi_std","sfi_n_folds",
                                              "sfi_ci_lo","sfi_ci_hi","sfi_baseline",
                                              "sfi_p_vs_baseline","sfi_above_baseline",
                                              "sfi_passes"]})
    else:
        row.update({k: np.nan for k in ["sfi_mean","sfi_std","sfi_n_folds",
                                          "sfi_ci_lo","sfi_ci_hi","sfi_baseline",
                                          "sfi_p_vs_baseline","sfi_above_baseline","sfi_passes"]})

    report_rows.append(row)


report = pd.DataFrame(report_rows).set_index("feature")

# ─── STEP 3: Combined pass/fail and ranking ───────────────────────────────────

print("\n" + "="*60)
print("STEP 3: Combined filtering")
print("="*60)

# Count how many methods are available per feature
report["n_methods_available"] = (
    report["mdi_passes"].notna().astype(int) +
    report["mda_passes"].notna().astype(int) +
    report["sfi_passes"].notna().astype(int)
)

# Count passes
report["n_methods_passed"] = (
    report["mdi_passes"].fillna(False).astype(int) +
    report["mda_passes"].fillna(False).astype(int) +
    report["sfi_passes"].fillna(False).astype(int)
)

# Hard kill: detrimental in MDA = disqualified regardless
report["mda_kills"] = report["mda_detrimental"].fillna(False)

# Survival tiers:
#   STRONG: passes all available methods (no kills)
#   MODERATE: passes 2/3 available methods (no kills)
#   WEAK: passes 1/3 (keep as candidates)
#   REJECTED: passes 0 OR has a MDA kill

def assign_tier(row):
    if row["mda_kills"]:
        return "REJECTED (detrimental)"
    n_pass = row["n_methods_passed"]
    n_avail = row["n_methods_available"]
    if n_avail == 0:
        return "UNKNOWN (not in any file)"
    ratio = n_pass / n_avail
    if ratio == 1.0:
        return "STRONG"
    elif ratio >= 0.67:
        return "MODERATE"
    elif ratio >= 0.34:
        return "WEAK"
    else:
        return "REJECTED"

report["tier"] = report.apply(assign_tier, axis=1)

# Composite rank: average MDI/MDA/SFI ranks (lower = better)
for method in ["mdi", "mda", "sfi"]:
    col = f"{method}_mean"
    if col in report.columns:
        report[f"{method}_rank"] = report[col].rank(ascending=False, na_option="bottom")

rank_cols = [c for c in ["mdi_rank", "mda_rank", "sfi_rank"] if c in report.columns]
if rank_cols:
    report["composite_rank"] = report[rank_cols].mean(axis=1)
    report = report.sort_values("composite_rank")

# ─── STEP 4: Print summary ────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 4: Results")
print("="*60)

tier_counts = report["tier"].value_counts()
print("\nTier breakdown:")
for tier, count in tier_counts.items():
    print(f"  {tier}: {count} features")

print("\n─── STRONG features (pass all available methods) ───")
strong = report[report["tier"] == "STRONG"]
if len(strong) > 0:
    cols_to_show = ["mdi_mean", "mda_mean", "mda_z", "sfi_mean", "composite_rank"]
    cols_present = [c for c in cols_to_show if c in strong.columns]
    print(strong[cols_present].to_string())
else:
    print("  None found — loosening criteria or check file format")

print("\n─── MODERATE features (pass 2/3 methods) ───")
moderate = report[report["tier"] == "MODERATE"]
if len(moderate) > 0:
    cols_to_show = ["mdi_mean", "mdi_passes", "mda_mean", "mda_passes", "sfi_mean", "sfi_passes", "composite_rank"]
    cols_present = [c for c in cols_to_show if c in moderate.columns]
    print(moderate[cols_present].to_string())
else:
    print("  None found")

print("\n─── DETRIMENTAL features (MDA negative — eliminate immediately) ───")
detrimental = report[report["mda_kills"] == True]
if len(detrimental) > 0:
    print(detrimental[["mda_mean", "mda_z"]].to_string())
else:
    print("  None found (good!)")

# ─── STEP 5: Save outputs ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 5: Saving outputs")
print("="*60)

out_dir = os.path.join(BASE, "filtered")
os.makedirs(out_dir, exist_ok=True)

# Full report
report_path = os.path.join(out_dir, "feature_report.csv")
report.to_csv(report_path)
print(f"  Full report → {report_path}")

# Surviving features (STRONG + MODERATE)
survivors = report[report["tier"].isin(["STRONG", "MODERATE"])].copy()
survivors_path = os.path.join(out_dir, "surviving_features.csv")
survivors.to_csv(survivors_path)
print(f"  Survivors ({len(survivors)} features) → {survivors_path}")

# Clean list for pipeline use
feature_list = survivors.index.tolist()
list_path = os.path.join(out_dir, "feature_list.txt")
with open(list_path, "w") as f:
    f.write("\n".join(feature_list))
print(f"  Feature list → {list_path}")

# HTML visual summary
try:
    html_rows = []
    for feat, row in report.iterrows():
        tier = row["tier"]
        color = {
            "STRONG": "#d4edda",
            "MODERATE": "#fff3cd",
            "WEAK": "#f8f9fa",
            "REJECTED": "#f8d7da",
            "REJECTED (detrimental)": "#dc3545",
        }.get(tier, "white")
        
        def fmt(val, decimals=4):
            if pd.isna(val):
                return "—"
            if isinstance(val, bool):
                return "✅" if val else "❌"
            try:
                return f"{float(val):.{decimals}f}"
            except:
                return str(val)
        
        html_rows.append(f"""
        <tr style="background:{color}">
            <td><b>{feat}</b></td>
            <td>{tier}</td>
            <td>{fmt(row.get('mdi_mean'))}</td>
            <td>{"✅" if row.get('mdi_passes') else "❌"}</td>
            <td>{fmt(row.get('mda_mean'))}</td>
            <td>{fmt(row.get('mda_z'), 2)}</td>
            <td>{"✅" if row.get('mda_passes') else "❌"}</td>
            <td>{fmt(row.get('sfi_mean'))}</td>
            <td>{"✅" if row.get('sfi_passes') else "❌"}</td>
            <td>{fmt(row.get('composite_rank'), 1)}</td>
        </tr>""")
    
    html = f"""<!DOCTYPE html><html><head><style>
    body {{font-family: monospace; font-size: 12px;}}
    table {{border-collapse: collapse; width: 100%;}}
    th, td {{border: 1px solid #ddd; padding: 4px 8px; text-align: right;}}
    th {{background: #343a40; color: white; position: sticky; top: 0;}}
    td:first-child, td:nth-child(2) {{text-align: left;}}
    </style></head><body>
    <h2>Feature Importance Report — {len(report)} features</h2>
    <p>MDA Z-threshold: {MDA_Z_THRESHOLD} | SFI baseline: {SFI_LOG_LOSS_BASELINE:.4f} | Bootstrap CI: {BOOTSTRAP_CI:.0%}</p>
    <table>
    <tr>
        <th>Feature</th><th>Tier</th>
        <th>MDI mean</th><th>MDI ✓</th>
        <th>MDA mean</th><th>MDA z</th><th>MDA ✓</th>
        <th>SFI mean</th><th>SFI ✓</th>
        <th>Composite Rank</th>
    </tr>
    {"".join(html_rows)}
    </table></body></html>"""
    
    html_path = os.path.join(out_dir, "feature_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  HTML report → {html_path}")
except Exception as e:
    print(f"  HTML generation failed: {e}")

print(f"\n✅ Done. {len(survivors)} features survive into the next stage.")
print(f"   STRONG ({len(report[report['tier']=='STRONG'])}): use in all models")
print(f"   MODERATE ({len(report[report['tier']=='MODERATE'])}): use with caution, flag for PCA check")
print(f"   Next step: run PCA + Kendall's tau on surviving features (see script comments)")