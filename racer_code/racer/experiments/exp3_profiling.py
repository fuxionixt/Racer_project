"""Exp 3 -- profiling: how faithfully does an RF reproduce the POLICY tiers, and how robust is it?
Tiers are derived from the policy formula, so accuracy here is *fidelity*, not validity."""
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold
from racer.metrics import mean_ci_over_runs
from racer.profiling import PolicyScorer, RFProfiler, POLICY_COLS

a = parser(__doc__).parse_args()
df = get_profiles(a).dropna(subset=POLICY_COLS).reset_index(drop=True)
feat_sets = {"policy features (4)": POLICY_COLS,
             "all features": list(df.columns),
             "non-policy features only": [c for c in df.columns if c not in POLICY_COLS]}
rng = np.random.default_rng(config.SEED); out = {}
tier_all = PolicyScorer().fit(df).tier(df)   # only used to stratify folds
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=config.SEED)
for fs, cols in feat_sets.items():
    if not cols: continue
    acc, acc_noise, acc_miss = [], [], []
    for tr, te in cv.split(df, tier_all):
        dtr, dte = df.iloc[tr], df.iloc[te]
        ps = PolicyScorer().fit(dtr)                      # thresholds from TRAIN only
        rf = RFProfiler(cols).fit(dtr, ps.tier(dtr))
        y = ps.tier(dte)
        acc.append(np.mean(rf.predict(dte) == y))
        noisy = dte.copy()
        for c in cols: noisy[c] = noisy[c] + rng.normal(0, 0.1 * dtr[c].std(), len(noisy))
        acc_noise.append(np.mean(rf.predict(noisy) == y))
        miss = dte.copy()
        for c in cols: miss.loc[rng.random(len(miss)) < 0.2, c] = np.nan
        acc_miss.append(np.mean(rf.predict(miss) == y))
    out[fs] = {"fidelity_cv": mean_ci_over_runs(acc), "with_10pct_noise": mean_ci_over_runs(acc_noise),
               "with_20pct_missing": mean_ci_over_runs(acc_miss)}
    print(fs, {k: round(v[0], 3) for k, v in out[fs].items()})
maj = pd.Series(tier_all).value_counts(normalize=True).max() if (pd := __import__("pandas")) else 0
# policy-only coverage when 20% of any policy feature is missing (no imputation possible)
miss = df.copy()
for c in POLICY_COLS: miss.loc[rng.random(len(miss)) < 0.2, c] = np.nan
ps = PolicyScorer().fit(df)
out["policy_formula_coverage_under_20pct_missing"] = float(np.mean([t is not None for t in ps.tier(miss)]))
out["majority_class_rate"] = float(maj)
save(a, "exp3_profiling.json", out)
