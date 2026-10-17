"""Sensitivity checks of the sampler, the unit of analysis and the learning-curve
fit.

Reads only files already in results/ (and rebuild/model_positives.csv when
present), and writes results/sensitivity_checks.csv:

  * sampler diagnostics of degree_matched_pairs(): fallback draws, repeated
    (reused) negatives, mean degree of positives and negatives, and the
    per-disease and pooled AUROC of annotation degree used alone;
  * the primary random-forest summaries with each of the 64 merged disease
    groups weighted once instead of each of the 71 diseases;
  * the full model against the within-disease permutation null;
  * sensitivity of the tenfold learning-curve continuation to the fit.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
from degree_matched import degree_matched_pairs

SEEDS = [0, 1, 2, 3, 4]
rows = []


def add(check, quantity, value):
    rows.append(dict(check=check, quantity=quantity, value=value))


def tci(x):
    x = np.asarray(x, float)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def tost_vs_half(x, delta=0.05):
    x = np.asarray(x, float) - 0.5
    se = x.std(ddof=1) / np.sqrt(len(x))
    p1 = st.t.sf((x.mean() + delta) / se, len(x) - 1)
    p2 = st.t.cdf((x.mean() - delta) / se, len(x) - 1)
    return max(p1, p2)


# ------------------------------------------------ sampler diagnostics
if os.path.exists("rebuild/model_positives.csv"):
    POS = pd.read_csv("rebuild/model_positives.csv")[["hmdb_id", "disease"]]
else:  # the 4,791 positives are also listed one per row here
    POS = pd.read_csv("results/per_metabolite_rank.csv")[["hmdb_id", "disease"]]
POS = POS.drop_duplicates()
deg = POS.groupby("hmdb_id").disease.nunique()
neg_all = []
for seed in SEEDS:
    P = degree_matched_pairs(POS, np.random.default_rng(100 + seed))
    P["deg"] = P.hmdb_id.map(deg)
    pos, neg = P[P.label == 1], P[P.label == 0]
    neg_all.append(neg)
    per = [roc_auc_score(g.label, g.deg) for _, g in P.groupby("disease")]
    add("sampler", f"seed{seed}_fallback_draws",
        int((neg.deg.values != pos.deg.values).sum()))
    add("sampler", f"seed{seed}_repeated_negatives",
        int(neg.duplicated(["hmdb_id", "disease"]).sum()))
    add("sampler", f"seed{seed}_mean_degree_pos", pos.deg.mean())
    add("sampler", f"seed{seed}_mean_degree_neg", neg.deg.mean())
    add("sampler", f"seed{seed}_degree_auc_per_disease_mean", np.mean(per))
    add("sampler", f"seed{seed}_degree_auc_per_disease_min", np.min(per))
    add("sampler", f"seed{seed}_degree_auc_per_disease_max", np.max(per))
    add("sampler", f"seed{seed}_diseases_exactly_half",
        int(np.sum(np.isclose(per, 0.5))))
    add("sampler", f"seed{seed}_degree_auc_pooled", roc_auc_score(P.label, P.deg))
add("sampler", "distinct_negative_pairs_all_seeds",
    pd.concat(neg_all).drop_duplicates(["hmdb_id", "disease"]).shape[0])

# ------------------------------------------- 71 diseases vs 64 groups
grp = json.load(open("rebuild/disease_groups.json"))  # disease -> shared-descriptor group
D = pd.read_csv("results/definitive_per_disease.csv")
rf = D[D.model == "rf"].groupby(["regime", "mode", "disease"]).auc.mean().reset_index()
rf["group"] = rf.disease.map(grp)
for (reg, mode), g in rf.groupby(["regime", "mode"]):
    for unit, x in [("71_diseases", g.auc.values),
                    ("64_groups", g.groupby("group").auc.mean().values)]:
        m, lo, hi = tci(x)
        add("units", f"{reg}_{mode}_{unit}_mean", m)
        add("units", f"{reg}_{mode}_{unit}_ci_lo", lo)
        add("units", f"{reg}_{mode}_{unit}_ci_hi", hi)
for reg in ["lmo", "ldo"]:
    w = rf[rf.regime == reg].pivot(index="disease", columns="mode", values="auc")
    G = w.groupby(w.index.map(grp)).mean()
    for unit, X in [("71_diseases", w), ("64_groups", G)]:
        dlt = X["full"] - X["perm"]
        add("units", f"{reg}_gain_vs_perm_{unit}", dlt.mean())
        add("units", f"{reg}_gain_vs_perm_{unit}_p_t", st.ttest_1samp(dlt, 0).pvalue)
        if reg == "ldo":
            add("units", f"ldo_full_tost_{unit}", tost_vs_half(X["full"]))

# ------------------------------------------- within-disease permutation
W = pd.read_csv("results/perm_within.csv")
for reg in ["lmo", "ldo"]:
    f = D[(D.model == "rf") & (D.regime == reg) & (D["mode"] == "full")].groupby("disease").auc.mean()
    p = W[(W.model == "rf") & (W.regime == reg)].groupby("disease").auc.mean()
    dlt = (f - p).dropna()
    add("within_null", f"{reg}_full", f.mean())
    add("within_null", f"{reg}_perm_within", p.mean())
    add("within_null", f"{reg}_diff", dlt.mean())
    add("within_null", f"{reg}_p_t", st.ttest_1samp(dlt, 0).pvalue)
    add("within_null", f"{reg}_p_wilcoxon", st.wilcoxon(dlt).pvalue)
    add("within_null", f"{reg}_n_improved", int((dlt > 0).sum()))

# ------------------------------------------- learning-curve continuation
L = pd.read_csv("results/learning_curve.csv")
L = L[L.regime == "lmo"]
NTRAIN = 7666.0
power = lambda n, a, b, c: a - b * n ** (-c)
curves = [(f"seed{s}", g.fraction.values, g.mean_auc.values) for s, g in L.groupby("seed")]
mg = L.groupby("fraction").mean_auc.mean()
curves.append(("mean", mg.index.values, mg.values))
for name, fr, y in curves:
    x = fr * NTRAIN
    p, _ = curve_fit(power, x, y, p0=[0.8, 1.0, 0.4],
                     bounds=([0.5, 0, 0.01], [1.0, 50, 3]), maxfev=40000)
    add("learning_curve", f"{name}_power_asymptote", p[0])
    add("learning_curve", f"{name}_power_at_10x", power(10 * NTRAIN, *p))
    add("learning_curve", f"{name}_loglinear_at_10x",
        np.polyval(np.polyfit(np.log10(x), y, 1), np.log10(10 * NTRAIN)))

out = pd.DataFrame(rows)
out.to_csv("results/sensitivity_checks.csv", index=False)
print(out.to_string(index=False))
