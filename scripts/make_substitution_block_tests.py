"""Real versus random-Gaussian disease block in the distance-stratified
substitution analysis (Supplementary Table 9).

Reads results/substitution_slope.csv and substitution_strata.csv (real block)
and results/substitution_slope_rand.csv and substitution_strata_rand.csv
(random block; MDA_STRAT_BLOCK=random run_substitution_stratified.py). For each
block the slope is a per-seed least-squares fit of AUROC change on realized
mean substitution distance, summarized by a t-interval over seeds. Computes:

  1. the random-block slope: mean, 95% CI, one-sample t-test against 0;
  2. the paired difference of slopes by seed, real minus random;
  3. the slopes refitted within the range of realized substitution distance
     that both runs share (overlap of their pooled realized ranges), per seed,
     with a paired t-test.

Writes results/substitution_block_tests.csv and prints every value.
"""
import numpy as np
import pandas as pd
from scipy import stats


def ci95(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def per_seed_slopes(R):
    return pd.Series({s: np.polyfit(g.dist, g.delta, 1)[0] for s, g in R.groupby("seed")})


def main():
    Fr = pd.read_csv("results/substitution_slope.csv").set_index("seed")
    Fg = pd.read_csv("results/substitution_slope_rand.csv").set_index("seed")
    Rr = pd.read_csv("results/substitution_strata.csv")
    Rg = pd.read_csv("results/substitution_strata_rand.csv")
    assert list(Fr.index) == list(Fg.index), "the two runs must cover the same seeds"
    rows = []

    def add(q, m, lo=np.nan, hi=np.nan, p=np.nan, n=len(Fr), note=""):
        rows.append(dict(quantity=q, value=m, ci_lo=lo, ci_hi=hi, p=p, n_seeds=n, note=note))

    m, lo, hi = ci95(Fr.slope)
    add("slope, real block", m, lo, hi, stats.ttest_1samp(Fr.slope, 0).pvalue)
    m, lo, hi = ci95(Fg.slope)
    add("slope, random block", m, lo, hi, stats.ttest_1samp(Fg.slope, 0).pvalue)
    m, lo, hi = ci95(Fg.intercept)
    add("intercept, random block", m, lo, hi, stats.ttest_1samp(Fg.intercept, 0).pvalue)
    d = Fr.slope - Fg.slope
    m, lo, hi = ci95(d)
    add("slope difference, real minus random (paired by seed)", m, lo, hi,
        stats.ttest_rel(Fr.slope, Fg.slope).pvalue)

    lo_d = max(Rr.dist.min(), Rg.dist.min())
    hi_d = min(Rr.dist.max(), Rg.dist.max())
    note = f"shared realised distance range {lo_d:.3f} to {hi_d:.3f}"
    sr = per_seed_slopes(Rr[Rr.dist.between(lo_d, hi_d)])
    sg = per_seed_slopes(Rg[Rg.dist.between(lo_d, hi_d)])
    sg = sg.loc[sr.index]
    m, lo, hi = ci95(sr)
    add("slope within shared range, real block", m, lo, hi, stats.ttest_1samp(sr, 0).pvalue,
        note=note + f"; {int(Rr.dist.between(lo_d, hi_d).sum())} of {len(Rr)} draws")
    m, lo, hi = ci95(sg)
    add("slope within shared range, random block", m, lo, hi, stats.ttest_1samp(sg, 0).pvalue,
        note=note + f"; {int(Rg.dist.between(lo_d, hi_d).sum())} of {len(Rg)} draws")
    m, lo, hi = ci95(sr - sg)
    add("slope difference within shared range, real minus random", m, lo, hi,
        stats.ttest_rel(sr, sg).pvalue, note=note)

    out = pd.DataFrame(rows)
    out.to_csv("results/substitution_block_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
