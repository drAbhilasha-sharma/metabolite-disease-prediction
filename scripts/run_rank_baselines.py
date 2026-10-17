"""Ranking baselines for the within-metabolite design.

The model's within-metabolite ranking is produced by run_specificity.py
(design == "within_metabolite"). This script computes the reference baselines
that the model must be compared against, and writes:

  results/ranking_baselines.csv        headline metrics, model vs baselines
  results/table_ranking_vs_baseline.csv  paired tests, model vs prevalence

Baselines
---------
uniform      diseases ranked in random order (expectation over permutations,
             estimated by simulation with a fixed seed)
prevalence   diseases ranked by number of associated metabolites, using no
             information about the query metabolite
analytical   n_true / n_diseases, the closed-form hit@1 for a random draw

The prevalence baseline is the meaningful comparator: association resources
have strongly unequal disease degrees, so a metabolite-agnostic ranker already
scores far above uniform chance.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

from run_specificity import DISEASES, MDIS, POS, SEEDS

N_SIM = 200


def metrics_from_ranks(r1):
    r1 = np.asarray(r1, dtype=float)
    return dict(hits1=float((r1 == 1).mean()), hits5=float((r1 <= 5).mean()),
                hits10=float((r1 <= 10).mean()), mrr=float((1.0 / r1).mean()))


def best_rank(order, true_set):
    """1-based rank of the highest-ranked true disease."""
    for i, d in enumerate(order, start=1):
        if d in true_set:
            return i
    return len(order) + 1


def main():
    spec = pd.read_csv("results/specificity_all.csv")
    W = spec[spec.design == "within_metabolite"].copy()
    mets = sorted(W.hmdb_id.unique())
    truth = {m: set(MDIS[m]) for m in mets}

    # ---- model: average per metabolite across seeds/folds, then over metabolites
    per_met = W.groupby("hmdb_id")[["hit1", "hit5", "hit10", "mrr"]].mean()
    model = dict(hits1=per_met.hit1.mean(), hits5=per_met.hit5.mean(),
                 hits10=per_met.hit10.mean(), mrr=per_met.mrr.mean())

    # ---- prevalence baseline: rank diseases by number of metabolites
    size = POS.groupby("disease").hmdb_id.nunique()
    prev_order = list(size.reindex(DISEASES).fillna(0).sort_values(
        ascending=False).index)
    prev_r1 = {m: best_rank(prev_order, truth[m]) for m in mets}
    prevalence = metrics_from_ranks(list(prev_r1.values()))

    # ---- uniform baseline: an independent random ranking per metabolite.
    # A shared permutation would place the same disease first for every
    # metabolite, which correlates the estimates and inflates their variance.
    rng = np.random.default_rng(20250808)
    acc = []
    for _ in range(N_SIM):
        r1 = [best_rank(list(rng.permutation(DISEASES)), truth[m]) for m in mets]
        acc.append(metrics_from_ranks(r1))
    uniform = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}

    # ---- analytical chance for hits@1
    analytical = float(np.mean([len(truth[m]) / len(DISEASES) for m in mets]))

    out = pd.DataFrame([
        dict(baseline="Uniform random", **uniform),
        dict(baseline="Disease prevalence (size only)", **prevalence),
        dict(baseline=f"Analytical chance (n_true/{len(DISEASES)})",
             hits1=analytical, hits5=np.nan, hits10=np.nan, mrr=np.nan),
        dict(baseline="MODEL (measured)", **model),
    ])
    out.to_csv("results/ranking_baselines.csv", index=False)

    # ---- paired tests: model vs prevalence, unit = metabolite
    rows = []
    mm = per_met.reindex(mets)
    pv = {
        "hits@1": np.array([1.0 if prev_r1[m] == 1 else 0.0 for m in mets]),
        "hits@5": np.array([1.0 if prev_r1[m] <= 5 else 0.0 for m in mets]),
        "hits@10": np.array([1.0 if prev_r1[m] <= 10 else 0.0 for m in mets]),
        "MRR": np.array([1.0 / prev_r1[m] for m in mets]),
    }
    md = {"hits@1": mm.hit1.values, "hits@5": mm.hit5.values, "hits@10": mm.hit10.values,
          "MRR": mm.mrr.values}
    for name in ["hits@1", "hits@5", "hits@10", "MRR"]:
        a, b = md[name], pv[name]
        d = a - b
        t = stats.ttest_rel(a, b)
        try:
            w = stats.wilcoxon(a, b).pvalue
        except ValueError:
            w = np.nan
        # Independent stream per metric so the interval does not depend on
        # how many metrics precede it in the loop.
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        brng = np.random.default_rng(seed)
        bs = brng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
        rows.append(dict(metric=name, model=a.mean(), prevalence_baseline=b.mean(),
                         diff=d.mean(), ci_lo=np.percentile(bs, 2.5),
                         ci_hi=np.percentile(bs, 97.5), p_t=t.pvalue,
                         p_wilcoxon=w, dz=d.mean() / d.std(ddof=1),
                         n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()),
                         n=len(d)))
    tab = pd.DataFrame(rows)
    tab.to_csv("results/table_ranking_vs_baseline.csv", index=False)

    print(out.round(4).to_string(index=False))
    print()
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
