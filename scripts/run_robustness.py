"""Robustness of the interpolation/extrapolation gap to design choices.

Four experiments, each closing a specific objection to the main result.

  temporal    A prospective holdout. Every disease block in the source release
              carries dated literature references; the earliest supporting
              reference is the year the association entered the literature.
              Training uses associations first reported before a cutoff,
              testing uses those first reported after it. This is a stronger
              test of real-world use than a structural split, because it asks
              the model to anticipate discoveries that had not been made when
              its training data was assembled. Two arms are run: the ordinary
              one, and one restricted to test diseases that also appear in
              training (temporal interpolation) versus those that do not.

  sampler     The degree-matched sampler is the most load-bearing design
              choice in the paper. Six variants are run across both regimes:
              exact degree match at 1:1 (the paper's configuration), a
              tolerance of +/-1 degree stratum, 1:2 and 1:3 positive-negative
              ratios, matching jointly on degree and molecular-size class, and
              a uniform sampler as the permissive extreme.

  seeds       The extrapolation claim rests on an equivalence test, so its
              stability in the number of seeds matters. The extrapolation
              regime is re-run over 20 seeds and the equivalence p-value and
              confidence-interval half-width are recomputed cumulatively at
              each seed count, showing where they stabilise.

  corrob      Curation noise. Associations are split into those corroborated
              by more than one supporting reference and those resting on a
              single report, and both regimes are re-run within each subset.
              If the near-chance extrapolation result were an artefact of
              noisy single-source curation it should lift in the corroborated
              subset.

Section is selected by argv[1]: temporal | sampler | seeds | corrob | all
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from degree_matched import degree_strata, degree_matched_pairs
from run_definitive import fit_predict, per_disease_pooled
from run_specificity import DISEASES, POS, FPX, FPI, DFv, DFI, N_FOLDS

SIBLING_OF = json.load(open("rebuild/sibling_groups.json"))
MODEL = "rf"
os.makedirs("results", exist_ok=True)

# Molecular weights are needed only by the degree_plus_size sampler. They come
# from rebuild/descriptors_all.csv (compute_descriptors_all.py); without it the
# other analyses still run and that sampler raises a clear error.
if os.path.exists("rebuild/descriptors_all.csv"):
    DESC = pd.read_csv("rebuild/descriptors_all.csv")
    MW = DESC.set_index("hmdb_id")["mw"] if "mw" in DESC.columns else None
else:
    MW = None


def folds_for(regime, P, seed):
    if regime == "lmo":
        return evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)
    return evalproto.disease_grouped_folds(P, DISEASES, N_FOLDS, seed,
                                           sibling_of=SIBLING_OF)


def X_of(P):
    return np.hstack([FPX[[FPI[m] for m in P.hmdb_id]],
                      DFv[[DFI[d] for d in P.disease]]])


def cv_auc(P, regime, seed, n_folds=N_FOLDS):
    """Out-of-fold predictions, skipping folds a small subset leaves empty."""
    X, y = X_of(P), P.label.values
    pred = np.full(len(P), np.nan)
    for tr, te in folds_for(regime, P, seed):
        if len(tr) == 0 or len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        pred[te] = fit_predict(MODEL, seed, X[tr], y[tr], X[te])
    ok = ~np.isnan(pred)
    return per_disease_pooled(P[ok].reset_index(drop=True), pred[ok])


# ----------------------------------------------------------------- samplers
def sample_variant(pos, rng, variant):
    """Return a labelled pair table under one negative-sampling variant."""
    if variant == "exact_1to1":
        return degree_matched_pairs(pos, rng)
    if variant.startswith("ratio_"):
        k = int(variant.split("_")[1])
        parts = [degree_matched_pairs(pos, rng) for _ in range(k)]
        base = parts[0]
        negs = pd.concat([p[p.label == 0] for p in parts], ignore_index=True)
        return pd.concat([base[base.label == 1], negs], ignore_index=True)
    deg = pos.groupby("hmdb_id").disease.nunique()
    strat = degree_strata(deg)
    if variant == "tol_pm1":
        # widen each stratum by merging with its neighbours
        s = strat.astype(int)
        strat = pd.Series(np.clip(s // 2, 0, None), index=strat.index)
    elif variant == "degree_plus_size":
        if MW is None:
            raise RuntimeError("molecular weight column unavailable")
        mw = MW.reindex(deg.index)
        size_cls = pd.qcut(mw.rank(method="first"), 3, labels=False)
        strat = strat.astype(str) + "_" + size_cls.astype(str)
    elif variant == "uniform":
        strat = pd.Series("all", index=deg.index)
    else:
        raise ValueError(variant)
    by = {}
    for m, s in strat.items():
        by.setdefault(s, []).append(m)
    ann = pos.groupby("disease").hmdb_id.apply(set).to_dict()
    rows = []
    for d, g in pos.groupby("disease"):
        taken = set()
        for m in g.hmdb_id:
            pool = [x for x in by.get(strat.get(m, "all"), [])
                    if x not in ann[d] and x not in taken]
            if not pool:
                pool = [x for x in deg.index if x not in ann[d] and x not in taken]
            if not pool:
                continue
            neg = pool[int(rng.integers(len(pool)))]
            taken.add(neg)
            rows.append((m, d, 1))
            rows.append((neg, d, 0))
    return pd.DataFrame(rows, columns=["hmdb_id", "disease", "label"])


def run_sampler():
    variants = ["exact_1to1", "tol_pm1", "ratio_2", "ratio_3",
                "degree_plus_size", "uniform"]
    out = []
    t0 = time.time()
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        for v in variants:
            P = sample_variant(POS[["hmdb_id", "disease"]], rng, v).reset_index(drop=True)
            for regime in ["lmo", "ldo"]:
                d = cv_auc(P, regime, seed)
                d["variant"], d["regime"], d["seed"] = v, regime, seed
                d["n_pairs"] = len(P)
                out.append(d)
                print(f"[{time.time()-t0:6.0f}s] seed{seed} {v:17s} {regime} "
                      f"mean_auc={d.auc.mean():.3f} n={len(P)}", flush=True)
    pd.concat(out, ignore_index=True).to_csv("results/sampler_grid.csv", index=False)
    print("wrote results/sampler_grid.csv", flush=True)


# ----------------------------------------------------------------- temporal
def run_temporal():
    AY = pd.read_csv("rebuild/assoc_years.csv")
    P0 = POS[["hmdb_id", "disease"]].merge(AY, on=["hmdb_id", "disease"], how="left")
    P0 = P0.dropna(subset=["year_min"])
    rows = []
    for cut in [2005, 2008, 2010, 2012]:
        tr_pos = P0[P0.year_min < cut]
        te_pos = P0[P0.year_min >= cut]
        if len(te_pos) < 50:
            continue
        seen = set(tr_pos.disease)
        for seed in range(5):
            rng = np.random.default_rng(400 + seed)
            Ptr = degree_matched_pairs(tr_pos[["hmdb_id", "disease"]], rng)
            Pte = degree_matched_pairs(te_pos[["hmdb_id", "disease"]], rng)
            # a test pair whose metabolite was in training is not a new
            # discovery for that metabolite; record the split both ways
            tr_mets = set(Ptr[Ptr.label == 1].hmdb_id)
            p = fit_predict(MODEL, seed, X_of(Ptr), Ptr.label.values, X_of(Pte))
            Pte = Pte.assign(p=p,
                             dis_seen=Pte.disease.isin(seen),
                             met_seen=Pte.hmdb_id.isin(tr_mets))
            for name, sub in [("all", Pte),
                              ("metabolite_unseen", Pte[~Pte.met_seen])]:
                if sub.label.nunique() < 2 or len(sub) < 30:
                    continue
                # per-disease AUROC, the unit of analysis used throughout,
                # alongside the pooled figure
                pd_auc = [roc_auc_score(g.label, g.p) for _, g in
                          sub.groupby("disease") if g.label.nunique() > 1]
                rows.append(dict(cutoff=cut, seed=seed, subset=name,
                                 n=len(sub), n_pos=int(sub.label.sum()),
                                 n_dis=sub.disease.nunique(),
                                 auc_pooled=roc_auc_score(sub.label, sub.p),
                                 auc=float(np.mean(pd_auc)),
                                 n_dis_scored=len(pd_auc)))
            print(f"cut{cut} seed{seed} n_train={len(Ptr)} n_test={len(Pte)} "
                  f"per_disease_auc={rows[0 if not rows else -2]['auc']:.3f}",
                  flush=True)
    pd.DataFrame(rows).to_csv("results/temporal_holdout.csv", index=False)
    print("wrote results/temporal_holdout.csv", flush=True)


# -------------------------------------------------------------------- seeds
def run_seeds():
    rows = []
    t0 = time.time()
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        d = cv_auc(P, "ldo", seed)
        d["seed"] = seed
        rows.append(d)
        print(f"[{time.time()-t0:6.0f}s] seed{seed} ldo mean_auc={d.auc.mean():.4f}",
              flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv("results/ldo_seeds20.csv", index=False)

    # cumulative TOST against chance as seeds accumulate
    BOUND = 0.05
    stab = []
    for k in range(3, 21):
        v = D[D.seed < k].groupby("seed").auc.mean().values
        m, sd = v.mean(), v.std(ddof=1)
        se = sd / np.sqrt(len(v))
        t_lo = (m - (0.5 - BOUND)) / se
        t_hi = (m - (0.5 + BOUND)) / se
        p = max(stats.t.sf(t_lo, len(v) - 1), stats.t.cdf(t_hi, len(v) - 1))
        hw = stats.t.ppf(0.975, len(v) - 1) * se
        stab.append(dict(n_seeds=k, mean_auc=m, sd=sd, ci_halfwidth=hw, tost_p=p))
    pd.DataFrame(stab).to_csv("results/seed_stability.csv", index=False)
    print(pd.DataFrame(stab).round(4).to_string(index=False), flush=True)


# ------------------------------------------------------------- corroboration
def run_corrob():
    AY = pd.read_csv("rebuild/assoc_years.csv")
    P0 = POS[["hmdb_id", "disease"]].merge(AY, on=["hmdb_id", "disease"], how="left")
    P0["n_refs"] = P0.n_refs.fillna(0)
    rows = []
    # n_refs is recorded per disease block in the source release, so this
    # split separates sparsely-cited diseases from well-cited ones rather
    # than individual associations; recorded as such.
    for name, sub in [("single_reference", P0[P0.n_refs <= 1]),
                      ("multi_reference", P0[P0.n_refs >= 2])]:
        if len(sub) < 200:
            continue
        for seed in range(5):
            rng = np.random.default_rng(700 + seed)
            P = degree_matched_pairs(sub[["hmdb_id", "disease"]], rng).reset_index(drop=True)
            for regime in ["lmo", "ldo"]:
                d = cv_auc(P, regime, seed)
                d["subset"], d["regime"], d["seed"] = name, regime, seed
                d["n_pairs"] = len(P)
                rows.append(d)
                print(f"{name:17s} seed{seed} {regime} "
                      f"mean_auc={d.auc.mean():.3f} n={len(P)}", flush=True)
    pd.concat(rows, ignore_index=True).to_csv("results/corroboration.csv", index=False)
    print("wrote results/corroboration.csv", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("temporal", "all"):
        run_temporal()
    if what in ("corrob", "all"):
        run_corrob()
    if what in ("sampler", "all"):
        run_sampler()
    if what in ("seeds", "all"):
        run_seeds()
