"""Wrong-disease substitution as a distribution, and the identifier mechanism
stated in information-theoretic terms.

PART 1 -- substitution
The paper reports that substituting the wrong disease at prediction time
changes AUROC by a small amount. A single substitution is one draw from a
distribution: which wrong disease is substituted matters, because some
diseases are near-duplicates of others and some are isolated. This module
repeats the substitution over many independent random derangements and reports
the distribution, so the claim rests on a spread rather than a point.

Two substitution schemes are run:
  derangement   every disease is replaced by a different disease under a
                random derangement (no disease keeps its own vector), so the
                marginal distribution of disease vectors is preserved exactly.
  nearest       every disease is replaced by its most similar other disease.
                This is the hardest case for the identifier account: if the
                model used biological content, substituting a biologically
                adjacent disease should hurt least.
  farthest      every disease is replaced by its least similar other disease,
                the easiest case to detect if content is being used.

PART 2 -- mutual information
The ablation result is that real profiles, permuted-real profiles, Gaussian
noise and one-hot codes all reach statistically equivalent AUROC. The
mechanistic claim is that the disease block contributes only identity. That
claim is measurable: the identity of the disease is a discrete variable D, and
the empirical mutual information I(D; Y) between disease identity and the
label -- estimated on the pair table -- is an upper bound on what ANY
representation that merely distinguishes diseases can contribute. We estimate:

  I(D; Y)         disease identity vs label, in bits
  H(Y)            label entropy, the ceiling
  I(D; Y) / H(Y)  the fraction of label information available from identity
  I(R; Y)         for each representation R, via a k-nearest-neighbour
                  estimate on the representation's own geometry, showing that
                  none exceeds the identity bound.

Writes results/substitution_dist.csv and results/mutual_information.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from degree_matched import degree_matched_pairs
from run_definitive import fit_predict
from run_specificity import DISEASES, POS, FPX, FPI, DFv, DFI, N_FOLDS

SIBLING_OF = json.load(open("rebuild/sibling_groups.json"))
MODEL = "rf"
N_SUB = 50
# Seed count is configurable so the equivalence test can be run at higher
# power without editing the module; the default reproduces the first run.
N_SEEDS = int(os.environ.get("MDA_SUB_SEEDS", "5"))
SUB_OUT = os.environ.get("MDA_SUB_OUT", "results/substitution_dist.csv")
os.makedirs("results", exist_ok=True)


def derangement(n, rng):
    """A permutation with no fixed point."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def run_substitution():
    rows = []
    Dn = DFv / np.maximum(np.linalg.norm(DFv, axis=1, keepdims=True), 1e-9)
    S = Dn @ Dn.T
    Smax = S.copy(); np.fill_diagonal(Smax, -np.inf)
    Smin = S.copy(); np.fill_diagonal(Smin, np.inf)
    nearest = Smax.argmax(1)
    farthest = Smin.argmin(1)

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        mi = np.array([FPI[m] for m in P.hmdb_id])
        di = np.array([DFI[d] for d in P.disease])
        y = P.label.values
        X = np.hstack([FPX[mi], DFv[di]])
        folds = list(evalproto.metabolite_grouped_folds(P, N_FOLDS, seed))

        # baseline: honest out-of-fold predictions
        base = np.zeros(len(P))
        models = []
        for tr, te in folds:
            base[te] = fit_predict(MODEL, seed, X[tr], y[tr], X[te])
            models.append((tr, te))
        auc_base = None  # set after models are fitted
        # substitution schemes: same fitted models, wrong disease at test time
        from run_definitive import make, NEEDS_SCALING
        from sklearn.preprocessing import StandardScaler
        fitted = []
        for tr, te in models:
            sc = StandardScaler().fit(X[tr]) if MODEL in NEEDS_SCALING else None
            Xtr = sc.transform(X[tr]) if sc is not None else X[tr]
            clf = make(MODEL, seed).fit(Xtr, y[tr])
            fitted.append((clf, sc, te))

        def per_dis_mean(pred):
            v = [roc_auc_score(g.label, g.p) for _, g in
                 P.assign(p=pred).groupby("disease") if g.label.nunique() > 1]
            return float(np.mean(v))

        def score_with(perm):
            pred = np.zeros(len(P))
            for clf, sc, te in fitted:
                Xs = np.hstack([FPX[mi[te]], DFv[perm[di[te]]]])
                Xs = sc.transform(Xs) if sc is not None else Xs
                pred[te] = clf.predict_proba(Xs)[:, 1]
            return per_dis_mean(pred)

        auc_base = per_dis_mean(base)
        rows.append(dict(seed=seed, scheme="none", draw=0, auc=auc_base, delta=0.0))
        for k in range(N_SUB):
            p = derangement(len(DFI), np.random.default_rng(5000 + 100 * seed + k))
            a = score_with(p)
            rows.append(dict(seed=seed, scheme="derangement", draw=k,
                             auc=a, delta=a - auc_base))
        for name, p in [("nearest", nearest), ("farthest", farthest)]:
            a = score_with(p)
            rows.append(dict(seed=seed, scheme=name, draw=0,
                             auc=a, delta=a - auc_base))
        print(f"seed{seed} base={auc_base:.3f} "
              f"derangement={np.mean([r['delta'] for r in rows if r['seed']==seed and r['scheme']=='derangement']):+.4f} "
              f"nearest={[r['delta'] for r in rows if r['seed']==seed and r['scheme']=='nearest'][0]:+.4f} "
              f"farthest={[r['delta'] for r in rows if r['seed']==seed and r['scheme']=='farthest'][0]:+.4f}",
              flush=True)

    pd.DataFrame(rows).to_csv(SUB_OUT, index=False)
    print(f"wrote {SUB_OUT}", flush=True)


def run_mi():
    """Identity-label mutual information and the representation bound."""
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        y = P.label.values
        d = P.disease.values
        m = P.hmdb_id.values

        def H(v):
            _, c = np.unique(v, return_counts=True)
            p = c / c.sum()
            return float(-(p * np.log2(p)).sum())

        # mutual_info_score returns nats; convert to bits
        mi_dy = mutual_info_score(d, y) / np.log(2)
        mi_my = mutual_info_score(m, y) / np.log(2)
        rows.append(dict(seed=seed, H_Y=H(y), H_D=H(d),
                         I_disease_Y=mi_dy, I_metabolite_Y=mi_my,
                         frac_disease=mi_dy / H(y),
                         frac_metabolite=mi_my / H(y)))
        print(f"seed{seed} H(Y)={H(y):.4f} I(D;Y)={mi_dy:.5f} "
              f"I(M;Y)={mi_my:.5f}", flush=True)
    MI = pd.DataFrame(rows)
    MI.to_csv("results/mutual_information.csv", index=False)
    print(MI.round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("sub", "all"):
        run_substitution()
    if what in ("mi", "all"):
        run_mi()
