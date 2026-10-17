"""Learning curves and label-noise sensitivity.

  1. Data volume: training pairs are subsampled to 10-100% and the pipeline is
     refit with the test folds held fixed.
  2. Label noise: a fraction of training labels is flipped before refitting;
     test labels are left intact.

Both run in each cross-validation regime under the degree-matched protocol.
Writes results/learning_curve.csv and results/label_noise.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from degree_matched import degree_matched_pairs
from run_definitive import build_X, fit_predict, folds_for
from run_specificity import POS, SEEDS

FRACTIONS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
NOISE = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
MODEL = "rf"
CURVE_SEEDS = SEEDS[:3]


def evaluate(P, X, y, folds, sub_frac=None, noise=None, seed=0):
    """Pooled out-of-fold AUROC per disease, with optional training subsample
    or training-label corruption. Test folds are never modified."""
    rng = np.random.default_rng(1000 + seed)
    oof = np.full(len(P), np.nan)
    for tr, te in folds:
        ytr = y[tr].copy()
        tr_use = tr
        if sub_frac is not None and sub_frac < 1.0:
            k = max(20, int(round(len(tr) * sub_frac)))
            pick = rng.choice(len(tr), size=k, replace=False)
            tr_use, ytr = tr[pick], ytr[pick]
        if noise:
            flip = rng.random(len(ytr)) < noise
            ytr = np.where(flip, 1 - ytr, ytr)
        if len(np.unique(ytr)) < 2:
            continue
        oof[te] = fit_predict(MODEL, seed, X[tr_use], ytr, X[te])
    m = ~np.isnan(oof)
    df = P.loc[m].assign(p=oof[m])
    out = []
    for d, g in df.groupby("disease"):
        if g.label.nunique() > 1:
            out.append(roc_auc_score(g.label, g.p))
    return np.array(out)


def main():
    rows_c, rows_n = [], []
    t0 = time.time()
    for seed in CURVE_SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        X = build_X(P.hmdb_id.values, P.disease.values, "both")
        y = P.label.values
        # both regimes
        for regime in ("lmo", "ldo"):
            folds = list(folds_for(regime, P, seed))
            for f in FRACTIONS:
                a = evaluate(P, X, y, folds, sub_frac=f, seed=seed)
                rows_c.append(dict(seed=seed, regime=regime, fraction=f,
                                   n_train_frac=f, mean_auc=a.mean(),
                                   sd_auc=a.std(ddof=1), n_diseases=len(a)))
                print(f"  [{time.time()-t0:5.0f}s] curve {regime} seed={seed} "
                      f"frac={f:.2f} auc={a.mean():.4f}", flush=True)
            for nz in NOISE:
                a = evaluate(P, X, y, folds, noise=nz, seed=seed)
                rows_n.append(dict(seed=seed, regime=regime, noise=nz,
                                   mean_auc=a.mean(), sd_auc=a.std(ddof=1),
                                   n_diseases=len(a)))
                print(f"  [{time.time()-t0:5.0f}s] noise {regime} seed={seed} "
                      f"p={nz:.2f} auc={a.mean():.4f}", flush=True)
    pd.DataFrame(rows_c).to_csv("results/learning_curve.csv", index=False)
    pd.DataFrame(rows_n).to_csv("results/label_noise.csv", index=False)


if __name__ == "__main__":
    main()
