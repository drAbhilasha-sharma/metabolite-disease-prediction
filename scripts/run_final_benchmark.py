"""Matched-negative design: all learners, both regimes, with controls.

Negatives for disease d are sampled from metabolites annotated to some other
disease, never from the unannotated pool, so whether a metabolite is annotated
at all carries no information about the label. (Negatives from the unannotated
pool are evaluated in run_benchmark.py.)

Regimes:
  LMO  held-out metabolites; diseases seen in training
  LDO  held-out diseases; no training labels for them
Controls, run in the identical design:
  blind  disease feature block zeroed
  perm   disease identity permuted

Metrics: AUROC and average precision per disease, averaged over seeds within
disease before any test.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from run_specificity import (DISEASES, DFv, DFI, FPX, FPI, MDIS, DPOS,
                             feats, matched_pairs, SEEDS, N_FOLDS)

LEARNERS = {
    "logreg": lambda s: LogisticRegression(max_iter=3000, C=1.0, random_state=s),
    "rf":     lambda s: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                               n_jobs=-1, random_state=s),
    "mlp":    lambda s: MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400,
                                      early_stopping=True, random_state=s),
    "xgb":    lambda s: XGBClassifier(n_estimators=400, max_depth=6,
                                      learning_rate=0.08, subsample=0.9,
                                      colsample_bytree=0.6, n_jobs=-1,
                                      eval_metric="logloss", random_state=s),
}
NEEDS_SCALING = {"logreg", "mlp"}


def evaluate(regime, mode):
    """regime in {lmo, ldo}; mode in {full, blind, perm}."""
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(7000 + seed)
        P = matched_pairs(rng)
        if mode == "perm":
            P = P.copy()
            P["disease"] = rng.permutation(P["disease"].values)
        X = feats(P.hmdb_id.values, P.disease.values, blind=(mode == "blind"))
        y = P.label.values

        if regime == "lmo":
            folds = evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)
        else:
            folds = evalproto.disease_grouped_folds(P, DISEASES, N_FOLDS, seed)

        for k, (tr, te) in enumerate(folds):
            if len(tr) == 0 or len(te) == 0:
                continue
            Xtr, Xte = X[tr], X[te]
            Xtr_s, Xte_s = None, None
            for name, make in LEARNERS.items():
                if name in NEEDS_SCALING:
                    if Xtr_s is None:
                        sc = StandardScaler().fit(Xtr)
                        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
                    A, B = Xtr_s, Xte_s
                else:
                    A, B = Xtr, Xte
                m = make(seed).fit(A, y[tr])
                p = m.predict_proba(B)[:, 1]
                sub = P.iloc[te].copy()
                sub["y"], sub["p"] = y[te], p
                for d, g in sub.groupby("disease"):
                    if g.y.nunique() < 2:
                        continue
                    rows.append({
                        "regime": regime, "mode": mode, "model": name,
                        "seed": seed, "fold": k, "disease": d,
                        "n": len(g), "n_pos": int(g.y.sum()),
                        "auc": roc_auc_score(g.y, g.p),
                        "ap": average_precision_score(g.y, g.p),
                    })
            print(f"  {regime}/{mode} seed={seed} fold={k}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    jobs = [("lmo", "full"), ("lmo", "blind"), ("lmo", "perm"),
            ("ldo", "full"), ("ldo", "blind"), ("ldo", "perm")]
    out = pd.concat([evaluate(r, m) for r, m in jobs], ignore_index=True)
    out.to_csv("results/final_benchmark.csv", index=False)
    print(out.groupby(["regime", "mode", "model"])[["auc", "ap"]]
             .mean().round(4).to_string())
