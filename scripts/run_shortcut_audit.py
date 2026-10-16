"""Shortcuts available in the held-out-disease (extrapolation) regime.

For every seed of the primary protocol this rebuilds the degree-matched pair
table and the disease-grouped folds exactly as run_definitive.py does, and
scores each held-out pair with two label-free-at-test-time quantities that a
learner can recover from the TRAINING folds alone:

  label_rate   fraction of the metabolite's training rows that are positive
               (what a model memorizing the metabolite block converges to)
  train_degree number of training-fold positive edges of the metabolite
               (what a graph learner sees through the training adjacency)

Per-disease AUROC of each quantity is averaged over the 71 diseases.
Writes results/shortcut_audit.csv.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import evalproto
from degree_matched import degree_matched_pairs

SEEDS = [0, 1, 2, 3, 4]
POS = pd.read_csv("rebuild/model_positives.csv")[["hmdb_id", "disease"]].drop_duplicates()
SIB = json.load(open("rebuild/sibling_groups.json"))
DISEASES = sorted(POS.disease.unique())


def per_disease_auc(df, col):
    out = []
    for _, g in df.groupby("disease"):
        if g.label.nunique() == 2:
            out.append(roc_auc_score(g.label, g[col]))
    return np.mean(out)


rows = []
for seed in SEEDS:
    P = degree_matched_pairs(POS, np.random.default_rng(100 + seed)).reset_index(drop=True)
    for regime in ["ldo", "lmo"]:
        if regime == "ldo":
            folds = evalproto.disease_grouped_folds(P, DISEASES, 5, seed, sibling_of=SIB)
        else:
            folds = evalproto.metabolite_grouped_folds(P, 5, seed)
        T = []
        for tr, te in folds:
            train, test = P.iloc[tr], P.iloc[te].copy()
            rate = train.groupby("hmdb_id").label.mean()
            tdeg = train[train.label == 1].groupby("hmdb_id").size()
            test["label_rate"] = test.hmdb_id.map(rate).fillna(0.5)
            test["train_degree"] = test.hmdb_id.map(tdeg).fillna(0)
            test["has_train_rows"] = test.hmdb_id.isin(rate.index)
            T.append(test)
        T = pd.concat(T)
        rows.append(dict(seed=seed, regime=regime,
                         auc_label_rate=per_disease_auc(T, "label_rate"),
                         auc_train_degree=per_disease_auc(T, "train_degree"),
                         frac_test_rows_with_train_rows=T.has_train_rows.mean()))
R = pd.DataFrame(rows)
R.to_csv("results/shortcut_audit.csv", index=False)
print(R.round(4).to_string(index=False))
print(R.groupby("regime")[["auc_label_rate", "auc_train_degree"]].mean().round(4))
