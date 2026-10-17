"""Within-disease label permutation.

run_definitive.py permutes the disease column, which breaks the
metabolite-disease link but also the degree matching built within each disease.
This control instead shuffles the labels among the rows of each disease. Class
balance, the degree distribution, disease size and the feature matrix are
preserved, and only which metabolite is recorded as associated changes, so for
a design with degree-matched negatives it is the appropriate null.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
import run_definitive as RD
from degree_matched import degree_matched_pairs
from run_specificity import DISEASES, POS, SEEDS, N_FOLDS

GROUP_OF = json.load(open("rebuild/disease_groups.json"))


def permute_within_disease(P, rng):
    out = P.copy()
    lab = out.label.values.copy()
    for d, idx in out.groupby("disease").indices.items():
        lab[idx] = rng.permutation(lab[idx])
    out["label"] = lab
    return out


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        Pp = permute_within_disease(P, np.random.default_rng(900 + seed))
        X = RD.build_X(P.hmdb_id.values, P.disease.values, "both")
        y = Pp.label.values
        for regime in ["lmo", "ldo"]:
            fl = list(RD.folds_for(regime, P, seed))
            for model in ["logreg", "rf", "xgb", "mlp"]:
                oof = np.full(len(P), np.nan)
                for tr, te in fl:
                    oof[te] = RD.fit_predict(model, seed, X[tr], y[tr], X[te])
                m = ~np.isnan(oof)
                pdf = RD.per_disease_pooled(Pp.loc[m], oof[m])
                pdf["regime"], pdf["mode"] = regime, "perm_within"
                pdf["model"], pdf["seed"] = model, seed
                rows.append(pdf)
                print(f"  {regime}/{model} seed={seed} auc={pdf.auc.mean():.4f}", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv("results/perm_within.csv", index=False)
    print(R.groupby(["regime", "model"])[["auc", "ap"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
