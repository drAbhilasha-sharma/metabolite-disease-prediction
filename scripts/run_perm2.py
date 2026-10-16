"""Within-disease label permutation: the exact null for this design.

WHY A SECOND PERMUTATION CONTROL
--------------------------------
run_definitive.py permutes the DISEASE COLUMN. That destroys the
metabolite-disease link, but it also destroys the degree matching (which was
constructed per disease) and it redistributes the "home-disease size" property
that we measured to be anti-correlated with the label (AUROC 0.331). The arm
therefore lands below chance for a reason unrelated to the biology, and cannot
be read as a chance reference.

Within-disease label permutation is the exact null. Because negatives are
degree-matched to positives WITHIN a disease, the two classes already share a
degree distribution; shuffling the label among a disease's rows therefore
preserves every marginal - class balance, degree distribution, home-disease
size, the feature matrix itself - and changes only WHICH metabolite is recorded
as truly associated. Any AUROC above 0.5 under this control would indicate
residual leakage; 0.5 confirms the design is clean.
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

SIBLING_OF = json.load(open("rebuild/sibling_groups.json"))


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
