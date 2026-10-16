"""Random-identifier control: does the disease block need to be biological at all?

The shuffled-profile ablation showed that giving each disease a real but WRONG
gene profile costs almost nothing. That is consistent with the disease block
functioning as an arbitrary identifier rather than as biology, but a real
profile still has realistic statistics (sparsity, correlation structure), so
the test is not airtight.

This control replaces the disease block with vectors that carry no biology
whatsoever:

  rand_id   a fixed random Gaussian vector per disease, same dimensionality
  onehot    a one-hot indicator per disease (a pure identifier, 71 dims)

If either matches the real-profile model, the conclusion is that the model
learns a per-disease set of structures and the biological content of the
disease representation is not used.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_definitive as RD
from degree_matched import degree_matched_pairs
from run_specificity import DISEASES, POS, SEEDS

FPI = RD.FPI
FPX = RD.FPX


def build_with(dis_mat, mets, diss, dindex):
    Xf = FPX[[FPI[m] for m in mets]]
    Xd = dis_mat[[dindex[d] for d in diss]]
    return np.hstack([Xf, Xd]).astype(np.float32)


def main():
    dindex = {d: i for i, d in enumerate(sorted(DISEASES))}
    nd = len(dindex)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        fl = list(RD.folds_for("lmo", P, seed))
        y = P.label.values
        gen = np.random.default_rng(700 + seed)
        arms = {
            "rand_id": gen.standard_normal((nd, RD.DFv.shape[1])).astype(np.float32),
            "onehot": np.eye(nd, dtype=np.float32),
        }
        for arm, M in arms.items():
            X = build_with(M, P.hmdb_id.values, P.disease.values, dindex)
            oof = np.full(len(P), np.nan)
            for tr, te in fl:
                oof[te] = RD.fit_predict("rf", seed, X[tr], y[tr], X[te])
            m = ~np.isnan(oof)
            pdf = RD.per_disease_pooled(P.loc[m], oof[m])
            pdf["arm"], pdf["seed"] = arm, seed
            rows.append(pdf)
            print(f"  {arm} seed={seed} auc={pdf.auc.mean():.4f}", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv("results/id_control.csv", index=False)
    print(R.groupby("arm")[["auc", "ap"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
