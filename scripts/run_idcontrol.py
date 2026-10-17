"""Randomized-representation controls for the disease block.

The disease block is replaced by vectors that carry no gene-association
content:

  rand_id   a fixed random Gaussian vector per disease, of the same
            dimensionality as the gene-association profile
  onehot    a one-hot indicator per disease (71 dimensions)

Comparing these arms with the real profile, and with another disease's real
profile (run_definitive.py ablate), shows how much of the interpolation result
depends on the content of the disease block and how much on its distinguishing
the diseases.
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
