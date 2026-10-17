"""Analogue audit: does performance depend on similarity to training compounds?

A scaffold split is not informative for this dataset because most modeled
metabolites are acyclic and share the empty Bemis-Murcko scaffold. Instead,
each test pair is assigned the maximum Tanimoto similarity between its
metabolite and the training metabolites associated with the same disease, and
AUROC is reported within quartiles of that similarity. The similarity used
alone as a score is also evaluated.

Writes results/analog_audit_raw.parquet, results/analog_strata.csv and
results/analog_nnsim_only.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
import run_definitive as RD
from degree_matched import degree_matched_pairs
from run_specificity import DISEASES, POS, FPX, FPI, SEEDS, N_FOLDS

MODEL = "rf"
FPB = (FPX[:, :1024] > 0)          # binary ECFP4 block for Tanimoto


def tanimoto_to_set(qi, ref_idx):
    """Max Tanimoto from query row qi to any row in ref_idx."""
    if len(ref_idx) == 0:
        return np.nan
    q = FPB[qi]
    R = FPB[ref_idx]
    inter = (R & q).sum(1)
    union = (R | q).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nanmax(np.where(union > 0, inter / union, 0.0))


def main():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        X = RD.build_X(P.hmdb_id.values, P.disease.values, "both")
        y = P.label.values
        oof = np.full(len(P), np.nan)
        nnsim = np.full(len(P), np.nan)
        for tr, te in evalproto.metabolite_grouped_folds(P, N_FOLDS, seed):
            oof[te] = RD.fit_predict(MODEL, seed, X[tr], y[tr], X[te])
            # per test row: similarity to the POSITIVE training metabolites of
            # that row's disease (the compounds that define the disease's
            # chemical neighborhood in training)
            trdf = P.loc[tr]
            pos_by_d = {d: np.array([FPI[m] for m in g.hmdb_id])
                        for d, g in trdf[trdf.label == 1].groupby("disease")}
            for i in te:
                ref = pos_by_d.get(P.disease.iloc[i], np.array([], dtype=int))
                nnsim[i] = tanimoto_to_set(FPI[P.hmdb_id.iloc[i]], ref)
        out = P.copy()
        out["p"], out["nnsim"], out["seed"] = oof, nnsim, seed
        rows.append(out)
        print(f"  seed={seed} done | median nnsim={np.nanmedian(nnsim):.3f}", flush=True)
    A = pd.concat(rows, ignore_index=True).dropna(subset=["p", "nnsim"])
    A.to_parquet("results/analog_audit_raw.parquet")

    # strata by nearest-neighbor similarity, pooled across seeds
    qs = A.nnsim.quantile([0, .25, .5, .75, 1.0]).values
    A["stratum"] = pd.cut(A.nnsim, bins=np.unique(qs), include_lowest=True)
    srows = []
    for (s, sd), g in A.groupby(["stratum", "seed"], observed=True):
        if g.label.nunique() < 2:
            continue
        srows.append(dict(stratum=str(s), seed=sd, n=len(g),
                          nnsim_mid=g.nnsim.median(),
                          auc=roc_auc_score(g.label, g.p)))
    S = pd.DataFrame(srows)
    S.to_csv("results/analog_strata.csv", index=False)
    print(S.groupby("stratum")[["nnsim_mid", "auc", "n"]].mean().round(4).to_string())

    # is similarity itself predictive of the label? (it must not be, or the
    # stratification is confounded with the label)
    au = []
    for d, g in A[A.seed == 0].groupby("disease"):
        if g.label.nunique() > 1:
            au.append(roc_auc_score(g.label, g.nnsim))
    print(f"\nnnsim-only AUROC (per disease, mean): {np.mean(au):.4f}  "
          f"[0.5 = similarity carries no label information]")
    pd.DataFrame(dict(auc=au)).to_csv("results/analog_nnsim_only.csv", index=False)


if __name__ == "__main__":
    main()
