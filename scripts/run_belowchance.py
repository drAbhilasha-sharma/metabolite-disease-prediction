"""Label-shuffle diagnostic for control arms that score below chance.

Because the degree-matched sampler balances every disease at 50% positives,
disease identity carries no marginal information about the label, and a
signal-free model can only fit sampling fluctuations of its training folds.
Under grouped splitting the held-out fluctuation is independent of the fitted
one, which can push out-of-fold AUROC below 0.5.

The disease-only and structure-only arms are run under random,
metabolite-grouped and disease-grouped splits, each with intact training labels
and with training labels shuffled within each fold. If the explanation holds,
shuffling returns each below-chance arm to 0.5.

Writes results/belowchance.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from degree_matched import degree_matched_pairs
from run_definitive import fit_predict
from run_specificity import DISEASES, POS, FPX, FPI, DFv, DFI, N_FOLDS

GROUP_OF = json.load(open("rebuild/disease_groups.json"))
MODEL = "rf"
os.makedirs("results", exist_ok=True)


def main():
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        mi = np.array([FPI[m] for m in P.hmdb_id])
        di = np.array([DFI[d] for d in P.disease])
        y = P.label.values
        blocks = {"disease_only": DFv[di],
                  "structure_only": FPX[mi]}
        splits = {
            "random": list(KFold(N_FOLDS, shuffle=True,
                                 random_state=seed).split(P)),
            "grouped_metabolite": list(evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)),
            "grouped_disease": list(evalproto.disease_grouped_folds(
                P, DISEASES, N_FOLDS, seed, group_of=GROUP_OF)),
        }
        for bname, X in blocks.items():
            for sname, fl in splits.items():
                for shuffled in [False, True]:
                    pred = np.zeros(len(P))
                    for tr, te in fl:
                        ytr = y[tr]
                        if shuffled:
                            ytr = np.random.default_rng(
                                7000 + seed).permutation(ytr)
                        pred[te] = fit_predict(MODEL, seed, X[tr], ytr, X[te])
                    rows.append(dict(seed=seed, block=bname, split=sname,
                                     train_labels="shuffled" if shuffled else "intact",
                                     auc=roc_auc_score(y, pred)))
                    print(f"seed{seed} {bname:15s} {sname:19s} "
                          f"{'shuf' if shuffled else 'real'} "
                          f"auc={rows[-1]['auc']:.4f}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv("results/belowchance.csv", index=False)
    print(D.groupby(["block", "split", "train_labels"]).auc
          .agg(["mean", "std"]).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
