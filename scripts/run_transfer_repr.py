"""Does a similarity-bearing disease representation transfer to unseen diseases?

The paper's central claim is mechanistic: the disease block behaves as an
arbitrary identifier, so nothing carries across the boundary to a disease the
model has never seen. That claim makes a falsifiable prediction -- swap in a
representation that is similarity-bearing by construction, and extrapolation
should improve if the mechanism is representational rather than fundamental.

This module tests that prediction. The pipeline, folds, sampler and learner
are held identical to run_definitive.py; only the disease block is exchanged:

  gene_expression   the representation used throughout the paper (baseline)
  mesh_tree         multi-hot over MeSH ancestor prefixes (hierarchy-derived)
  mesh_text         SVD of TF-IDF over the descriptor scope note
  tree_plus_text    concatenation of the two ontology representations
  random_gauss      dimension-matched Gaussian noise (identifier control)

The random control is dimension-matched to the ontology representation rather
than to the gene-expression block, which closes the obvious objection that any
difference is a width effect rather than a content effect.

Writes results/transfer_repr.csv (per-disease AUROC per regime x repr x seed).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from degree_matched import degree_matched_pairs
from run_definitive import fit_predict, per_disease_pooled
from run_specificity import DISEASES, POS, FPX, FPI, DFv, DFI, SEEDS, N_FOLDS

SIBLING_OF = json.load(open("rebuild/sibling_groups.json"))
MODEL = "rf"
os.makedirs("results", exist_ok=True)

TREE = np.load("rebuild/disease_mesh_tree.npy")
TEXT = np.load("rebuild/disease_mesh_text.npy")


def repr_matrix(name, seed):
    if name == "gene_expression":
        return DFv
    if name == "mesh_tree":
        return TREE
    if name == "mesh_text":
        return TEXT
    if name == "tree_plus_text":
        return np.hstack([TREE, TEXT])
    if name == "random_gauss":
        # dimension-matched to the ontology block, not to the gene block
        rng = np.random.default_rng(9000 + seed)
        return rng.normal(size=(len(DFI), TREE.shape[1] + TEXT.shape[1])
                          ).astype(np.float32)
    raise ValueError(name)


def folds_for(regime, P, seed):
    if regime == "lmo":
        return evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)
    return evalproto.disease_grouped_folds(P, DISEASES, N_FOLDS, seed,
                                           sibling_of=SIBLING_OF)


def main():
    reprs = ["gene_expression", "mesh_tree", "mesh_text",
             "tree_plus_text", "random_gauss"]
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        mi = np.array([FPI[m] for m in P.hmdb_id])
        di = np.array([DFI[d] for d in P.disease])
        y = P.label.values
        for regime in ["lmo", "ldo"]:
            fl = list(folds_for(regime, P, seed))
            for rname in reprs:
                R = repr_matrix(rname, seed)
                X = np.hstack([FPX[mi], R[di]])
                pred = np.zeros(len(P))
                for tr, te in fl:
                    pred[te] = fit_predict(MODEL, seed, X[tr], y[tr], X[te])
                d = per_disease_pooled(P, pred)
                d["regime"], d["repr"], d["seed"] = regime, rname, seed
                rows.append(d)
                print(f"[{time.time()-t0:6.0f}s] seed{seed} {regime} "
                      f"{rname:16s} mean_auc={d.auc.mean():.3f}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv("results/transfer_repr.csv", index=False)
    print("wrote results/transfer_repr.csv", out.shape, flush=True)


if __name__ == "__main__":
    main()
