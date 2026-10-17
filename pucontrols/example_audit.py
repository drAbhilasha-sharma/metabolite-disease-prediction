"""Worked example: the audit package applied to the metabolite-disease corpus.

Run from the repository root:

    python -m pucontrols.example_audit

The library's public interface, the same one a user would call on their own
edge list, is run in two configurations that differ only in the negative
sampler:

    uniform          negatives drawn uniformly from the annotated metabolites
    degree_matched   negatives matched on annotation degree

Writes results/toolkit_audit.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pucontrols import audit, degree_matched_negatives, sample_negatives


def main():
    POS = pd.read_csv("rebuild/model_positives.csv")
    FP = pd.read_csv("rebuild/descriptors_all.csv").set_index("hmdb_id")
    DF = pd.read_csv("rebuild/disease_features.csv", index_col=0)

    edges = (POS[["hmdb_id", "disease"]]
             .rename(columns={"hmdb_id": "source", "disease": "target"})
             .drop_duplicates())
    edges = edges[edges.source.isin(FP.index) & edges.target.isin(DF.index)]

    FPn = FP.reindex(sorted(set(edges.source)))
    FPn = FPn.fillna(FPn.median(numeric_only=True))
    n_src = FPn.shape[1]

    def featurize(pairs, train_idx):
        A = FPn.reindex(pairs.source).values.astype(np.float32)
        B = DF.reindex(pairs.target).values.astype(np.float32)
        return np.nan_to_num(np.hstack([A, B]))

    out = []
    for name, sampler in [("uniform", sample_negatives),
                          ("degree_matched", degree_matched_negatives)]:
        print(f"--- sampler: {name}", flush=True)
        rep = audit(edges, featurize, n_seeds=3, sampler=sampler,
                    n_source_cols=n_src, target_features=DF)
        print(rep.summary(), flush=True)
        d = rep.to_frame()
        d["sampler"] = name
        out.append(d)

    os.makedirs("results", exist_ok=True)
    pd.concat(out, ignore_index=True).to_csv("results/toolkit_audit.csv",
                                             index=False)
    print("wrote results/toolkit_audit.csv", flush=True)


if __name__ == "__main__":
    main()
