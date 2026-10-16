"""Disease representations designed to transfer across diseases.

The representations tested so far -- gene-expression profile, pathway
membership, one-hot identifier -- are all, by the paper's own ablation result,
functionally interchangeable with an arbitrary identifier: each distinguishes
diseases from one another but none places biologically related diseases near
each other in a way a learner could exploit for a disease it has never seen.

This module builds two representations that are constructed to have exactly
that property, from the MeSH descriptor release already on disk:

  mesh_tree   Each MeSH descriptor occupies one or more positions in a
              hierarchical tree (e.g. C18.452.394.750 for a metabolic
              disorder). A disease is represented by the multi-hot vector of
              every ancestor prefix of every tree position it occupies.
              Two diseases sharing a parent therefore share coordinates by
              construction -- the representation is similarity-bearing before
              any model sees a label.

  mesh_text   The descriptor's scope note (its free-text definition) and
              entry terms, embedded as an L2-normalised TF-IDF vector reduced
              by truncated SVD. Diseases described in similar language land
              near each other. This is the bag-of-words analogue of the
              language-model embedding a reviewer would ask for, built from
              the release itself so it needs no external service.

Both are written for the 71 modelled diseases in the same row order as
rebuild/disease_features.csv, so they drop into the existing pipeline as
alternative disease blocks.
"""
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

SRC = "raw/data_raw/mesh/MeSH_desc2026.xml"
N_SVD = 48


def parse_mesh(wanted):
    """Pull tree numbers, scope note and entry terms for the wanted UIs."""
    rec = {}
    ctx = ET.iterparse(SRC, events=("end",))
    for _, el in ctx:
        if el.tag != "DescriptorRecord":
            continue
        ui = el.findtext("DescriptorUI")
        if ui in wanted:
            trees = [t.text for t in el.iter("TreeNumber") if t.text]
            scope = " ".join(s.text or "" for s in el.iter("ScopeNote"))
            terms = " ".join(s.text or "" for s in el.iter("String"))
            rec[ui] = dict(tree=trees, text=(scope + " " + terms).strip())
        el.clear()
    return rec


def ancestors(tree_no):
    """All prefixes of a dotted tree position, including itself."""
    parts = tree_no.split(".")
    return [".".join(parts[:i + 1]) for i in range(len(parts))]


def main():
    os.makedirs("rebuild", exist_ok=True)
    DF = pd.read_csv("rebuild/disease_features.csv", index_col=0)
    dis = DF.index.tolist()
    d2m = json.load(open("rebuild/disease2mesh.json"))
    ui = [d2m.get(d) for d in dis]
    n_mapped = sum(u is not None for u in ui)
    print(f"{n_mapped}/{len(dis)} diseases carry a MeSH descriptor", flush=True)

    rec = parse_mesh({u for u in ui if u})
    print(f"parsed {len(rec)} descriptor records", flush=True)

    # ---- tree representation: multi-hot over ancestor prefixes
    vocab = sorted({a for u in ui if u and u in rec
                    for t in rec[u]["tree"] for a in ancestors(t)})
    vi = {v: i for i, v in enumerate(vocab)}
    T = np.zeros((len(dis), len(vocab)), dtype=np.float32)
    for r, u in enumerate(ui):
        if u and u in rec:
            for t in rec[u]["tree"]:
                for a in ancestors(t):
                    T[r, vi[a]] = 1.0
    # depth-weight: deeper (more specific) coordinates carry less mass than
    # shallow ones, so shared high-level structure dominates the similarity
    w = np.array([1.0 / (1 + v.count(".")) for v in vocab], dtype=np.float32)
    T = T * w
    nrm = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.maximum(nrm, 1e-9)
    print(f"tree representation: {T.shape}, "
          f"{(T.sum(1) > 0).sum()} diseases non-empty", flush=True)

    # ---- text representation: TF-IDF over scope note + entry terms -> SVD
    docs = [rec[u]["text"] if (u and u in rec) else "" for u in ui]
    tv = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True)
    X = tv.fit_transform(docs)
    k = int(min(N_SVD, min(X.shape) - 1))
    E = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    print(f"text representation: {E.shape} from {X.shape[1]} terms", flush=True)

    np.save("rebuild/disease_mesh_tree.npy", T.astype(np.float32))
    np.save("rebuild/disease_mesh_text.npy", E.astype(np.float32))

    # ---- how similarity-bearing is each? report the off-diagonal structure
    def offdiag_stats(M, name):
        S = M @ M.T
        iu = np.triu_indices(len(M), 1)
        v = S[iu]
        return dict(repr=name, mean_cos=float(v.mean()), max_cos=float(v.max()),
                    frac_gt_05=float((v > 0.5).mean()),
                    frac_zero=float((v <= 1e-6).mean()))

    DFv = DF.values.astype(np.float32)
    DFn = DFv / np.maximum(np.linalg.norm(DFv, axis=1, keepdims=True), 1e-9)
    S = pd.DataFrame([offdiag_stats(DFn, "gene_expression"),
                      offdiag_stats(T, "mesh_tree"),
                      offdiag_stats(E, "mesh_text")])
    S.to_csv("results/repr_similarity_structure.csv", index=False)
    print(S.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
