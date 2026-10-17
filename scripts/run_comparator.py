"""The four diagnostic controls applied to gene-disease prediction.

Gene-disease associations with direct curated evidence from the CTD extract
(the same extract from which the disease profiles are built) form a second
bipartite association task with a different node type.

Features: Gaussian interaction profile (GIP) kernels, the representation this
literature uses when intrinsic node features are unavailable. Profiles are
computed from training edges only; test edges are removed before the kernel is
built, so no feature contains its own label.

Arms:
  full     the conventional pipeline
  perm     disease labels permuted across pairs
  blind    disease block removed
  degree   annotation degree alone

each under random pair-level K-fold and under K-fold grouped by node (whole
genes held out). The identical construction is also run on the metabolite
corpus (whole metabolites held out in the grouped split) for reference.

Writes results/comparator_controls.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_specificity import POS

N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
MIN_DEG = 10          # keep diseases with at least this many partners
MAX_DIS = 71          # match the primary corpus's disease count exactly
os.makedirs("results", exist_ok=True)


def load_ctd():
    df = pd.read_csv("rebuild/ctd_direct.tsv", sep="\t")
    df = df.rename(columns={df.columns[0]: "gene", "DiseaseID": "disease"})
    df = df[["gene", "disease"]].dropna().drop_duplicates()
    keep = df.disease.value_counts()
    df = df[df.disease.isin(keep[keep >= MIN_DEG].index)]
    # match the primary corpus in disease count so the two diagnostics are
    # run at comparable scale rather than differing by an order of magnitude
    top = df.disease.value_counts().index[:MAX_DIS]
    df = df[df.disease.isin(top)]
    keep_g = df.gene.value_counts()
    df = df[df.gene.isin(keep_g[keep_g >= 2].index)]
    return df.rename(columns={"gene": "node"}).reset_index(drop=True)


def load_primary():
    df = POS[["hmdb_id", "disease"]].drop_duplicates()
    return df.rename(columns={"hmdb_id": "node"}).reset_index(drop=True)


def sample_pairs(assoc, rng):
    """Uniform unlabeled negatives -- the conventional protocol's sampler."""
    nodes = assoc.node.unique()
    ann = assoc.groupby("disease").node.apply(set).to_dict()
    rows = []
    for d, g in assoc.groupby("disease"):
        banned = ann[d]
        pool = np.array([n for n in nodes if n not in banned])
        if len(pool) == 0:
            continue
        neg = rng.choice(pool, size=min(len(g), len(pool)), replace=False)
        for n in g.node:
            rows.append((n, d, 1))
        for n in neg:
            rows.append((n, d, 0))
    return pd.DataFrame(rows, columns=["node", "disease", "label"])


def gip_features(P, train_idx, node_idx, dis_idx, gamma_scale=1.0):
    """GIP-kernel profiles built from TRAINING POSITIVES ONLY.

    Returns the node-profile and disease-profile blocks for every pair in P.
    """
    n_n, n_d = len(node_idx), len(dis_idx)
    A = np.zeros((n_n, n_d), dtype=np.float32)
    tr = P.iloc[train_idx]
    pos = tr[tr.label == 1]
    A[[node_idx[x] for x in pos.node], [dis_idx[x] for x in pos.disease]] = 1.0

    def kernel(M):
        sq = (M * M).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2 * (M @ M.T)
        g = gamma_scale / max(sq.mean(), 1e-9)
        return np.exp(-g * np.maximum(d2, 0)).astype(np.float32)

    Kn = kernel(A)
    Kd = kernel(A.T)
    ni = np.array([node_idx[x] for x in P.node])
    di = np.array([dis_idx[x] for x in P.disease])
    return Kn[ni], Kd[di], A


def run_dataset(name, assoc):
    node_idx = {n: i for i, n in enumerate(sorted(assoc.node.unique()))}
    dis_idx = {d: i for i, d in enumerate(sorted(assoc.disease.unique()))}
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(200 + seed)
        P = sample_pairs(assoc, rng).reset_index(drop=True)
        y = P.label.values
        splits = {
            "random_pair": list(KFold(N_FOLDS, shuffle=True,
                                      random_state=seed).split(P)),
            "grouped_node": list(GroupKFold(N_FOLDS).split(P, y, groups=P.node)),
        }
        for sname, fl in splits.items():
            for arm in ["full", "perm", "blind", "degree"]:
                pred = np.zeros(len(P))
                for tr, te in fl:
                    Pm = P.copy()
                    if arm == "perm":
                        Pm.loc[:, "disease"] = rng.permutation(Pm.disease.values)
                    Kn, Kd, A = gip_features(Pm, tr, node_idx, dis_idx)
                    if arm == "blind":
                        X = Kn
                    elif arm == "degree":
                        deg_n = A.sum(1)[[node_idx[x] for x in Pm.node]]
                        deg_d = A.sum(0)[[dis_idx[x] for x in Pm.disease]]
                        X = np.column_stack([deg_n, deg_d])
                    else:
                        X = np.hstack([Kn, Kd])
                    clf = RandomForestClassifier(n_estimators=300,
                                                 min_samples_leaf=2, n_jobs=-1,
                                                 random_state=seed)
                    clf.fit(X[tr], y[tr])
                    pred[te] = clf.predict_proba(X[te])[:, 1]
                rows.append(dict(dataset=name, seed=seed, split=sname, arm=arm,
                                 n_pairs=len(P),
                                 n_nodes=len(node_idx), n_dis=len(dis_idx),
                                 auc=roc_auc_score(y, pred)))
                print(f"{name:14s} seed{seed} {sname:13s} {arm:7s} "
                      f"auc={rows[-1]['auc']:.4f}", flush=True)
    return rows


def main():
    ctd = load_ctd()
    prim = load_primary()
    print(f"comparator: {len(ctd)} associations, {ctd.node.nunique()} genes, "
          f"{ctd.disease.nunique()} diseases", flush=True)
    print(f"primary:    {len(prim)} associations, {prim.node.nunique()} "
          f"metabolites, {prim.disease.nunique()} diseases", flush=True)
    rows = run_dataset("gene_disease", ctd) + run_dataset("metabolite_disease", prim)
    D = pd.DataFrame(rows)
    D.to_csv("results/comparator_controls.csv", index=False)
    print(D.groupby(["dataset", "split", "arm"]).auc.agg(["mean", "std"])
          .round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
