"""Graph learners under the degree-matched protocol.

Three graph-based learners are run on the same degree-matched pairs, folds and
regimes as the tabular learners of run_definitive.py:

  gcn        two-layer graph convolutional encoder over the bipartite
             metabolite-disease graph with a bilinear edge decoder, implemented
             in torch with a sparse normalized adjacency
  labelprop  personalized-PageRank label propagation on the metabolite
             projection of the training graph; no fitted parameters
  mf         matrix factorization of the training adjacency by truncated SVD

The graph is built from training pairs only; test edges are removed before
message passing. Under the held-out-disease regime a test disease therefore has
no training edges, and its embedding comes from its feature vector alone.

Writes results/graph_models.csv (per-disease AUROC, one row per
regime x mode x model x seed x disease).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from degree_matched import degree_matched_pairs
from run_specificity import DISEASES, POS, FPX, FPI, DFv, DFI, SEEDS, N_FOLDS

GROUP_OF = json.load(open("rebuild/disease_groups.json"))
os.makedirs("results", exist_ok=True)
torch.set_num_threads(4)

N_MET = len(FPI)
N_DIS = len(DFI)
EMB = 64
EPOCHS = 200
LR = 0.01


# ------------------------------------------------------------------ helpers
def folds_for(regime, P, seed):
    if regime == "lmo":
        return evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)
    return evalproto.disease_grouped_folds(P, DISEASES, N_FOLDS, seed,
                                           group_of=GROUP_OF)


def _sub_fp(mets):
    """Fingerprint rows for the metabolites appearing in this pair table."""
    return np.array([FPI[m] for m in mets])


def build_norm_adj(rows, cols, n_r, n_c):
    """Symmetrically normalized sparse adjacency of a bipartite graph.

    Returns a torch sparse tensor of shape (n_r + n_c, n_r + n_c) with the
    standard D^-1/2 (A + I) D^-1/2 normalization used by a GCN layer.
    """
    n = n_r + n_c
    r = np.concatenate([rows, cols + n_r, np.arange(n)])
    c = np.concatenate([cols + n_r, rows, np.arange(n)])
    deg = np.bincount(r, minlength=n).astype(np.float32)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    val = dinv[r] * dinv[c]
    idx = torch.tensor(np.vstack([r, c]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(val, dtype=torch.float32),
                                   (n, n)).coalesce()


class BipartiteGCN(nn.Module):
    """Two-layer GCN encoder over the bipartite graph + bilinear decoder."""

    def __init__(self, d_met, d_dis, emb=EMB):
        super().__init__()
        self.pm = nn.Linear(d_met, emb)
        self.pd = nn.Linear(d_dis, emb)
        self.w1 = nn.Linear(emb, emb)
        self.w2 = nn.Linear(emb, emb)
        self.dec = nn.Bilinear(emb, emb, 1)

    def encode(self, A, Xm, Xd):
        h = torch.cat([self.pm(Xm), self.pd(Xd)], 0)
        h = F.relu(self.w1(torch.sparse.mm(A, h)))
        h = self.w2(torch.sparse.mm(A, h))
        return h

    def forward(self, A, Xm, Xd, mi, di, n_met):
        h = self.encode(A, Xm, Xd)
        return self.dec(h[mi], h[n_met + di]).squeeze(-1)


def run_gcn(Ptr, Pte, Xm, Xd, seed, n_met_sub, met_map):
    """Train a GCN on training pairs only; score the held-out pairs."""
    torch.manual_seed(seed)
    pos = Ptr[Ptr.label == 1]
    A = build_norm_adj(np.array([met_map[m] for m in pos.hmdb_id]),
                       np.array([DFI[d] for d in pos.disease]),
                       n_met_sub, N_DIS)
    mdl = BipartiteGCN(Xm.shape[1], Xd.shape[1])
    opt = torch.optim.Adam(mdl.parameters(), lr=LR, weight_decay=1e-4)
    mi = torch.tensor([met_map[m] for m in Ptr.hmdb_id], dtype=torch.long)
    di = torch.tensor([DFI[d] for d in Ptr.disease], dtype=torch.long)
    y = torch.tensor(Ptr.label.values, dtype=torch.float32)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            mdl(A, Xm, Xd, mi, di, n_met_sub), y)
        loss.backward()
        opt.step()
    mdl.eval()
    with torch.no_grad():
        mi_te = torch.tensor([met_map[m] for m in Pte.hmdb_id], dtype=torch.long)
        di_te = torch.tensor([DFI[d] for d in Pte.disease], dtype=torch.long)
        s = torch.sigmoid(mdl(A, Xm, Xd, mi_te, di_te, n_met_sub))
    return s.numpy()


def run_labelprop(Ptr, Pte, met_map, n_met_sub, alpha=0.5, n_iter=20):
    """Personalized-PageRank propagation on the metabolite projection.

    Score(m, d) = mass arriving at m after diffusing from the metabolites
    already annotated to d, over the metabolite-metabolite graph induced by
    shared training diseases. No parameters are fitted.
    """
    pos = Ptr[Ptr.label == 1]
    mi = np.array([met_map[m] for m in pos.hmdb_id])
    di = np.array([DFI[d] for d in pos.disease])
    B = np.zeros((n_met_sub, N_DIS), dtype=np.float32)
    B[mi, di] = 1.0
    S = B @ B.T                     # metabolites sharing >=1 training disease
    np.fill_diagonal(S, 0.0)
    rs = S.sum(1, keepdims=True)
    S = S / np.maximum(rs, 1e-9)
    Y = B.copy()
    Z = B.copy()
    for _ in range(n_iter):
        Z = alpha * (S @ Z) + (1 - alpha) * Y
    return np.array([Z[met_map[m], DFI[d]] for m, d in
                     zip(Pte.hmdb_id, Pte.disease)])


def run_mf(Ptr, Pte, met_map, n_met_sub, seed, k=32):
    """Truncated-SVD matrix factorization of the training adjacency."""
    pos = Ptr[Ptr.label == 1]
    B = np.zeros((n_met_sub, N_DIS), dtype=np.float32)
    B[[met_map[m] for m in pos.hmdb_id], [DFI[d] for d in pos.disease]] = 1.0
    k = int(min(k, min(B.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    U = svd.fit_transform(B)
    V = svd.components_.T
    R = U @ V.T
    return np.array([R[met_map[m], DFI[d]] for m, d in
                     zip(Pte.hmdb_id, Pte.disease)])


def per_disease(P, pred):
    out = []
    df = P.assign(p=pred)
    for d, g in df.groupby("disease"):
        if g.label.nunique() < 2:
            continue
        out.append(dict(disease=d, n=len(g), n_pos=int(g.label.sum()),
                        auc=roc_auc_score(g.label, g.p),
                        ap=average_precision_score(g.label, g.p)))
    return pd.DataFrame(out)


def main():
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        mets = sorted(P.hmdb_id.unique())
        met_map = {m: i for i, m in enumerate(mets)}
        n_met_sub = len(mets)
        Xm = torch.tensor(FPX[_sub_fp(mets)], dtype=torch.float32)
        Xd = torch.tensor(DFv, dtype=torch.float32)
        for regime in ["lmo", "ldo"]:
            fl = list(folds_for(regime, P, seed))
            for mode in ["full", "perm"]:
                Pm = P.copy()
                if mode == "perm":
                    Pm["disease"] = rng.permutation(Pm.disease.values)
                for model in ["gcn", "labelprop", "mf"]:
                    pred = np.zeros(len(Pm))
                    for tr, te in fl:
                        Ptr, Pte = Pm.iloc[tr], Pm.iloc[te]
                        if model == "gcn":
                            p = run_gcn(Ptr, Pte, Xm, Xd, seed, n_met_sub, met_map)
                        elif model == "labelprop":
                            p = run_labelprop(Ptr, Pte, met_map, n_met_sub)
                        else:
                            p = run_mf(Ptr, Pte, met_map, n_met_sub, seed)
                        pred[te] = p
                    pd_df = per_disease(Pm, pred)
                    pd_df["regime"], pd_df["mode"] = regime, mode
                    pd_df["model"], pd_df["seed"] = model, seed
                    rows.append(pd_df)
                    print(f"[{time.time()-t0:6.0f}s] seed{seed} {regime} {mode} "
                          f"{model:10s} mean_auc={pd_df.auc.mean():.3f} "
                          f"n_dis={len(pd_df)}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv("results/graph_models.csv", index=False)
    print("wrote results/graph_models.csv", out.shape, flush=True)


if __name__ == "__main__":
    main()
