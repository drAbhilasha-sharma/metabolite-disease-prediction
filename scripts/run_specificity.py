"""
Disease-specificity evaluation (fix for the one-class-detection artifact).

The positives-vs-unlabelled framing lets a model score well by learning
"is this metabolite disease-associated at all?" without using the disease.
Evidence: swapping the disease to a wrong one cost only 0.022 AUC, and 97% of
XGBoost's importance mass sat on metabolite features.

This module evaluates the question the paper actually claims to answer, using
two designs in which "annotated vs unannotated" carries NO signal:

  (A) WITHIN-METABOLITE RANKING
      Fix a metabolite that is annotated to >=1 disease. Rank ALL modelled
      diseases for it. Every candidate is an annotated metabolite, so the
      one-class shortcut is unavailable by construction. Metric: whether the
      true disease outranks the others (AUC over diseases, and MRR/hits@k).
      Random baseline = 0.5 AUC exactly.

  (B) MATCHED-NEGATIVE CLASSIFICATION
      For each disease, negatives are metabolites annotated to OTHER diseases
      (not unlabelled ones), matched to the positives. A model that only knows
      "this metabolite is disease-associated" scores exactly 0.5 here.

Both use metabolite-disjoint splits, seeds aggregated per disease before any
test (no pseudoreplication), and report the disease-blind control.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5

FPm = pd.read_csv("rebuild/fp_index.csv").hmdb_id.values
FPX = np.load("rebuild/fp_matrix.npy").astype(np.float32)
FPI = {m: i for i, m in enumerate(FPm)}
DF = pd.read_csv("rebuild/disease_features.csv", index_col=0)
POS = pd.read_csv("rebuild/model_positives.csv")
DISEASES = sorted(POS.disease.unique())
DFv = DF.loc[DISEASES].values.astype(np.float32)
DFI = {d: i for i, d in enumerate(DISEASES)}
NMET = FPX.shape[1]

# disease -> set of its positive metabolites
DPOS = {d: set(g.hmdb_id) for d, g in POS.groupby("disease")}
MDIS = POS.groupby("hmdb_id").disease.apply(set).to_dict()


def feats(mets, dis, blind=False):
    mi = np.array([FPI[m] for m in mets])
    Xm = FPX[mi]
    if blind:
        return Xm
    di = np.array([DFI[d] for d in dis])
    return np.hstack([Xm, DFv[di]])


def matched_pairs(rng):
    """Negatives = metabolites annotated to OTHER diseases, never unlabelled."""
    rows = []
    allm = np.array(sorted(MDIS))
    for d in DISEASES:
        pos = np.array(sorted(DPOS[d]))
        pool = np.array([m for m in allm if d not in MDIS[m]])
        n_neg = min(len(pos) * 5, len(pool))
        neg = rng.choice(pool, size=n_neg, replace=False)
        rows += [(d, m, 1) for m in pos] + [(d, m, 0) for m in neg]
    return pd.DataFrame(rows, columns=["disease", "hmdb_id", "label"])


def model(seed):
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.6, random_state=seed,
                         n_jobs=-1, eval_metric="logloss", tree_method="hist")


def run_matched(blind=False):
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(2000 + seed)
        P = matched_pairs(rng)
        X = feats(P.hmdb_id.values, P.disease.values, blind)
        y = P.label.values
        for k, (tr, te) in enumerate(evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)):
            m = model(seed).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            sub = P.iloc[te].copy()
            sub["y"] = y[te]
            sub["p"] = p
            for d, g in sub.groupby("disease"):
                if g.y.nunique() < 2:
                    continue
                rows.append({"design": "matched" + ("_blind" if blind else ""),
                             "seed": seed, "fold": k, "disease": d,
                             "n": len(g), "n_pos": int(g.y.sum()),
                             "auc": roc_auc_score(g.y, g.p)})
            print(f"  matched{'_blind' if blind else ''} seed={seed} fold={k}", flush=True)
    return pd.DataFrame(rows)


def run_within_metabolite():
    """Rank all diseases for a held-out metabolite. Chance AUC = 0.5."""
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(3000 + seed)
        P = matched_pairs(rng)
        X = feats(P.hmdb_id.values, P.disease.values)
        y = P.label.values
        for k, (tr, te) in enumerate(evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)):
            m = model(seed).fit(X[tr], y[tr])
            test_mets = sorted(set(P.iloc[te].hmdb_id) & set(MDIS))
            for met in test_mets:
                true_d = MDIS[met]
                if not true_d or len(true_d) >= len(DISEASES):
                    continue
                Xq = feats([met] * len(DISEASES), DISEASES)
                s = m.predict_proba(Xq)[:, 1]
                lab = np.array([1 if d in true_d else 0 for d in DISEASES])
                order = np.argsort(-s)
                ranks = np.where(lab[order] == 1)[0] + 1
                rows.append({"design": "within_metabolite", "seed": seed, "fold": k,
                             "hmdb_id": met, "n_true": int(lab.sum()),
                             "auc": roc_auc_score(lab, s),
                             "mrr": float(1.0 / ranks[0]),
                             "hit1": int(ranks[0] == 1),
                             "hit5": int(ranks[0] <= 5),
                             "hit10": int(ranks[0] <= 10)})
            print(f"  within seed={seed} fold={k} n={len(test_mets)}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = []
    if which in ("all", "matched"):
        out += [run_matched(False), run_matched(True)]
    if which in ("all", "within"):
        out.append(run_within_metabolite())
    res = pd.concat(out, ignore_index=True)
    res.to_csv(f"results/specificity_{which}.csv", index=False)
    print(res.groupby("design").auc.agg(["mean", "std", "count"]).round(4))
