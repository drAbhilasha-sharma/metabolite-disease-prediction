"""Definitive evaluation: pooled out-of-fold predictions, full ablation grid.

WHY THIS SUPERSEDES THE FOLD-AVERAGED RUN
-----------------------------------------
run_degmatched_benchmark.py computed one AUROC per (disease, fold) and averaged
them. Measured on its output: 37.5% of those fold-level AUROCs rest on fewer
than 10 test points and 12.8% are degenerate (exactly 0.0 or 1.0). A mean over
degenerate estimates is not a defensible point estimate.

Fix: within a (regime, mode, model, seed) cell, concatenate the out-of-fold
predictions across all folds, THEN compute one AUROC per disease on the pooled
vector. Every test point is still predicted by a model that never saw it, so
the estimate is honest, but each disease now contributes a single estimate
computed on all of its examples rather than five noisy fifths.

Design carried over (see logs/19_shortcut_audit.md):
  * negatives degree-matched to positives      -> annotation-breadth confound closed
  * negatives drawn from OTHER diseases        -> annotated/unannotated shortcut closed
  * MeSH-sibling groups kept within one fold   -> identical-representation leakage closed

SECTIONS (argv[1])
  main       primary grid: 2 regimes x 3 modes x 4 learners x 5 seeds
  ablate     feature-block and representation ablations (best learner, LMO)
  all        both
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
from run_specificity import DISEASES, POS, FPX, FPI, DFv, DFI, SEEDS, N_FOLDS

SIBLING_OF = json.load(open("rebuild/sibling_groups.json"))
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------- learners
def make(model, seed):
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                      n_jobs=-1, random_state=seed)
    if model == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", n_jobs=-1,
                             random_state=seed, verbosity=0)
    if model == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                             early_stopping=True, random_state=seed)
    raise ValueError(model)

NEEDS_SCALING = {"logreg", "mlp"}


def fit_predict(model, seed, Xtr, ytr, Xte):
    if model in NEEDS_SCALING:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = make(model, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


# ------------------------------------------------------- feature assembly
def build_X(mets, dis, block="both", DFmat=None, FPmat=None):
    """block: both | fp_only (disease removed) | dis_only (structure removed)."""
    FPmat = FPX if FPmat is None else FPmat
    DFmat = DFv if DFmat is None else DFmat
    mi = np.array([FPI[m] for m in mets])
    di = np.array([DFI[d] for d in dis])
    if block == "fp_only":
        return FPmat[mi]
    if block == "dis_only":
        return DFmat[di]
    return np.hstack([FPmat[mi], DFmat[di]])


def per_disease_pooled(P, pred, metric_prefix=""):
    """One AUROC/AP per disease from POOLED out-of-fold predictions."""
    out = []
    df = P.assign(p=pred)
    for d, g in df.groupby("disease"):
        if g.label.nunique() < 2:
            continue
        out.append(dict(disease=d, n=len(g), n_pos=int(g.label.sum()),
                        auc=roc_auc_score(g.label, g.p),
                        ap=average_precision_score(g.label, g.p)))
    return pd.DataFrame(out)


def folds_for(regime, P, seed):
    if regime == "lmo":
        return evalproto.metabolite_grouped_folds(P, N_FOLDS, seed)
    return evalproto.disease_grouped_folds(P, DISEASES, N_FOLDS, seed,
                                           sibling_of=SIBLING_OF)


# ------------------------------------------------------------------- main
def run_main():
    rows, preds = [], []
    t0 = time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        for regime in ["lmo", "ldo"]:
            fl = list(folds_for(regime, P, seed))
            for mode in ["full", "perm", "blind"]:
                Pm = P.copy()
                if mode == "perm":
                    # permute the disease column within the pair table: destroys
                    # the metabolite-disease link, preserves both marginals
                    Pm["disease"] = rng.permutation(Pm["disease"].values)
                block = "fp_only" if mode == "blind" else "both"
                X = build_X(Pm.hmdb_id.values, Pm.disease.values, block)
                y = Pm.label.values
                for model in ["logreg", "rf", "xgb", "mlp"]:
                    oof = np.full(len(Pm), np.nan)
                    for tr, te in fl:
                        oof[te] = fit_predict(model, seed, X[tr], y[tr], X[te])
                    m = ~np.isnan(oof)
                    pdf = per_disease_pooled(P.loc[m], oof[m])
                    pdf["regime"], pdf["mode"] = regime, mode
                    pdf["model"], pdf["seed"] = model, seed
                    rows.append(pdf)
                    pr = P.loc[m, ["hmdb_id", "disease", "label"]].copy()
                    pr["p"] = oof[m]
                    pr["regime"], pr["mode"] = regime, mode
                    pr["model"], pr["seed"] = model, seed
                    preds.append(pr)
                    print(f"  [{time.time()-t0:6.0f}s] {regime}/{mode}/{model} "
                          f"seed={seed} auc={pdf.auc.mean():.4f}", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R.to_csv("results/definitive_per_disease.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet("results/definitive_oof.parquet")
    print(R.groupby(["regime", "mode", "model"])[["auc", "ap"]].mean().round(4).to_string())


# --------------------------------------------------------------- ablations
def run_ablate(model="rf"):
    """Feature-block and disease-representation ablations, LMO regime."""
    KEGG = pd.read_csv("rebuild/disease_kegg_features.csv", index_col=0)
    KEGGv = KEGG.reindex(DISEASES).fillna(0).values.astype(np.float32)
    rows = []
    t0 = time.time()
    arms = [
        ("both",        "both",     None,   "structure + disease genes"),
        ("fp_only",     "fp_only",  None,   "structure only"),
        ("dis_only",    "dis_only", None,   "disease only"),
        ("kegg",        "both",     KEGGv,  "structure + disease pathways"),
        ("shuffled_dis","both",     "shuf", "structure + shuffled disease vectors"),
    ]
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        fl = list(folds_for("lmo", P, seed))
        for arm, block, DFmat, label in arms:
            if isinstance(DFmat, str) and DFmat == "shuf":
                # shuffle rows of the disease matrix: each disease keeps a real
                # gene profile, but the WRONG one -> tests whether the specific
                # profile matters or merely its existence as an ID
                DFuse = DFv[rng.permutation(len(DFv))]
            else:
                DFuse = DFmat
            X = build_X(P.hmdb_id.values, P.disease.values, block, DFmat=DFuse)
            y = P.label.values
            oof = np.full(len(P), np.nan)
            for tr, te in fl:
                oof[te] = fit_predict(model, seed, X[tr], y[tr], X[te])
            pdf = per_disease_pooled(P, oof)
            pdf["arm"], pdf["label"], pdf["seed"] = arm, label, seed
            pdf["n_features"] = X.shape[1]
            rows.append(pdf)
            print(f"  [{time.time()-t0:6.0f}s] {arm} seed={seed} "
                  f"auc={pdf.auc.mean():.4f} d={X.shape[1]}", flush=True)
    A = pd.concat(rows, ignore_index=True)
    A.to_csv("results/ablations.csv", index=False)
    print(A.groupby("arm")[["auc", "ap"]].mean().round(4).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("main", "all"):
        run_main()
    if what in ("ablate", "all"):
        run_ablate()
