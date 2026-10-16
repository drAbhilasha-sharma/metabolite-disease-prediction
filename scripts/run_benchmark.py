"""
Rebuilt benchmark: metabolite-disease association prediction.

Two evaluation regimes, both with metabolite-disjoint splits:
  CV   - leave-metabolites-out. Diseases are all seen in training.
         Answers: can the model rank NEW metabolites for a KNOWN disease?
  LDO  - leave-diseases-out. Whole diseases are held out.
         Answers: does it EXTRAPOLATE to a disease with no training labels?
         This is the manuscript's original question, now with 71 diseases
         instead of 4.

Models: logistic regression, random forest, XGBoost, and an MLP. All consume
the same feature vector [metabolite ECFP4 | disease embedding], so differences
are attributable to the learner, not the representation.

Controls: a label-permutation run (disease labels shuffled within the pair
table) and a disease-blind run (metabolite features only, no disease features)
which measures how much performance is metabolite popularity alone.

Everything is logged: per-fold, per-disease, per-seed.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import evalproto

SEEDS = [0, 1, 2, 3, 4]
NEG_RATIO = 5.0
N_FOLDS = 5

FPm = pd.read_csv("rebuild/fp_index.csv").hmdb_id.values
FPX = np.load("rebuild/fp_matrix.npy").astype(np.float32)
FPI = {m: i for i, m in enumerate(FPm)}
DF = pd.read_csv("rebuild/disease_features.csv", index_col=0)
POS = pd.read_csv("rebuild/model_positives.csv")
UNL = np.load("rebuild/unlabelled_ids.npy", allow_pickle=True)
DISEASES = sorted(POS.disease.unique())
DFv = DF.loc[DISEASES].values.astype(np.float32)
DFI = {d: i for i, d in enumerate(DISEASES)}


def featurize(pairs, disease_blind=False):
    mi = np.array([FPI[m] for m in pairs.hmdb_id.values])
    Xm = FPX[mi]
    if disease_blind:
        return Xm
    di = np.array([DFI[d] for d in pairs.disease.values])
    return np.hstack([Xm, DFv[di]])


def make_models(seed):
    return {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, random_state=seed, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, n_jobs=-1),
        "xgb": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                             random_state=seed, n_jobs=-1, eval_metric="logloss",
                             tree_method="hist"),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=300,
                             early_stopping=True, n_iter_no_change=15,
                             random_state=seed),
    }


def per_disease_metrics(pairs, y, p, regime, model, seed, fold):
    rows = []
    df = pairs.copy()
    df["y"] = y
    df["p"] = p
    for d, g in df.groupby("disease"):
        if g.y.nunique() < 2:
            continue
        rows.append({
            "regime": regime, "model": model, "seed": seed, "fold": fold,
            "disease": d, "n": len(g), "n_pos": int(g.y.sum()),
            "auc": roc_auc_score(g.y, g.p),
            "ap": average_precision_score(g.y, g.p),
            "prior": float(g.y.mean()),
        })
    return rows


def run(regime, permute=False, disease_blind=False, tag=""):
    all_rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        pairs = evalproto.build_labelled_pairs(POS, UNL, NEG_RATIO, rng)
        if permute:
            # shuffle which disease each row belongs to, keeping the label column
            pairs = pairs.copy()
            pairs["disease"] = rng.permutation(pairs["disease"].values)
        X = featurize(pairs, disease_blind)
        y = pairs.label.values

        if regime == "cv":
            folds = evalproto.metabolite_grouped_folds(pairs, N_FOLDS, seed)
        else:
            folds = evalproto.disease_grouped_folds(pairs, DISEASES, N_FOLDS, seed)

        for k, (tr, te) in enumerate(folds):
            if len(tr) == 0 or len(te) == 0:
                continue
            for name, mdl in make_models(seed).items():
                t0 = time.time()
                mdl.fit(X[tr], y[tr])
                p = mdl.predict_proba(X[te])[:, 1]
                rows = per_disease_metrics(pairs.iloc[te], y[te], p, regime + tag,
                                           name, seed, k)
                for r in rows:
                    r["fit_s"] = round(time.time() - t0, 2)
                all_rows.extend(rows)
            print(f"  {regime}{tag} seed={seed} fold={k} done", flush=True)
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = []
    if which in ("all", "cv"):
        out.append(run("cv"))
    if which in ("all", "ldo"):
        out.append(run("ldo"))
    if which in ("all", "ctrl"):
        out.append(run("cv", permute=True, tag="_perm"))
        out.append(run("cv", disease_blind=True, tag="_blind"))
        out.append(run("ldo", disease_blind=True, tag="_blind"))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(f"results/benchmark_{which}.csv", index=False)
    print(res.groupby(["regime", "model"]).auc.agg(["mean", "std", "count"]).round(4))
