"""Annotation-status audit: is annotation predictable from study effort?

A random-forest classifier separates annotated from unannotated compounds over
the full candidate universe from three disjoint blocks: curation-effort fields
(biospecimen count and reference-concentration counts), computed
physicochemical descriptors, and both together. A field that records
measurement in a disease context is evaluated separately, since it is close to
the label. The per-descriptor difference between the annotated and unannotated
pools is also reported.

Descriptors come from rebuild/descriptors_all.csv, computed for every compound
in the pool.

Writes results/attention_auc.csv, results/attention_descriptors.csv and
results/attention_effect_sizes.csv
"""
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DESC_COLS = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings",
             "heavy_atoms", "frac_csp3", "n_hetero", "formal_charge"]
# Two kinds of curation field, kept apart deliberately.
# CIRCULAR_COLS record that a compound was measured in a DISEASE context, so
# they are close to the label itself and cannot be read as study effort.
# EFFORT_COLS record measurement activity that is not disease-conditioned.
CIRCULAR_COLS = ["has_abnormal_conc"]
EFFORT_COLS = ["n_biospecimen", "has_normal_conc", "status_quantified",
               "status_detected"]
ALL_CURATION = EFFORT_COLS + CIRCULAR_COLS
SEEDS = [0, 1, 2, 3, 4]


def load():
    POS = pd.read_csv("rebuild/model_positives.csv")
    D = pd.read_csv("rebuild/descriptors_all.csv").drop_duplicates("hmdb_id")
    S = pd.read_csv("rebuild/metabolite_status.csv").drop_duplicates("hmdb_id")
    S["status_quantified"] = (S.status == "quantified").astype(int)
    S["status_detected"] = (S.status == "detected").astype(int)
    unl = np.load("rebuild/unlabeled_ids.npy", allow_pickle=True)

    ann = set(POS.hmdb_id)
    pool = pd.DataFrame({"hmdb_id": sorted(ann | set(unl.tolist()))})
    pool["annotated"] = pool.hmdb_id.isin(ann).astype(int)
    X = (pool.merge(D, on="hmdb_id", how="left")
             .merge(S[["hmdb_id"] + ALL_CURATION], on="hmdb_id", how="left"))
    return X


def cv_auc(X, y, cols, model="rf"):
    """Cross-validated AUROC of a feature block for annotation status."""
    Z = X[cols].astype(float).fillna(X[cols].astype(float).median())
    out = []
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in skf.split(Z, y):
            if model == "rf":
                clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                             n_jobs=-1, random_state=seed)
                clf.fit(Z.iloc[tr], y[tr])
            else:
                sc = StandardScaler().fit(Z.iloc[tr])
                clf = LogisticRegression(max_iter=3000, random_state=seed)
                clf.fit(sc.transform(Z.iloc[tr]), y[tr])
                oof[te] = clf.predict_proba(sc.transform(Z.iloc[te]))[:, 1]
                continue
            oof[te] = clf.predict_proba(Z.iloc[te])[:, 1]
        out.append(roc_auc_score(y, oof))
    return float(np.mean(out)), float(np.std(out, ddof=1))


def main():
    X = load()
    y = X.annotated.values
    have_eff = X[ALL_CURATION].notna().any(axis=1)
    rows = []
    for name, cols, sub in [
            ("study effort only", EFFORT_COLS, have_eff),
            ("disease-context record", CIRCULAR_COLS, have_eff),
            ("descriptors only", DESC_COLS, np.ones(len(X), bool)),
            ("effort + descriptors", EFFORT_COLS + DESC_COLS, have_eff)]:
        Xi, yi = X[sub], y[sub]
        m, s = cv_auc(Xi, yi, cols)
        rows.append(dict(feature_block=name, n=int(sub.sum()),
                         n_annotated=int(yi.sum()), auc=m, sd=s))
        print(f"  {name:22s} n={int(sub.sum()):6d} auc={m:.4f}", flush=True)
    pd.DataFrame(rows).to_csv("results/attention_auc.csv", index=False)

    # per-descriptor separation between the annotated and unannotated pools
    eff = []
    for c in DESC_COLS + EFFORT_COLS:
        a = X.loc[X.annotated == 1, c].astype(float).dropna()
        b = X.loc[X.annotated == 0, c].astype(float).dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        u = st.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial correlation = 2*AUC-1, a bounded effect size
        auc_c = u.statistic / (len(a) * len(b))
        pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                             (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        eff.append(dict(variable=c, mean_annotated=a.mean(),
                        mean_unannotated=b.mean(),
                        cohens_d=(a.mean() - b.mean()) / pooled_sd,
                        auc_single=auc_c, rank_biserial=2 * auc_c - 1,
                        p=u.pvalue, n_annotated=len(a), n_unannotated=len(b)))
    E = pd.DataFrame(eff).sort_values("rank_biserial", key=abs, ascending=False)
    E.to_csv("results/attention_effect_sizes.csv", index=False)
    X[["hmdb_id", "annotated"] + DESC_COLS + EFFORT_COLS].to_csv(
        "results/attention_descriptors.csv", index=False)
    print(E.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
