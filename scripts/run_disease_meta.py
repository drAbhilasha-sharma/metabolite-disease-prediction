"""Disease-level meta-regression of per-disease AUROC.

Each disease contributes one AUROC (pooled out-of-fold, averaged over seeds) in
the interpolation regime. The AUROC is related to five disease properties: the
number of positives, the size of the gene profile, the distinctiveness of the
disease representation, the chemical spread of its metabolites and the mean
annotation degree of its positives, by Spearman correlation and by a
standardized multiple regression.

Writes results/disease_meta.csv and results/disease_meta_model.csv
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st
from sklearn.metrics.pairwise import cosine_similarity

R = pd.read_csv("results/definitive_per_disease.csv")
POS0 = pd.read_csv("rebuild/model_positives.csv")
GENES = json.load(open("rebuild/disease_genes.json"))
# positives and gene-set size per disease (the composition of Supplementary Table 1)
S1 = pd.DataFrame({"disease": sorted(POS0.disease.unique())})
S1["n_positives"] = S1.disease.map(POS0.groupby("disease").size())
S1["n_genes"] = S1.disease.map(lambda d: len(GENES.get(d, [])))
S1 = S1.sort_values("n_positives", ascending=False)  # row order of Supplementary Table 1
POS = pd.read_csv("rebuild/model_positives.csv")
DF = pd.read_csv("rebuild/disease_features.csv", index_col=0)
FPI = pd.read_csv("rebuild/fp_index.csv")
FP = np.load("rebuild/fp_matrix.npy")
idx = {m: i for i, m in enumerate(FPI.hmdb_id)}

# outcome: per-disease AUROC of the best learner, full arm, interpolation
best = (R[(R["mode"] == "full") & (R["regime"] == "lmo")]
        .groupby("model").auc.mean().idxmax())
y = (R[(R["mode"] == "full") & (R["regime"] == "lmo") & (R["model"] == best)]
     .groupby("disease").auc.mean())

# covariate 1-2: annotation volume and gene-profile size
cov = S1.set_index("disease")[["n_positives", "n_genes"]].copy()

# covariate 3: representation distinctiveness -- 1 - max cosine to any other disease
Sim = cosine_similarity(DF.values)
np.fill_diagonal(Sim, -1.0)
cov["distinctiveness"] = pd.Series(1.0 - Sim.max(axis=1), index=DF.index)

# covariate 4: chemical spread of the disease's own metabolites (mean pairwise
# Tanimoto distance); a disease whose metabolites form one tight cluster is an
# easier target than one spread across chemical space
def tanimoto_spread(mets):
    rows = [idx[m] for m in mets if m in idx]
    if len(rows) < 2:
        return np.nan
    if len(rows) > 120:
        rows = list(np.random.default_rng(0).choice(rows, 120, replace=False))
    X = FP[rows].astype(bool)
    inter = X @ X.T
    tot = X.sum(1)[:, None] + X.sum(1)[None, :] - inter
    T = np.where(tot > 0, inter / np.maximum(tot, 1), 0.0)
    iu = np.triu_indices(len(rows), 1)
    return float(1.0 - T[iu].mean())

cov["chem_spread"] = pd.Series(
    {d: tanimoto_spread(g.hmdb_id.tolist()) for d, g in POS.groupby("disease")})

# covariate 5: annotation degree of the disease's positives -- how many other
# diseases those same metabolites belong to
deg = POS.groupby("hmdb_id").disease.nunique()
cov["mean_pos_degree"] = POS.groupby("disease").hmdb_id.apply(
    lambda s: deg.loc[s].mean())

D = cov.join(y.rename("auc")).dropna()
D["log_n_positives"] = np.log10(D.n_positives)
D["log_n_genes"] = np.log10(D.n_genes.clip(lower=1))

PRED = ["log_n_positives", "log_n_genes", "distinctiveness",
        "chem_spread", "mean_pos_degree"]

# univariate Spearman, then a standardized multiple regression
rows = []
for p in PRED:
    rho, pv = st.spearmanr(D[p], D.auc)
    rows.append(dict(covariate=p, spearman_rho=rho, p_spearman=pv))
U = pd.DataFrame(rows)

Z = (D[PRED] - D[PRED].mean()) / D[PRED].std()
fit = sm.OLS(D.auc.values, sm.add_constant(Z.values)).fit()
U["beta_std"] = fit.params[1:]
U["beta_se"] = fit.bse[1:]
U["beta_ci_lo"] = fit.conf_int()[1:, 0]
U["beta_ci_hi"] = fit.conf_int()[1:, 1]
U["p_beta"] = fit.pvalues[1:]

D.to_csv("results/disease_meta.csv")
U.to_csv("results/disease_meta_model.csv", index=False)
print(f"learner={best}  n_diseases={len(D)}  R2={fit.rsquared:.3f}  "
      f"adjR2={fit.rsquared_adj:.3f}  F_p={fit.f_pvalue:.2e}")
print(U.round(4).to_string(index=False))
