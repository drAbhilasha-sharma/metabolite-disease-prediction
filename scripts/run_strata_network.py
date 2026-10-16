"""Stratified performance and the topology of the association network.

Two analyses that a single global AUROC hides.

STRATIFIED PERFORMANCE
    Per-metabolite out-of-fold scores are grouped by chemical super-class, by
    Bemis-Murcko scaffold frequency, and by how many diseases the metabolite is
    annotated to. If skill is concentrated in the promiscuous compounds, the
    global figure is carried by a subset that a screen would already know about.

NETWORK TOPOLOGY
    The metabolite-disease association set is a bipartite graph. Its degree
    distributions, assortativity and clustering determine how much of the label
    matrix is predictable from degree alone -- the graph-theoretic statement of
    the popularity confound. We compare the observed graph against a
    degree-preserving null (configuration model) to separate topology from
    degree.

Writes results/strata_class.csv, results/strata_scaffold.csv,
results/strata_degree.csv, results/network_stats.csv and
results/network_degree_dist.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(7)
N_NULL = 200


def per_metabolite_scores():
    """One AUROC-style score per metabolite: the within-disease rank quality of
    its predictions, averaged over the diseases it appears in."""
    OOF = pd.read_parquet("results/definitive_oof.parquet")
    s = OOF[(OOF["regime"] == "lmo") & (OOF["mode"] == "full")
            & (OOF["model"] == "rf")]
    g = (s.groupby(["disease", "hmdb_id"])
          .agg(label=("label", "first"), p=("p", "mean")).reset_index())
    rows = []
    for d, gg in g.groupby("disease"):
        if gg.label.nunique() < 2:
            continue
        # percentile of each positive within that disease's candidate list
        r = gg.p.rank(pct=True)
        for hid, lab, pc in zip(gg.hmdb_id, gg.label, r):
            if lab == 1:
                rows.append(dict(disease=d, hmdb_id=hid, pct_rank=pc))
    return pd.DataFrame(rows)


def strata(M, key, labels=None):
    out = []
    for k, g in M.groupby(key):
        if len(g) < 15:
            continue
        m = g.pct_rank.mean()
        se = g.pct_rank.std(ddof=1) / np.sqrt(len(g))
        t = st.t.ppf(0.975, len(g) - 1)
        out.append(dict(stratum=str(k), n=len(g), mean_pct_rank=m,
                        ci_lo=m - t * se, ci_hi=m + t * se))
    D = pd.DataFrame(out).sort_values("mean_pct_rank", ascending=False)
    return D


def network():
    POS = pd.read_csv("rebuild/model_positives.csv")
    md = POS.groupby("hmdb_id").disease.nunique()
    dd = POS.groupby("disease").hmdb_id.nunique()
    E = POS[["hmdb_id", "disease"]].drop_duplicates()

    # degree assortativity across the bipartite edge list
    x = md.loc[E.hmdb_id].values.astype(float)
    y = dd.loc[E.disease].values.astype(float)
    r_assort = float(np.corrcoef(x, y)[0, 1])

    # how much of the edge set a degree-only model recovers, against a
    # degree-preserving null
    obs_pairs = set(map(tuple, E.values))
    mets, diss = md.index.to_numpy(), dd.index.to_numpy()
    pm = (md / md.sum()).loc[mets].values
    pdz = (dd / dd.sum()).loc[diss].values
    hits = []
    for _ in range(N_NULL):
        a = RNG.choice(len(mets), size=len(E), p=pm)
        b = RNG.choice(len(diss), size=len(E), p=pdz)
        hits.append(len({(mets[i], diss[j]) for i, j in zip(a, b)} & obs_pairs))
    hits = np.array(hits, float)

    stats = dict(
        n_metabolites=int(E.hmdb_id.nunique()), n_diseases=int(E.disease.nunique()),
        n_edges=int(len(E)),
        density=len(E) / (E.hmdb_id.nunique() * E.disease.nunique()),
        met_degree_mean=float(md.mean()), met_degree_max=int(md.max()),
        met_degree_gini=float(gini(md.values)),
        dis_degree_mean=float(dd.mean()), dis_degree_max=int(dd.max()),
        dis_degree_gini=float(gini(dd.values)),
        degree_assortativity=r_assort,
        null_edge_recovery_frac=float(hits.mean() / len(E)),
        null_edge_recovery_sd=float(hits.std(ddof=1) / len(E)),
    )
    # power-law-ish tail: OLS slope on log-log complementary CDF
    for nm, v in [("met", md.values), ("dis", dd.values)]:
        u, c = np.unique(v, return_counts=True)
        ccdf = 1.0 - np.cumsum(c) / c.sum()
        ok = (ccdf > 0) & (u > 0)
        sl = st.linregress(np.log10(u[ok]), np.log10(ccdf[ok]))
        stats[f"{nm}_ccdf_slope"] = float(sl.slope)
        stats[f"{nm}_ccdf_r2"] = float(sl.rvalue ** 2)
    return stats, md, dd


def gini(v):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def main():
    M = per_metabolite_scores()
    # chemical class of each metabolite: from the full HMDB parse when present,
    # otherwise from the deposited per-metabolite extract (modeled metabolites only)
    src = ("rebuild/metabolites_all.csv" if os.path.exists("rebuild/metabolites_all.csv")
           else "rebuild/metabolite_classes.csv")
    MA = pd.read_csv(src)[["hmdb_id", "super_class", "chem_class"]].drop_duplicates("hmdb_id")
    SC = pd.read_csv("rebuild/scaffolds.csv").drop_duplicates("hmdb_id")
    POS = pd.read_csv("rebuild/model_positives.csv")
    deg = POS.groupby("hmdb_id").disease.nunique()

    M = M.merge(MA, on="hmdb_id", how="left").merge(SC, on="hmdb_id", how="left")
    M["n_diseases"] = M.hmdb_id.map(deg)
    M["degree_bin"] = pd.cut(M.n_diseases, [0, 1, 2, 4, 8, 1000],
                             labels=["1", "2", "3-4", "5-8", "9+"])
    sf = M.scaffold.value_counts()
    M["scaffold_freq"] = pd.cut(M.scaffold.map(sf), [0, 1, 5, 25, 100, 1e6],
                                labels=["unique", "2-5", "6-25", "26-100", ">100"])

    strata(M, "super_class").to_csv("results/strata_class.csv", index=False)
    strata(M, "scaffold_freq").to_csv("results/strata_scaffold.csv", index=False)
    strata(M, "degree_bin").to_csv("results/strata_degree.csv", index=False)

    # is rank quality driven by promiscuity?
    rho, pv = st.spearmanr(M.n_diseases, M.pct_rank)
    stats, md, dd = network()
    stats["pctrank_vs_degree_rho"] = float(rho)
    stats["pctrank_vs_degree_p"] = float(pv)
    pd.DataFrame([stats]).to_csv("results/network_stats.csv", index=False)

    dist = pd.concat([
        pd.DataFrame({"side": "metabolite", "degree": md.values}),
        pd.DataFrame({"side": "disease", "degree": dd.values})])
    dist.to_csv("results/network_degree_dist.csv", index=False)
    M.to_csv("results/per_metabolite_rank.csv", index=False)
    print(pd.DataFrame([stats]).T.to_string())


if __name__ == "__main__":
    main()
