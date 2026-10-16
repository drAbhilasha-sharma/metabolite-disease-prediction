"""Summary statistics reported in the paper, computed from the analysis outputs.

Run after the analysis scripts (see run_all.sh). Each summary is written to
results/ and skipped with a message if its inputs are missing.

  table2_primary.csv          per-learner AUROC (95% t-interval across diseases)
                              and AP, both regimes, three arms
  ldo_vs_chance.csv           held-out-disease AUROC against chance, with TOST
  table4_ablation_stats.csv   paired tests between feature-block arms
  table5_id_control.csv       randomised-representation controls vs the real profile
  transfer_repr_tests.csv     disease-representation exchange vs the original profile
  graph_vs_baseline.csv       graph learners vs the random-forest baseline
  sampler_grid_tests.csv      negative-sampler grid, TOST against chance
  sampler_fallback.csv        fallback draws of the degree-matched sampler
  sampler_degree_leak.csv     degree-only AUROC under each sampler
  belowchance_tests.csv       below-chance arms, intact vs shuffled training labels
  comparator_shares.csv       share of the excess over chance, both corpora
  learning_curve_fit.json     bounded power-law fit of the learning curves
  temporal_tests.csv          prospective split, TOST against chance
  corroboration_tests.csv     curation-depth strata, TOST against chance
  conventional_summary.csv    conventional protocol, mean AUROC per arm and learner
  final_benchmark_summary.csv matched-negative design, mean AUROC per arm and learner

Conventions: per-disease AUROC is the unit of analysis; seed-level tests use the
per-seed mean over diseases; TOST uses the equivalence margin 0.5 +/- 0.05.

    PYTHONPATH=scripts python scripts/summarize_results.py
"""
import ast
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "results/"
BOUND = 0.05
MODELS = ["logreg", "rf", "xgb", "mlp"]
MODEL_LBL = {"logreg": "Logistic regression", "rf": "Random forest",
             "xgb": "Gradient boosting", "mlp": "Neural network"}
REG_LBL = {"lmo": "Held-out metabolites", "ldo": "Held-out diseases"}


# ------------------------------------------------------------------ helpers
def t_ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def g3(p):
    return float(f"{p:.3g}")


def tost(x, bound=BOUND, centre=0.5):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    p_lo = stats.t.sf((x.mean() - (centre - bound)) / se, len(x) - 1)
    p_hi = stats.t.cdf((x.mean() - (centre + bound)) / se, len(x) - 1)
    return max(p_lo, p_hi)


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    _, lo, hi = t_ci(d)
    return dict(diff=d.mean(), ci_lo=lo, ci_hi=hi, p_t=stats.ttest_rel(a, b).pvalue,
                p_wilcox=stats.wilcoxon(d).pvalue, dz=d.mean() / d.std(ddof=1),
                n_better=int((d > 0).sum()), n=len(d))


def seed_means(df, keys):
    return df.groupby(keys + ["seed"]).auc.mean()


def seed_paired(a, b):
    d = a - b
    _, lo, hi = t_ci(d)
    return d.mean(), lo, hi, stats.ttest_rel(a, b).pvalue


def read(name):
    return pd.read_csv(R + name)


def positives():
    return (pd.read_csv("rebuild/model_positives.csv")[["hmdb_id", "disease"]]
            .drop_duplicates().reset_index(drop=True))


# --------------------------------------------------------- main benchmark
def table2_primary():
    D = read("definitive_per_disease.csv")
    rows = []
    for reg in ["lmo", "ldo"]:
        for m in MODELS:
            row = {"Regime": REG_LBL[reg], "Learner": MODEL_LBL[m]}
            for mode, nm in [("full", "Full"), ("perm", "Disease-permuted"),
                             ("blind", "Disease-blind")]:
                s = D[(D.regime == reg) & (D.model == m) & (D["mode"] == mode)]
                mu, lo, hi = t_ci(s.groupby("disease").auc.mean())
                row[f"AUROC {nm}"] = f"{mu:.3f} ({lo:.3f}-{hi:.3f})"
                row[f"AP {nm}"] = f"{s.groupby('disease').ap.mean().mean():.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def ldo_vs_chance():
    D = read("definitive_per_disease.csv")
    rows = []
    for m in ["rf", "xgb", "mlp", "logreg"]:
        a = D[(D.regime == "ldo") & (D["mode"] == "full") & (D.model == m)] \
            .groupby("disease").auc.mean()
        mu, lo, hi = t_ci(a)
        p = tost(a.values)
        rows.append(dict(model=m, mean_auc=round(mu, 4), ci=f"{lo:.3f} to {hi:.3f}",
                         p_vs_chance=g3(stats.ttest_1samp(a, 0.5).pvalue),
                         p_wilcox=g3(stats.wilcoxon(a - 0.5).pvalue),
                         p_tost_equiv=float(f"{p:.4g}"),
                         verdict="EQUIVALENT to chance" if p < 0.05 else "inconclusive"))
    return pd.DataFrame(rows)


def _arm(arm, fname="ablations.csv"):
    A = read(fname)
    return A[A.arm == arm].groupby("disease").auc.mean()


def table4_ablation_stats():
    rows = []
    for a, b in [("both", "shuffled_dis"), ("both", "fp_only"), ("both", "dis_only"),
                 ("both", "kegg"), ("shuffled_dis", "fp_only")]:
        A, B = _arm(a), _arm(b)
        st = paired(A.values, B.loc[A.index].values)
        rows.append(dict(comparison=f"{a} vs {b}", diff=round(st["diff"], 4),
                         ci=f"{st['ci_lo']:.3f} to {st['ci_hi']:.3f}", p_t=g3(st["p_t"]),
                         p_wilcox=g3(st["p_wilcox"]), dz=round(st["dz"], 3),
                         n_better=st["n_better"], n=st["n"]))
    return pd.DataFrame(rows)


def table5_id_control():
    real = _arm("both")
    rows = []
    for arm in ["rand_id", "onehot"]:
        X = _arm(arm, "id_control.csv").loc[real.index]
        d = real.values - X.values
        _, lo, hi = t_ci(d)
        p = tost(d, centre=0.0)
        rows.append(dict(arm=arm, auc=round(X.mean(), 4), vs_real_diff=round(d.mean(), 4),
                         ci=f"{lo:.3f} to {hi:.3f}",
                         p_t=g3(stats.ttest_rel(real.values, X.values).pvalue),
                         p_tost_equiv=g3(p), equivalent="yes" if p < 0.05 else "no"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ interventions
def transfer_repr_tests():
    S = seed_means(read("transfer_repr.csv"), ["regime", "repr"])
    rows = []
    for reg in ["lmo", "ldo"]:
        base = S.loc[(reg, "gene_expression")]
        for rep in ["mesh_tree", "mesh_text", "tree_plus_text", "random_gauss"]:
            x = S.loc[(reg, rep)].loc[base.index]
            dlt, lo, hi, p = seed_paired(x.values, base.values)
            rows.append(dict(regime=reg, repr=rep, mean=x.mean(), delta=dlt,
                             ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(rows)


def graph_vs_baseline():
    G = read("graph_models.csv")
    D = read("definitive_per_disease.csv")
    S = seed_means(G[G["mode"] == "full"], ["regime", "model"])
    B = seed_means(D[(D.model == "rf") & (D["mode"] == "full")], ["regime"])
    rows = []
    for reg in ["lmo", "ldo"]:
        base = B.loc[reg]
        for m in ["gcn", "labelprop", "mf"]:
            x = S.loc[(reg, m)].loc[base.index]
            dlt, lo, hi, p = seed_paired(x.values, base.values)
            rows.append(dict(regime=reg, model=m, mean_auc=x.mean(), rf_baseline=base.mean(),
                             delta=dlt, ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ samplers
SAMPLERS = ["exact_1to1", "tol_pm1", "ratio_2", "ratio_3", "degree_plus_size", "uniform"]


def sampler_grid_tests():
    S = seed_means(read("sampler_grid.csv"), ["variant", "regime"])
    rows = []
    for v in SAMPLERS:
        x = S.loc[(v, "ldo")].values
        rows.append(dict(variant=v, ldo_mean=x.mean(), ldo_sd=x.std(ddof=1),
                         tost_p_vs_chance=tost(x), lmo_mean=S.loc[(v, "lmo")].mean()))
    return pd.DataFrame(rows)


def sampler_fallback():
    from degree_matched import degree_matched_pairs
    POS = positives()
    deg = POS.groupby("hmdb_id").disease.nunique()
    rows = []
    for seed in range(5):
        P = degree_matched_pairs(POS, np.random.default_rng(100 + seed))
        P["deg"] = P.hmdb_id.map(deg)
        pos, neg = P[P.label == 1], P[P.label == 0]
        fb = int((neg.deg.values != pos.deg.values).sum())
        rows.append(dict(seed=seed, n_pairs=len(P), n_neg=len(neg), fallback=fb,
                         exhausted=len(pos) - len(neg), pct_fallback=100 * fb / len(neg),
                         mean_deg_pos=pos.deg.mean(), mean_deg_neg=neg.deg.mean()))
    return pd.DataFrame(rows)


def sampler_degree_leak():
    """Degree-only AUROC under each sampler, with the sampler of run_robustness.py.

    The degree_plus_size variant needs molecular weights from
    rebuild/descriptors_all.csv, and the uniform variant is drawn after it from the
    same generator; without that file only the first four variants are computed.
    """
    import degree_matched
    from sklearn.metrics import roc_auc_score
    src = open("scripts/run_robustness.py").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "sample_variant")
    mw = None
    if os.path.exists("rebuild/descriptors_all.csv"):
        d = pd.read_csv("rebuild/descriptors_all.csv")
        mw = d.set_index("hmdb_id")["mw"] if "mw" in d.columns else None
    ns = dict(np=np, pd=pd, degree_matched_pairs=degree_matched.degree_matched_pairs,
              degree_strata=degree_matched.degree_strata, MW=mw)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "run_robustness.py", "exec"), ns)
    variants = SAMPLERS if mw is not None else SAMPLERS[:4]
    POS = positives()
    deg = POS.groupby("hmdb_id").disease.nunique()
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        for v in variants:
            P = ns["sample_variant"](POS, rng, v).reset_index(drop=True)
            P["deg"] = P.hmdb_id.map(deg)
            per = [roc_auc_score(g.label, g.deg) for _, g in P.groupby("disease")
                   if g.label.nunique() == 2]
            rows.append(dict(seed=seed, variant=v,
                             degree_auc_pooled=roc_auc_score(P.label, P.deg),
                             degree_auc_per_disease=float(np.mean(per)),
                             mean_deg_pos=P[P.label == 1].deg.mean(),
                             mean_deg_neg=P[P.label == 0].deg.mean()))
    if mw is None:
        print("  sampler_degree_leak: degree_plus_size and uniform need "
              "rebuild/descriptors_all.csv; computed the other four variants")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- robustness
def belowchance_tests():
    BC = read("belowchance.csv")
    rows = []
    for (blk, sp), g in BC.groupby(["block", "split"]):
        a = g[g.train_labels == "intact"].sort_values("seed").auc.values
        b = g[g.train_labels == "shuffled"].sort_values("seed").auc.values
        p1 = stats.ttest_rel(a, b).pvalue
        p2 = stats.ttest_1samp(a, 0.5).pvalue
        rows.append(dict(block=blk, split=sp, auc_intact=a.mean(), auc_shuffled=b.mean(),
                         delta=a.mean() - b.mean(),
                         p_vs_shuffled=1.0 if np.isnan(p1) else p1,
                         p_intact_vs_chance=1.0 if np.isnan(p2) else p2))
    return pd.DataFrame(rows)


def comparator_shares():
    M = read("comparator_controls.csv").groupby(["dataset", "split", "arm"]).auc.mean()
    rows = []
    for ds in ["gene_disease", "metabolite_disease"]:
        for sp in ["grouped_node", "random_pair"]:
            full = M.loc[(ds, sp, "full")]
            for arm in ["perm", "blind", "degree"]:
                a = M.loc[(ds, sp, arm)]
                share = (a - 0.5) / (full - 0.5) if sp == "random_pair" else np.nan
                rows.append(dict(dataset=ds, split=sp, arm=arm, auc=round(a, 3),
                                 full=round(full, 3), share_of_excess=share))
    return pd.DataFrame(rows)


def learning_curve_fit():
    from scipy.optimize import curve_fit
    L = read("learning_curve.csv")
    power = lambda n, a, b, c: a - b * n ** (-c)  # noqa: E731
    out = {"n_train_full": 1, "fits": {}}
    for reg in ["lmo", "ldo"]:
        g = L[L.regime == reg].groupby("fraction").mean_auc.mean()
        p, _ = curve_fit(power, g.index.values, g.values, p0=[0.8, 1.0, 0.4],
                         bounds=([0.5, 0, 0.01], [1.0, 50, 3]), maxfev=40000)
        out["fits"][reg] = dict(asymptote=p[0], b=p[1], c=p[2],
                                auc_at_10x=power(10, *p), auc_at_1x=power(1, *p))
    return out


def temporal_tests():
    TH = read("temporal_holdout.csv")
    rows = []
    for (sub, cut), g in TH.groupby(["subset", "cutoff"]):
        x = g.sort_values("seed").auc.values
        rows.append(dict(subset=sub, cutoff=cut, n_seeds=len(x), mean_auc=x.mean(),
                         sd_over_seeds=x.std(ddof=1), tost_p_vs_chance=tost(x), bound=BOUND))
    return pd.DataFrame(rows)


def corroboration_tests():
    C = seed_means(read("corroboration.csv"), ["subset", "regime"])
    rows = []
    for sub in ["single_reference", "multi_reference"]:
        x = C.loc[(sub, "ldo")].values
        rows.append(dict(subset=sub, lmo_mean=C.loc[(sub, "lmo")].mean(), ldo_mean=x.mean(),
                         ldo_sd=x.std(ddof=1), tost_p_vs_chance=tost(x), bound=BOUND))
    return pd.DataFrame(rows)


# ------------------------------------------------------- earlier protocols
def conventional_summary():
    B = pd.concat([read(f"benchmark_{w}.csv") for w in ["cv", "ldo", "ctrl"]], ignore_index=True)
    return B.groupby(["regime", "model"]).auc.mean().round(4).reset_index(name="mean_auc")


def final_benchmark_summary():
    F = read("final_benchmark.csv")
    return F.groupby(["regime", "mode", "model"]).auc.mean().round(4).reset_index(name="mean_auc")


SUMMARIES = [
    ("table2_primary.csv", table2_primary, ["definitive_per_disease.csv"]),
    ("ldo_vs_chance.csv", ldo_vs_chance, ["definitive_per_disease.csv"]),
    ("table4_ablation_stats.csv", table4_ablation_stats, ["ablations.csv"]),
    ("table5_id_control.csv", table5_id_control, ["ablations.csv", "id_control.csv"]),
    ("transfer_repr_tests.csv", transfer_repr_tests, ["transfer_repr.csv"]),
    ("graph_vs_baseline.csv", graph_vs_baseline, ["graph_models.csv", "definitive_per_disease.csv"]),
    ("sampler_grid_tests.csv", sampler_grid_tests, ["sampler_grid.csv"]),
    ("sampler_fallback.csv", sampler_fallback, []),
    ("sampler_degree_leak.csv", sampler_degree_leak, []),
    ("belowchance_tests.csv", belowchance_tests, ["belowchance.csv"]),
    ("comparator_shares.csv", comparator_shares, ["comparator_controls.csv"]),
    ("learning_curve_fit.json", learning_curve_fit, ["learning_curve.csv"]),
    ("temporal_tests.csv", temporal_tests, ["temporal_holdout.csv"]),
    ("corroboration_tests.csv", corroboration_tests, ["corroboration.csv"]),
    ("conventional_summary.csv", conventional_summary,
     ["benchmark_cv.csv", "benchmark_ldo.csv", "benchmark_ctrl.csv"]),
    ("final_benchmark_summary.csv", final_benchmark_summary, ["final_benchmark.csv"]),
]


def main():
    os.makedirs(R, exist_ok=True)
    for name, fn, needs in SUMMARIES:
        missing = [f for f in needs if not os.path.exists(R + f)]
        if missing:
            print(f"skip  {name:30s} missing {', '.join(missing)}")
            continue
        out = fn()
        if name.endswith(".json"):
            json.dump(out, open(R + name, "w"), indent=1)
        else:
            out.to_csv(R + name, index=False)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
