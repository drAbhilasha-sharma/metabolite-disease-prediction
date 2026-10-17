"""pucontrols -- five leakage controls for positive-unlabeled association studies.

A dataset-agnostic implementation of the diagnostic used in the accompanying
study. It takes an edge list and a feature callback and returns a report
saying whether a reported score reflects the intended relationship or an
artifact of how the benchmark was built.

The library assumes only the shape that this class of study shares: a
bipartite graph of (source, target) associations that are known to be
positive, with everything unobserved treated as negative.

    from pucontrols import audit

    report = audit(
        edges=df,                       # columns: source, target
        featurize=my_features,          # (pairs, train_idx) -> X
        n_seeds=5,
    )
    print(report.summary())

WHAT IS TESTED
--------------
1. permutation      Targets are permuted across pairs, breaking the
                    correspondence while preserving both degree marginals.
                    A score that survives this is not measuring the pairing.
2. target_blind     The target block is deleted, so every pair sharing a
                    source scores identically. The gap against the full model
                    is the target representation's real contribution.
3. degree_only      A univariate model on annotation degree alone. If this
                    approaches the full model, the benchmark is measuring
                    popularity.
4. group_split      The same model under source-grouped splitting instead of
                    random pair splitting. The drop is the amount of the
                    original score attributable to seeing the same source in
                    training.
5. duplicate_target Targets with identical feature vectors are located; if
                    any exist, a nominally held-out target may be present in
                    training under another name.

Each control returns an effect size against the full model with a bootstrap
interval, plus a verdict string. No control requires a corrected pipeline to
already exist -- the point is to run them on the pipeline you have.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

__all__ = ["audit", "AuditReport", "sample_negatives", "degree_matched_negatives"]


# --------------------------------------------------------------- samplers
def sample_negatives(edges: pd.DataFrame, rng, ratio: int = 1) -> pd.DataFrame:
    """Uniform unlabeled negatives -- the conventional protocol's sampler."""
    sources = edges.source.unique()
    ann = edges.groupby("target").source.apply(set).to_dict()
    rows = []
    for t, g in edges.groupby("target"):
        pool = np.array([s for s in sources if s not in ann[t]])
        if len(pool) == 0:
            continue
        k = min(ratio * len(g), len(pool))
        neg = rng.choice(pool, size=k, replace=False)
        rows += [(s, t, 1) for s in g.source]
        rows += [(s, t, 0) for s in neg]
    return pd.DataFrame(rows, columns=["source", "target", "label"])


def degree_matched_negatives(edges: pd.DataFrame, rng) -> pd.DataFrame:
    """Negatives matched to positives on EXACT annotation degree.

    Closes the degree confound at sampling time: a model cannot separate the
    classes by counting how often a source appears, because both classes carry
    the same degree distribution by construction.

    Each positive of degree k draws a negative of degree k. When no unused
    candidate of degree k is available for that target, the nearest available
    degree is used instead and the substitution is counted, so the fallback
    rate is reportable rather than silent. Quantile strata are deliberately
    NOT used: with a heavy-tailed degree distribution the top stratum spans a
    wide range, and matching only on stratum leaves a residual gap in mean
    degree between the classes large enough for a model to exploit.

    The returned frame carries the diagnostics on ``.attrs``:
    ``fallback_draws``, ``exhausted``, ``n_negatives``.
    """
    degree = edges.groupby("source").target.nunique()
    by_degree = {k: np.array(sorted(g.index)) for k, g in degree.groupby(degree)}
    deg_vals = np.array(sorted(by_degree))
    ann = edges.groupby("target").source.apply(set).to_dict()

    rows, fallbacks, exhausted = [], 0, 0
    for t, g in edges.groupby("target"):
        banned, used = ann[t], set()
        for s in g.source:
            rows.append((s, t, 1))
            k0 = int(degree.get(s, 0))
            drawn, exact = None, True
            for k in sorted(deg_vals, key=lambda k: (abs(k - k0), -k)):
                pool = [x for x in by_degree[k] if x not in banned]
                if not pool:
                    continue
                fresh = [x for x in pool if x not in used]
                cand = fresh if fresh else pool
                drawn = cand[int(rng.integers(len(cand)))]
                exact = (k == k0)
                break
            if drawn is None:
                exhausted += 1
                continue
            if not exact:
                fallbacks += 1
            used.add(drawn)
            rows.append((drawn, t, 0))

    out = pd.DataFrame(rows, columns=["source", "target", "label"])
    out.attrs.update(fallback_draws=fallbacks, exhausted=exhausted,
                     n_negatives=int((out.label == 0).sum()))
    return out


# ----------------------------------------------------------------- report
@dataclasses.dataclass
class AuditReport:
    rows: list

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self) -> str:
        out = []
        for r in self.rows:
            out.append(f"{r['control']:16s} auc={r['auc']:.3f} "
                       f"delta={r['delta']:+.3f} "
                       f"[{r['lo']:+.3f},{r['hi']:+.3f}]  {r['verdict']}")
        return "\n".join(out)


def _fit_predict(X, y, folds, seed):
    pred = np.zeros(len(y))
    for tr, te in folds:
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=seed)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def _boot_ci(v, rng, n=2000):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def audit(edges: pd.DataFrame,
          featurize: Callable[[pd.DataFrame, np.ndarray], np.ndarray],
          n_seeds: int = 5,
          n_folds: int = 5,
          sampler: Callable = sample_negatives,
          target_features: pd.DataFrame | None = None,
          n_source_cols: int | None = None,
          verbose: bool = True) -> AuditReport:
    """Run the five controls and return an AuditReport.

    edges      DataFrame with columns [source, target]; every row a known
               positive association.
    featurize  callback (pairs, train_index) -> feature matrix. It receives
               the labeled pair table and the training indices of the current
               fold, so representations derived from the association matrix
               can be built from training edges only. Returning a matrix that
               depends on test edges will silently invalidate the audit.
    n_source_cols
               number of leading columns of the feature matrix belonging to
               the source block; required for the target-blind control.
    target_features
               optional frame indexed by target, used only by the
               duplicate-target control.
    """
    per = {k: [] for k in ["full", "permutation", "target_blind",
                           "degree_only", "group_split"]}
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        P = sampler(edges, rng).reset_index(drop=True)
        y = P.label.values
        rand_folds = list(KFold(n_folds, shuffle=True,
                                random_state=seed).split(P))
        grp_folds = list(GroupKFold(n_folds).split(P, y, groups=P.source))

        X = featurize(P, np.arange(len(P)))
        per["full"].append(roc_auc_score(y, _fit_predict(X, y, rand_folds, seed)))

        Pp = P.copy()
        Pp["target"] = rng.permutation(Pp.target.values)
        Xp = featurize(Pp, np.arange(len(Pp)))
        per["permutation"].append(
            roc_auc_score(y, _fit_predict(Xp, y, rand_folds, seed)))

        k = n_source_cols if n_source_cols else X.shape[1] // 2
        Xb = X[:, :max(1, k)]
        per["target_blind"].append(
            roc_auc_score(y, _fit_predict(Xb, y, rand_folds, seed)))

        deg = P.source.map(edges.groupby("source").target.nunique()).fillna(0)
        per["degree_only"].append(
            roc_auc_score(y, _fit_predict(deg.values.reshape(-1, 1), y,
                                          rand_folds, seed)))

        per["group_split"].append(
            roc_auc_score(y, _fit_predict(X, y, grp_folds, seed)))
        if verbose:
            print(f"  seed {seed} done", flush=True)

    rng = np.random.default_rng(0)
    base = float(np.mean(per["full"]))
    rows = []
    for k, v in per.items():
        m = float(np.mean(v))
        lo, hi = _boot_ci(np.array(v) - base, rng)
        # Verdicts are referenced to CHANCE as well as to the full model. A
        # control that stays far above 0.5 is a finding in its own right even
        # when it sits below the full model: it means a signal-free or
        # single-variable arm already reproduces much of the reported score.
        excess = m - 0.5
        share = excess / max(base - 0.5, 1e-9)
        if k == "full":
            verdict = "reference"
        elif k == "permutation":
            verdict = ("FAIL: score survives label permutation "
                       f"({share:.0%} of the excess over chance retained)"
                       if share > 0.25 else "pass")
        elif k == "target_blind":
            verdict = ("FAIL: target representation is largely redundant "
                       f"({share:.0%} of the excess over chance retained)"
                       if share > 0.25 else "pass")
        elif k == "degree_only":
            verdict = ("FAIL: annotation degree alone reproduces "
                       f"{share:.0%} of the excess over chance"
                       if share > 0.25 else "pass")
        else:
            verdict = ("WARN: large drop under grouped splitting"
                       if m < base - 0.10 else "pass")
        rows.append(dict(control=k, auc=m, delta=m - base, lo=lo, hi=hi,
                         verdict=verdict))

    if target_features is not None:
        M = target_features.values.astype(float)
        M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        S = M @ M.T
        np.fill_diagonal(S, 0.0)
        n_dup = int((S > 0.999).any(1).sum())
        rows.append(dict(control="duplicate_target", auc=float("nan"),
                         delta=float(n_dup), lo=float("nan"), hi=float("nan"),
                         verdict=(f"FAIL: {n_dup} targets share an identical "
                                  "vector with another target"
                                  if n_dup else "pass")))
    return AuditReport(rows)
