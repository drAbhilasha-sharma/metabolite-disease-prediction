"""Degree-matched negative sampling.

For each positive metabolite of disease d, a negative is drawn from the
metabolites that (i) are annotated to at least one other disease, (ii) are not
annotated to d, and (iii) have the same annotation degree (number of annotated
diseases) as the positive. The two classes are therefore matched on annotation
breadth, and a predictor using degree alone scores at chance.

Sampling is without replacement within a disease while unused candidates
remain. When no candidate of the exact degree exists, the nearest available
degree is used (ties broken toward the higher degree) and the fallback is
counted. degree_strata() is kept for diagnostics and is not used by the
matching rule.
"""
import numpy as np
import pandas as pd

N_STRATA = 5


def degree_strata(degree: pd.Series, n_strata: int = N_STRATA) -> pd.Series:
    """Assign each metabolite to a breadth stratum by quantile."""
    ranks = degree.rank(method="first", pct=True)
    return pd.Series(np.minimum((ranks * n_strata).astype(int), n_strata - 1),
                     index=degree.index, name="stratum")


def degree_matched_pairs(pos_df, rng, n_strata: int = N_STRATA, report=False):
    """Build a label table with degree-matched negatives.

    pos_df : DataFrame with columns [hmdb_id, disease] - the positives.
    Returns DataFrame [hmdb_id, disease, label] plus fallback diagnostics.
    """
    degree = pos_df.groupby("hmdb_id").disease.nunique()
    strat = degree_strata(degree, n_strata)
    by_stratum = {s: np.array(sorted(g.index)) for s, g in strat.groupby(strat)}
    pos_sets = {d: set(g.hmdb_id) for d, g in pos_df.groupby("disease")}

    # exact-degree index, so a positive of degree k prefers a negative of degree k
    by_degree = {k: np.array(sorted(g.index)) for k, g in degree.groupby(degree)}
    deg_vals = np.array(sorted(by_degree))

    rows, fallbacks, exhausted = [], 0, 0
    for d, g in pos_df.groupby("disease"):
        banned = pos_sets[d]
        used = set()
        for mid in g.hmdb_id.values:
            rows.append((mid, d, 1))
            k0 = int(degree.get(mid, 0))
            # nearest available degree, preferring exact; reuse allowed only
            # after unused candidates are exhausted, so matching never fails
            drawn, exact = None, True
            for k in sorted(deg_vals, key=lambda k: (abs(k - k0), -k)):
                pool = [m for m in by_degree[k] if m not in banned]
                if not pool:
                    continue
                fresh = [m for m in pool if m not in used]
                drawn = (fresh if fresh else pool)[rng.integers(len(fresh) if fresh else len(pool))]
                exact = (k == k0)
                break
            if drawn is None:
                exhausted += 1
                continue
            if not exact:
                fallbacks += 1
            used.add(drawn)
            rows.append((drawn, d, 0))

    out = pd.DataFrame(rows, columns=["hmdb_id", "disease", "label"])
    if report:
        dd = out.assign(degree=out.hmdb_id.map(degree).fillna(0))
        print(f"  pairs={len(out)} fallback_draws={fallbacks} exhausted={exhausted}")
        print(f"  mean degree: pos {dd[dd.label==1].degree.mean():.2f} "
              f"neg {dd[dd.label==0].degree.mean():.2f}")
    return out
