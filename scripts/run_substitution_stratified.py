"""Does substitution cost depend on how far the substituted disease is?

The three schemes of run_substitution_mi.py (nearest / farthest / random
derangement) differ in two respects at once, not one:

  distance       the mean representation distance between a disease and the
                 disease that replaces it;
  map structure  nearest and farthest are many-to-one maps -- 71 diseases are
                 sent onto 48 and 27 distinct targets respectively, and 20 (9)
                 of them are mutual swaps in which a pair simply exchanges
                 vectors -- whereas a derangement is a bijection, so the
                 marginal distribution of disease vectors is preserved exactly
                 and no two diseases are merged.

Comparing the three therefore cannot isolate a distance effect: a difference
between them is equally explicable by how many diseases were collapsed onto a
shared vector. This module removes that confound by holding map structure fixed
and varying distance alone. Every substitution used here is a derangement --
bijective, no fixed points, identical marginals -- and derangements are sampled
into strata of increasing mean distance by simulated annealing on the
permutation. The distance effect is then estimated within the stratified family
by a per-seed regression of AUROC change on mean substitution distance.

Two quantities decide the mechanism:

  slope      how much AUROC is lost per unit of substitution distance. A model
             using disease biology predicts a steep slope: replacing a disease
             with a distant one should hurt far more than with a near one.
  intercept  the cost extrapolated to zero substitution distance -- what is
             lost by substituting a biologically near-identical disease. A
             model using the disease block as an identifier predicts that this
             is where most of the cost sits, because breaking the key costs the
             same however similar the replacement is.

Writes results/substitution_strata.csv (one row per seed x stratum x draw) and
results/substitution_slope.csv (per-seed slope and intercept).

MDA_STRAT_BLOCK=random repeats the analysis with the random Gaussian disease
block of the identifier control (run_idcontrol.py, arm rand_id): for seed s the
block is np.random.default_rng(700 + s).standard_normal((71, 52)) in float32,
rows in the sorted disease order of DFI. The model is trained on that block and
substitutions permute it; substitution distance (and so the strata targets) is
measured in the same block, per seed. Pairs, folds, strata, draws and annealing
are those of the real-block run. Outputs: results/substitution_strata_rand.csv
and results/substitution_slope_rand.csv. The default, MDA_STRAT_BLOCK=real,
is the real-block analysis.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalproto
from degree_matched import degree_matched_pairs
from run_definitive import fit_predict, make, NEEDS_SCALING
from run_substitution_mi import (DFI, DFv, FPI, FPX, MODEL, N_FOLDS, POS,
                                 derangement)

N_SEEDS = int(os.environ.get("MDA_STRAT_SEEDS", "10"))
N_DRAWS = int(os.environ.get("MDA_STRAT_DRAWS", "3"))
N_STRATA = 6
BLOCK = os.environ.get("MDA_STRAT_BLOCK", "real")
if BLOCK not in ("real", "random"):
    raise SystemExit(f"MDA_STRAT_BLOCK must be real or random, not {BLOCK!r}")
SUFFIX = "" if BLOCK == "real" else "_rand"


def disease_block(seed):
    """Disease block for one seed: the real profiles, or rand_id of run_idcontrol.py."""
    if BLOCK == "real":
        return DFv
    gen = np.random.default_rng(700 + seed)
    return gen.standard_normal((len(DFI), DFv.shape[1])).astype(np.float32)


def dist_matrix(M=None):
    """Cosine distance between disease representations."""
    M = DFv if M is None else M
    Dn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    return 1.0 - (Dn @ Dn.T)


def strata_targets(D):
    """Six targets spanning the reachable range of mean substitution distance."""
    off = D.copy()
    np.fill_diagonal(off, np.nan)
    lo = float(np.nanmean(np.nanmin(off, axis=1)))
    hi = float(np.nanmean(np.nanmax(off, axis=1)))
    return lo, hi, np.linspace(lo + 0.12 * (hi - lo), hi - 0.12 * (hi - lo), N_STRATA)


def mean_dist(D, perm):
    return float(np.mean(D[np.arange(len(perm)), perm]))


def anneal(D, target, rng, iters=20000, T0=0.05):
    """Sample a derangement whose mean substitution distance is near `target`.

    Starts from a uniform derangement and applies random transpositions of the
    permutation's images, accepting moves by Metropolis on |mean distance -
    target|. Transposing two images preserves bijectivity; moves that would
    create a fixed point are rejected, so the result is always a derangement.
    """
    n = len(D)
    p = derangement(n, rng)
    cur = abs(mean_dist(D, p) - target)
    for it in range(iters):
        i, j = rng.integers(0, n, 2)
        if i == j or p[i] == j or p[j] == i:
            continue  # would place a disease onto itself
        q = p.copy()
        q[i], q[j] = q[j], q[i]
        cand = abs(mean_dist(D, q) - target)
        T = T0 * (1.0 - it / iters) + 1e-6
        if cand < cur or rng.random() < np.exp(-(cand - cur) / T):
            p, cur = q, cand
        if cur < 1e-4:
            break
    assert not (p == np.arange(n)).any(), "fixed point in derangement"
    assert len(set(p.tolist())) == n, "not a bijection"
    return p


def main():
    # Span the reachable range of mean substitution distance. The lower end is
    # set by the nearest-neighbor map and the upper by the farthest map; a
    # bijection cannot reach either extreme, so strata are placed inside them.
    # The real block is the same for every seed; the random block is drawn per
    # seed, so its distances and targets are computed inside the seed loop.
    print(f"disease block: {BLOCK}", flush=True)
    if BLOCK == "real":
        D = dist_matrix()
        lo, hi, targets = strata_targets(D)
        print(f"nearest-map mean distance {lo:.3f}, farthest-map {hi:.3f}", flush=True)
        print("strata targets " + " ".join(f"{t:.3f}" for t in targets), flush=True)

    rows = []
    for seed in range(N_SEEDS):
        B = disease_block(seed)
        if BLOCK == "random":
            D = dist_matrix(B)
            lo, hi, targets = strata_targets(D)
            print(f"seed{seed} nearest-map {lo:.3f}, farthest-map {hi:.3f}; targets "
                  + " ".join(f"{t:.3f}" for t in targets), flush=True)
        rng = np.random.default_rng(100 + seed)
        P = degree_matched_pairs(POS[["hmdb_id", "disease"]], rng).reset_index(drop=True)
        mi = np.array([FPI[m] for m in P.hmdb_id])
        di = np.array([DFI[d] for d in P.disease])
        y = P.label.values
        X = np.hstack([FPX[mi], B[di]])
        folds = list(evalproto.metabolite_grouped_folds(P, N_FOLDS, seed))

        base = np.zeros(len(P))
        fitted = []
        for tr, te in folds:
            base[te] = fit_predict(MODEL, seed, X[tr], y[tr], X[te])
            sc = StandardScaler().fit(X[tr]) if MODEL in NEEDS_SCALING else None
            Xtr = sc.transform(X[tr]) if sc is not None else X[tr]
            fitted.append((make(MODEL, seed).fit(Xtr, y[tr]), sc, te))

        def per_dis_mean(pred):
            v = [roc_auc_score(g.label, g.p) for _, g in
                 P.assign(p=pred).groupby("disease") if g.label.nunique() > 1]
            return float(np.mean(v))

        def score_with(perm):
            pred = np.zeros(len(P))
            for clf, sc, te in fitted:
                Xs = np.hstack([FPX[mi[te]], B[perm[di[te]]]])
                Xs = sc.transform(Xs) if sc is not None else Xs
                pred[te] = clf.predict_proba(Xs)[:, 1]
            return per_dis_mean(pred)

        auc_base = per_dis_mean(base)
        for si, target in enumerate(targets):
            for k in range(N_DRAWS):
                g = np.random.default_rng(9000 + 1000 * seed + 10 * si + k)
                p = anneal(D, target, g)
                a = score_with(p)
                rows.append(dict(seed=seed, stratum=si, draw=k,
                                 target=target, dist=mean_dist(D, p),
                                 auc=a, delta=a - auc_base, base=auc_base))
        s = pd.DataFrame([r for r in rows if r["seed"] == seed])
        print(f"seed{seed} base={auc_base:.3f} " +
              " ".join(f"d={g.dist.mean():.2f}:{g.delta.mean():+.3f}"
                       for _, g in s.groupby("stratum")), flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(f"results/substitution_strata{SUFFIX}.csv", index=False)

    # Per-seed least-squares fit of delta on realized mean distance. Fitting
    # within seed keeps the paired structure: every seed contributes one slope
    # and one intercept, and the seed-to-seed spread carries the uncertainty.
    fits = []
    for seed, g in R.groupby("seed"):
        b, a = np.polyfit(g.dist, g.delta, 1)
        fits.append(dict(seed=seed, slope=b, intercept=a))
    F = pd.DataFrame(fits)
    F.to_csv(f"results/substitution_slope{SUFFIX}.csv", index=False)
    print("\nper-seed fit of AUROC change on substitution distance", flush=True)
    print(F.round(4).to_string(index=False), flush=True)
    print(f"\nslope     mean {F.slope.mean():+.4f}  sd {F.slope.std(ddof=1):.4f}", flush=True)
    print(f"intercept mean {F.intercept.mean():+.4f}  sd {F.intercept.std(ddof=1):.4f}", flush=True)
    print(f"wrote results/substitution_strata{SUFFIX}.csv, "
          f"results/substitution_slope{SUFFIX}.csv", flush=True)


if __name__ == "__main__":
    main()
