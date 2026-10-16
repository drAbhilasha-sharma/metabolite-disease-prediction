"""
Evaluation protocol for the rebuilt metabolite-disease association study.

This module encodes the design fixes for audit findings F2/F3/F6.

F2 fix - OPEN-WORLD CANDIDATE UNIVERSE
    The original universe contained only metabolites that were positive for at
    least one modelled disease, so "negative" meant "positive for a different
    disease". Here the candidate pool additionally contains HMDB metabolites
    that are detected in humans but carry NO disease annotation. Negatives are
    drawn from that unlabelled pool, so the model is asked the intended
    question: is this metabolite associated with this disease at all?

F3 fix - POWER
    90 diseases instead of 4. Ablation/comparison tests use disease as the unit
    of analysis with n=90, which reaches power ~1.0 for d=0.8 (vs 0.20 before).

F6 fix - NO PSEUDOREPLICATION
    Seeds are repeated measurements of the same diseases, not independent units.
    All inferential tests aggregate over seeds FIRST (per-disease mean), then
    test across diseases. Seed variability is reported as a stability statistic,
    never as sample size.

LEAKAGE CONTROL
    Splits are by METABOLITE, not by pair. A metabolite appears in exactly one
    fold across all diseases, so a metabolite seen in training cannot reappear
    as a test item for another disease. Negative sampling happens inside the
    fold, using only that fold's training metabolites.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def build_labelled_pairs(assoc_df, unlabelled_ids, neg_ratio, rng):
    """Build (disease, metabolite, label) rows with negatives from the UNLABELLED pool.

    assoc_df: columns [disease, hmdb_id] - the positives.
    unlabelled_ids: array of metabolite ids with NO disease annotation anywhere.
    neg_ratio: negatives per positive, per disease.
    """
    unl = np.asarray(unlabelled_ids)
    out = []
    for dis, grp in assoc_df.groupby("disease"):
        pos = grp.hmdb_id.unique()
        n_neg = int(round(len(pos) * neg_ratio))
        n_neg = min(n_neg, len(unl))
        neg = rng.choice(unl, size=n_neg, replace=False)
        for m in pos:
            out.append((dis, m, 1))
        for m in neg:
            out.append((dis, m, 0))
    return pd.DataFrame(out, columns=["disease", "hmdb_id", "label"])


def metabolite_grouped_folds(pairs, n_splits, seed):
    """Yield (train_idx, test_idx) where no metabolite spans train and test.

    GroupKFold is deterministic; we shuffle group->fold assignment with `seed`
    so different seeds give genuinely different partitions.
    """
    mets = pairs.hmdb_id.values
    uniq = np.unique(mets)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {m: perm[i] % n_splits for i, m in enumerate(uniq)}
    f = np.array([fold_of[m] for m in mets])
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]


def disease_grouped_folds(pairs, diseases, n_splits, seed, sibling_of=None):
    """Leave-diseases-out: hold out whole diseases (extrapolation setting).

    sibling_of : optional dict disease -> group key. Diseases that map to the
      same vocabulary descriptor have IDENTICAL disease feature vectors, so
      splitting them across the train/test boundary leaks the held-out
      representation into training and inflates the extrapolation score. When
      supplied, whole sibling groups are assigned to a fold together, making the
      held-out disease representation genuinely unseen.

      In this dataset 11 of 71 diseases (15.5%) fall into 4 such groups, e.g.
      the congenital adrenal hyperplasia variants all map to one descriptor.
    """
    rng = np.random.default_rng(seed)
    dz = np.array(sorted(diseases))
    if sibling_of is None:
        keys = list(dz)
    else:
        keys = sorted({sibling_of.get(d, d) for d in dz})
    perm = rng.permutation(len(keys))
    fold_of_key = {k: perm[i] % n_splits for i, k in enumerate(keys)}
    if sibling_of is None:
        fold_of = {d: fold_of_key[d] for d in dz}
    else:
        fold_of = {d: fold_of_key[sibling_of.get(d, d)] for d in dz}
    f = np.array([fold_of[d] for d in pairs.disease.values])
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]
