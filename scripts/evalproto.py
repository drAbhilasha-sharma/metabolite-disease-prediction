"""Shared pair construction and fold assignment.

  build_labeled_pairs       positives plus negatives drawn from the unannotated
                            pool (the conventional protocol, run_benchmark.py)
  metabolite_grouped_folds  folds grouped by metabolite: no metabolite appears
                            in both training and test (held-out metabolites)
  disease_grouped_folds     folds grouped by disease, with diseases that share
                            a MeSH descriptor kept in one fold (held-out
                            diseases)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def build_labeled_pairs(assoc_df, unlabeled_ids, neg_ratio, rng):
    """Build (disease, metabolite, label) rows with negatives from the unannotated pool.

    assoc_df: columns [disease, hmdb_id], the positives.
    unlabeled_ids: array of metabolite ids with no disease annotation.
    neg_ratio: negatives per positive, per disease.
    """
    unl = np.asarray(unlabeled_ids)
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

    The metabolite-to-fold assignment is shuffled with `seed`, so different
    seeds give different partitions.
    """
    mets = pairs.hmdb_id.values
    uniq = np.unique(mets)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {m: perm[i] % n_splits for i, m in enumerate(uniq)}
    f = np.array([fold_of[m] for m in mets])
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]


def disease_grouped_folds(pairs, diseases, n_splits, seed, group_of=None):
    """Folds grouped by disease (held-out diseases, extrapolation).

    group_of : optional dict disease -> group key. Diseases mapped to the same
      MeSH descriptor have identical disease feature vectors. When group_of is
      supplied, each group is assigned to a single fold, so no held-out disease
      representation is present in training. In this dataset 11 of the 71
      diseases fall into 4 such groups (for example, the congenital adrenal
      hyperplasia variants).
    """
    rng = np.random.default_rng(seed)
    dz = np.array(sorted(diseases))
    if group_of is None:
        keys = list(dz)
    else:
        keys = sorted({group_of.get(d, d) for d in dz})
    perm = rng.permutation(len(keys))
    fold_of_key = {k: perm[i] % n_splits for i, k in enumerate(keys)}
    if group_of is None:
        fold_of = {d: fold_of_key[d] for d in dz}
    else:
        fold_of = {d: fold_of_key[group_of.get(d, d)] for d in dz}
    f = np.array([fold_of[d] for d in pairs.disease.values])
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]
