"""Calibration, discrimination and decision-threshold behaviour.

The manuscript reports AUROC and AP, which are threshold-free. A method
proposed for screening also needs to be usable at a threshold, so this script
characterises the pooled out-of-fold predictions three further ways:

  * reliability -- predicted probability against observed frequency, with the
    Brier score decomposed into calibration and refinement components;
  * discrimination -- ROC and precision-recall curves per arm, computed on the
    pooled prediction vector rather than averaged over per-disease curves;
  * decision utility -- precision, recall and enrichment over prevalence as a
    function of the fraction of candidates a screen would take forward.

Writes results/calibration.csv, results/roc_curves.csv, results/pr_curves.csv,
results/topk_enrichment.csv and results/discrimination_summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, brier_score_loss)

OOF = pd.read_parquet("results/definitive_oof.parquet")
BEST = "rf"
ARMS = ["full", "perm", "blind"]
REGIMES = ["lmo", "ldo"]
N_BINS = 12
GRID = np.linspace(0.001, 0.30, 60)


def pooled(regime, mode, model=BEST):
    """Average the out-of-fold probability across seeds for each pair, so each
    (disease, metabolite) contributes once."""
    s = OOF[(OOF["regime"] == regime) & (OOF["mode"] == mode)
            & (OOF["model"] == model)]
    return (s.groupby(["disease", "hmdb_id"])
             .agg(label=("label", "first"), p=("p", "mean")).reset_index())


def main():
    cal, roc_rows, pr_rows, topk, summ = [], [], [], [], []
    for regime in REGIMES:
        for mode in ARMS:
            g = pooled(regime, mode)
            y, p = g.label.values, g.p.values
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, p)
            ap = average_precision_score(y, p)
            brier = brier_score_loss(y, p)
            base = y.mean()
            # Brier skill against the prevalence-only forecast
            bss = 1.0 - brier / (base * (1 - base))
            summ.append(dict(regime=regime, mode=mode, n=len(y), prevalence=base,
                             auroc=auc, ap=ap, brier=brier, brier_skill=bss))

            frac_pos, mean_pred = calibration_curve(y, p, n_bins=N_BINS,
                                                    strategy="quantile")
            edges = np.quantile(p, np.linspace(0, 1, N_BINS + 1))
            counts = np.histogram(p, bins=edges)[0]
            for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
                cal.append(dict(regime=regime, mode=mode, bin=i, mean_pred=mp,
                                frac_pos=fp, n=int(counts[i]) if i < len(counts) else 0))

            fpr, tpr, _ = roc_curve(y, p)
            k = max(1, len(fpr) // 400)
            for a, b in zip(fpr[::k], tpr[::k]):
                roc_rows.append(dict(regime=regime, mode=mode, fpr=a, tpr=b))

            pre, rec, _ = precision_recall_curve(y, p)
            k = max(1, len(pre) // 400)
            for a, b in zip(rec[::k], pre[::k]):
                pr_rows.append(dict(regime=regime, mode=mode, recall=a, precision=b))

            order = np.argsort(-p)
            ys = y[order]
            cum = np.cumsum(ys)
            for f in GRID:
                k2 = max(1, int(round(f * len(ys))))
                prec = cum[k2 - 1] / k2
                topk.append(dict(regime=regime, mode=mode, frac=f, k=k2,
                                 precision=prec, recall=cum[k2 - 1] / ys.sum(),
                                 enrichment=prec / base))

    pd.DataFrame(cal).to_csv("results/calibration.csv", index=False)
    pd.DataFrame(roc_rows).to_csv("results/roc_curves.csv", index=False)
    pd.DataFrame(pr_rows).to_csv("results/pr_curves.csv", index=False)
    pd.DataFrame(topk).to_csv("results/topk_enrichment.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv("results/discrimination_summary.csv", index=False)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
