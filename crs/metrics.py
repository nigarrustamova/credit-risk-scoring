"""Credit-risk model metrics: discrimination (AUC, Gini, KS, PR-AUC), calibration (Brier, ECE) and stability (PSI)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks(y, p) -> float:
    """Kolmogorov-Smirnov: the largest gap between the score distributions of good and bad borrowers."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins: how far predicted PD is from the observed default rate."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    err = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            err += len(idx) * abs(p[idx].mean() - y[idx].mean())
    return float(err / len(y))


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index between two score samples (bins from the expected sample's deciles).
    < 0.1 stable, 0.1-0.25 some shift, > 0.25 the population changed."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def report(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": round(float(auc), 4), "gini": round(float(2 * auc - 1), 4), "ks": round(ks(y, p), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 5),
            "ece": round(ece(y, p), 5), "default_rate": round(float(np.mean(y)), 4), "n": int(len(y))}
