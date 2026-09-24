"""Binary detection metrics with small-sample honesty (Wilson intervals, bootstrap AUC intervals, n shown everywhere)."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Highest TPR achievable while keeping FPR <= target."""
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def fpr_at_tpr(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Lowest FPR achievable while keeping TPR >= target."""
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr >= target - 1e-12
    return float(fpr[ok].min()) if ok.any() else 1.0


def bootstrap_auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Stratified bootstrap (resample positives and negatives separately) 95% interval for ROC-AUC."""
    rng = np.random.default_rng(seed)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return (math.nan, math.nan)
    aucs = []
    for _ in range(n_boot):
        p = rng.choice(pos, len(pos))
        n = rng.choice(neg, len(neg))
        aucs.append(_auc_mw(p, n))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def _auc_mw(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney form of ROC-AUC (ties count half), fast enough to bootstrap a few thousand points."""
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_metrics(y: np.ndarray, s: np.ndarray, threshold: float, cfg_eval: dict, seed: int, n_boot: int = 0) -> dict:
    """All headline metrics for one (system, slice). Metrics needing both classes are NaN when one is absent."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pred = s >= threshold
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    out = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "TPR": tp / n_pos if n_pos else math.nan,
        "FPR": fp / n_neg if n_neg else math.nan,
        "TPR_CI": wilson(tp, n_pos),
        "FPR_CI": wilson(fp, n_neg),
        "precision": tp / (tp + fp) if (tp + fp) else math.nan,
    }
    # Dataset prevalence is ~90% attacks, so plain precision mostly reflects the class ratio; also report it at an assumed low prevalence.
    prev = cfg_eval["assumed_prevalence"]
    tpr, fpr = out["TPR"], out["FPR"]
    denom = tpr * prev + fpr * (1 - prev) if not (math.isnan(tpr) or math.isnan(fpr)) else math.nan
    out["precision_at_prev"] = (tpr * prev / denom) if (denom and not math.isnan(denom)) else math.nan
    p, r = out["precision"], out["TPR"]
    out["F1"] = 2 * p * r / (p + r) if (n_pos and not math.isnan(p) and (p + r) > 0) else math.nan
    both = n_pos > 0 and n_neg > 0
    out["ROC_AUC"] = float(roc_auc_score(y, s)) if both else math.nan
    out["PR_AUC"] = float(average_precision_score(y, s)) if both else math.nan
    out["AUC_CI"] = bootstrap_auc_ci(y, s, n_boot, seed) if (both and n_boot) else (math.nan, math.nan)
    for f in cfg_eval["fixed_fprs"]:
        out[f"TPR@FPR{int(f * 100)}%"] = tpr_at_fpr(y, s, f) if both else math.nan
    for t in cfg_eval["fixed_tprs"]:
        out[f"FPR@TPR{int(t * 100)}%"] = fpr_at_tpr(y, s, t) if both else math.nan
    return out
