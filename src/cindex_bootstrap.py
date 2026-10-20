"""Bootstrap CIs and paired tests for Harrell C-index.

Per the Stage-0 brief: the honest fold std (0.049) is large because each val
fold has only ~30 events. Point estimates of mean C-index understate this
uncertainty. Stage 3+ uses these utilities to:

- Report per-fold C-index with a 95% CI on the patient sample.
- Report fold-mean C-index with a 95% CI on the cohort.
- Compare GNN-vs-Cox via paired bootstrap on identical patient predictions.

All C-indices use sksurv.metrics.concordance_index_censored (Harrell's C, with
sksurv's tie handling) for consistency with Stage 0.
"""
from __future__ import annotations

import numpy as np
from sksurv.metrics import concordance_index_censored


def cindex(T: np.ndarray, E: np.ndarray, risk: np.ndarray) -> float:
    """Single point estimate. risk = log-hazard or any score where higher = worse outcome."""
    return float(concordance_index_censored(E.astype(bool), T, risk)[0])


def bootstrap_cindex(
    T: np.ndarray,
    E: np.ndarray,
    risk: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Patient-level bootstrap CI for Harrell C on (T, E, risk).

    Resamples val patients with replacement, recomputes C, returns mean / std /
    quantile-based CI. n_boot=1000 is enough for 95% CI to ~0.5 c-index points.

    Returns dict with point, mean, std, ci_low, ci_high, alpha, n_boot.
    """
    rng = np.random.default_rng(seed)
    n = len(T)
    point = cindex(T, E, risk)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Need at least 1 event in the resample for C-index to be defined.
        if E[idx].sum() < 1:
            continue
        try:
            samples.append(cindex(T[idx], E[idx], risk[idx]))
        except (ValueError, ZeroDivisionError):
            continue
    samples = np.asarray(samples)
    return {
        "point": point,
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "ci_low": float(np.quantile(samples, alpha / 2)),
        "ci_high": float(np.quantile(samples, 1 - alpha / 2)),
        "alpha": alpha,
        "n_boot": n_boot,
        "n_valid": int(len(samples)),
    }


def paired_bootstrap_delta(
    T: np.ndarray,
    E: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired bootstrap on identical patients: tests Δ = C(a) - C(b).

    For Stage 3+ comparisons. Identical resampled patient set is scored under
    both models; the difference distribution gives a proper paired-test CI on
    the model comparison.

    Returns dict with delta_point, delta_mean, delta_std, delta_ci_low,
    delta_ci_high, p_one_sided (fraction of bootstrap samples where Δ <= 0).
    """
    rng = np.random.default_rng(seed)
    n = len(T)
    point_a = cindex(T, E, risk_a)
    point_b = cindex(T, E, risk_b)
    delta_point = point_a - point_b
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if E[idx].sum() < 1:
            continue
        try:
            ca = cindex(T[idx], E[idx], risk_a[idx])
            cb = cindex(T[idx], E[idx], risk_b[idx])
            deltas.append(ca - cb)
        except (ValueError, ZeroDivisionError):
            continue
    deltas = np.asarray(deltas)
    return {
        "cindex_a_point": point_a,
        "cindex_b_point": point_b,
        "delta_point": delta_point,
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "delta_ci_low": float(np.quantile(deltas, alpha / 2)),
        "delta_ci_high": float(np.quantile(deltas, 1 - alpha / 2)),
        "p_a_le_b": float((deltas <= 0).mean()),  # one-sided: P(a <= b)
        "alpha": alpha,
        "n_boot": n_boot,
        "n_valid": int(len(deltas)),
    }
