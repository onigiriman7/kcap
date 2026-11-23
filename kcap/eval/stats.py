import numpy as np


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0):
    """Mean of (b - a) with a 95% CI, resampling paired units."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bootstrap_ci(b - a, n_boot, seed)
