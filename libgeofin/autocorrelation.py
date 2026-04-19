"""
libgeofin.autocorrelation
=========================
Spatial autocorrelation statistics (open source · MIT)

Provides:
  - compute_moran_i : Global Moran's I with significance test
  - lisa_quadrant   : LISA quadrant classification (HH / LL / LH / HL / NS)
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np
from scipy import stats

__all__ = ["compute_moran_i", "lisa_quadrant"]

LISAQuadrant = Literal["HH", "LL", "LH", "HL", "NS"]


def compute_moran_i(
    values: np.ndarray,
    W: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute global Moran's I with a two-tailed normal-approximation significance test.

    Parameters
    ----------
    values : Attribute value array of length n (aligned with rows/columns of W)
    W      : Spatial weight matrix of shape (n, n) (typically row-standardised)

    Returns
    -------
    (moran_i, z_score, p_value) : Moran's I statistic, Z-score, two-tailed p-value

    Notes
    -----
    Formula:
        I = (n / S₀) · (yᵀ W y) / (yᵀ y)
    where y = values - mean(values) and S₀ = ΣΣ w_{ij}.
    Variance uses the analytical normal-approximation formula (Cliff & Ord, 1981).
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0, 1.0

    y = values - values.mean()
    y_var = float(np.dot(y, y))
    if y_var == 0:
        return 0.0, 0.0, 1.0

    S0 = float(W.sum())
    if S0 == 0:
        return 0.0, 0.0, 1.0

    numerator = n * float(np.dot(y, W @ y))
    moran = numerator / (S0 * y_var)

    # -- Variance (Cliff & Ord normal approximation) --
    E_I = -1.0 / (n - 1)
    S1 = 0.5 * float(np.sum((W + W.T) ** 2))
    S2 = float(np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2))
    k = (n * float(np.sum(y ** 4))) / y_var ** 2

    var_I_num = (
        n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
        - k * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    )
    var_I_den = (n - 1) * (n + 1) * (n + 2) * S0 ** 2
    var_I = var_I_num / var_I_den if var_I_den != 0 else 1e-6

    std_I = math.sqrt(abs(var_I)) if var_I > 0 else 1e-6
    z = (moran - E_I) / std_I
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    return float(moran), float(z), p


def lisa_quadrant(
    local_val: float,
    spatial_lag: float,
    global_mean: float,
    p_value: float,
    significance: float = 0.05,
) -> LISAQuadrant:
    """
    Classify the LISA quadrant from the Moran scatterplot.

    Parameters
    ----------
    local_val    : Attribute value of the target observation (standardised)
    spatial_lag  : Spatial lag of the target point (the corresponding row of W·y)
    global_mean  : Global mean (used as quadrant origin)
    p_value      : Significance p-value
    significance : Significance threshold (default 0.05)

    Returns
    -------
    LISAQuadrant : "HH" | "LL" | "LH" | "HL" | "NS"

    Notes
    -----
    Moran scatterplot quadrants:
      · HH (top-right)   : local high, neighbours high  → positive clustering, commercial premium core
      · LL (bottom-left) : local low,  neighbours low   → negative clustering, value trough
      · LH (top-left)    : local low,  neighbours high  → spatial cold-spot, catch-up potential
      · HL (bottom-right): local high, neighbours low   → spatial outlier, overheating risk
      · NS               : not statistically significant
    """
    if p_value > significance:
        return "NS"
    if local_val >= global_mean and spatial_lag >= global_mean:
        return "HH"
    if local_val < global_mean and spatial_lag < global_mean:
        return "LL"
    if local_val < global_mean and spatial_lag >= global_mean:
        return "LH"
    return "HL"
