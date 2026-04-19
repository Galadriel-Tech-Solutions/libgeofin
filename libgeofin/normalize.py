"""
libgeofin.normalize
===================
Coordinate and value normalisation utilities (open source · MIT)

Provides:
  - standardize               : Z-score standardisation
  - coords_to_distance_matrix : List of lat/lng coordinates → pairwise distance matrix (km)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from libgeofin.weights import haversine_km

__all__ = ["standardize", "coords_to_distance_matrix"]


def standardize(values: np.ndarray) -> np.ndarray:
    """
    Z-score standardisation.

    Parameters
    ----------
    values : Raw attribute value array (1-D)

    Returns
    -------
    np.ndarray : (values - mean) / std; returns an all-zero array when std == 0
    """
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def coords_to_distance_matrix(
    points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Convert a list of lat/lng coordinates to a pairwise Haversine distance matrix.

    Parameters
    ----------
    points : List of coordinate tuples in [(lat, lng), ...] form

    Returns
    -------
    np.ndarray : Symmetric distance matrix of shape (n, n) in kilometres; diagonal is 0
    """
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
            D[i, j] = d
            D[j, i] = d
    return D
