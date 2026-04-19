"""
libgeofin.weights
=================
Spatial weight matrix construction utilities (open source · MIT)

Provides:
  - haversine_km       : Great-circle distance calculation
  - gaussian_kernel    : Gaussian (RBF) kernel function
  - build_weight_matrix: Row-standardised spatial weight matrix (Gaussian or inverse-distance kernel)
"""

from __future__ import annotations

import math
from typing import List, Literal, Tuple

import numpy as np

__all__ = ["haversine_km", "gaussian_kernel", "build_weight_matrix"]


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Haversine formula: great-circle distance between two points on Earth's surface (km).

    Parameters
    ----------
    lat1, lng1 : Latitude and longitude of the first point (decimal degrees)
    lat2, lng2 : Latitude and longitude of the second point (decimal degrees)

    Returns
    -------
    float : Distance in kilometres
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def gaussian_kernel(distance_km: float, bandwidth_km: float) -> float:
    """
    Gaussian (radial basis function) kernel.

    Parameters
    ----------
    distance_km  : Distance between two points (km)
    bandwidth_km : Bandwidth parameter h (km)

    Returns
    -------
    float : Kernel weight in the range (0, 1]
    """
    return math.exp(-0.5 * (distance_km / bandwidth_km) ** 2)


def build_weight_matrix(
    points: List[Tuple[float, float]],
    bandwidth_km: float = 2.0,
    kernel: Literal["gaussian", "inverse_distance"] = "gaussian",
    row_standardize: bool = True,
) -> np.ndarray:
    """
    Build a spatial weight matrix W.

    Parameters
    ----------
    points          : List of coordinate tuples in [(lat, lng), ...] form
    bandwidth_km    : Bandwidth in kilometres (Gaussian kernel) or cut-off radius (inverse distance)
    kernel          : 'gaussian' (default) or 'inverse_distance'
    row_standardize : Whether to row-standardise W (default True)

    Returns
    -------
    np.ndarray : Spatial weight matrix of shape (n, n); returns shape (0, 0) when n=0
    """
    n = len(points)
    if n == 0:
        return np.empty((0, 0))

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
            if kernel == "gaussian":
                W[i, j] = gaussian_kernel(d, bandwidth_km)
            elif kernel == "inverse_distance":
                W[i, j] = 1.0 / max(d, 1e-6)

    if row_standardize:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums

    return W
