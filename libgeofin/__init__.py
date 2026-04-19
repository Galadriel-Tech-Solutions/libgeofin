"""
libgeofin
=========
Open-source spatial statistics library for financial geography analysis.

License : MIT
Homepage: https://github.com/geofinance-insights/libgeofin

Modules
-------
libgeofin.weights         Spatial weight matrix (Haversine distance + Gaussian kernel)
libgeofin.autocorrelation Global Moran's I & LISA quadrant classification
libgeofin.normalize       Coordinate normalisation and distance matrix
"""

from libgeofin.weights import haversine_km, gaussian_kernel, build_weight_matrix
from libgeofin.autocorrelation import compute_moran_i, lisa_quadrant
from libgeofin.normalize import standardize, coords_to_distance_matrix

__version__ = "0.1.0"
__all__ = [
    "haversine_km",
    "gaussian_kernel",
    "build_weight_matrix",
    "compute_moran_i",
    "lisa_quadrant",
    "standardize",
    "coords_to_distance_matrix",
]
