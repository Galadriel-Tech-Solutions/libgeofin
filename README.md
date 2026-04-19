# libgeofin

**Open-source spatial statistics library for geo-financial analysis** · MIT License

libgeofin is the open-source core algorithm library of the GeoFinance Insights project. Financial institutions and academic researchers are free to audit, reproduce, and use the spatial statistics formulae it provides.

## Installation

```bash
pip install libgeofin
# or install from source
pip install -e .
```

## Quick Start

```python
import numpy as np
from libgeofin import build_weight_matrix, compute_moran_i, lisa_quadrant

# 1. Build a spatial weight matrix
points = [(31.22, 121.47), (31.23, 121.48), (31.21, 121.46)]
W = build_weight_matrix(points, bandwidth_km=2.0)

# 2. Compute Moran's I
values = np.array([10.0, 12.0, 8.0])
moran_i, z_score, p_value = compute_moran_i(values, W)
print(f"Moran's I = {moran_i:.4f}, p = {p_value:.4f}")

# 3. LISA quadrant classification
spatial_lag = float(W[0] @ values)
quadrant = lisa_quadrant(
    local_val=values[0],
    spatial_lag=spatial_lag,
    global_mean=values.mean(),
    p_value=p_value,
)
print(f"LISA quadrant: {quadrant}")  # HH / LL / LH / HL / NS
```

## Modules

| Module                      | Function                    | Description                                    |
| --------------------------- | --------------------------- | ---------------------------------------------- |
| `libgeofin.weights`         | `haversine_km`              | Haversine great-circle distance (km)           |
| `libgeofin.weights`         | `gaussian_kernel`           | Gaussian kernel function                       |
| `libgeofin.weights`         | `build_weight_matrix`       | Row-standardised spatial weight matrix         |
| `libgeofin.autocorrelation` | `compute_moran_i`           | Global Moran's I + Z-test                      |
| `libgeofin.autocorrelation` | `lisa_quadrant`             | LISA quadrant classification                   |
| `libgeofin.normalize`       | `standardize`               | Z-score standardisation                        |
| `libgeofin.normalize`       | `coords_to_distance_matrix` | Lat/lng coordinates → pairwise distance matrix |

## Formula Reference

**Moran's I:**

$$I = \frac{n}{S_0} \cdot \frac{\mathbf{y}^\top W \mathbf{y}}{\mathbf{y}^\top \mathbf{y}}$$

**Spatially adjusted rent (implemented in the closed-source GeoFinance Pro layer):**

$$R_{adj} = R_{base} \cdot (1 + \rho \cdot Wy)$$

## Scope of Open-Source Release

This library contains only the **pure mathematical / statistical components**. The following belong to the closed-source GeoFinance Pro commercial platform and are not included here:

- Moran's I → rent-growth-rate conversion engine
- Dynamic IRR / NPV financial calculator
- POI data crawler and API integrations
- Automated investment research report generator
