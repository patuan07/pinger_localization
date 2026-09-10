"""
Contrast-grid rasterizer + scorer for 2D sonar point clouds (vendored).

Vendored (verbatim) from sonar_pipeline_experiment/voter.py, trimmed to just
what the seed-traced detector needs: `build_grids` and `Scorer`. Pure
numpy / scipy only (scipy.ndimage.gaussian_filter / map_coordinates).

Point-cloud data coordinates throughout: X = range (m), Y = lateral (m),
contrast = intensity minus the per-range-bin median (range normalization).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
RANGE_BIN      = 0.15   # m, for range-dependent intensity normalization
GRID_RES       = 0.03   # m/pixel
SMOOTH_SIGMA_PX = 0.8


def build_grids(X, Y, contrast):
    """Rasterize `contrast` (+ point density) onto a smoothed grid.

    Returns (contrast_grid, density_grid, meta) where meta has xmin/ymin/res
    (needed by Scorer.w2p)."""
    xmin, ymin = X.min(), Y.min()
    nx = int((X.max() - xmin) / GRID_RES) + 2
    ny = int((Y.max() - ymin) / GRID_RES) + 2
    ix = ((X - xmin) / GRID_RES).astype(int)
    iy = ((Y - ymin) / GRID_RES).astype(int)
    sum_grid = np.zeros((nx, ny))
    cnt_grid = np.zeros((nx, ny))
    np.add.at(sum_grid, (ix, iy), contrast)
    np.add.at(cnt_grid, (ix, iy), 1)
    sum_s = gaussian_filter(sum_grid, SMOOTH_SIGMA_PX)
    cnt_s = gaussian_filter(cnt_grid, SMOOTH_SIGMA_PX)
    contrast_grid = np.divide(sum_s, cnt_s, out=np.zeros_like(sum_s),
                              where=cnt_s > 1e-6)
    density_grid = cnt_s
    meta = dict(xmin=xmin, ymin=ymin, res=GRID_RES)
    return contrast_grid, density_grid, meta


class Scorer:
    """Sample contrast / density off a built grid at data-coordinate points."""

    def __init__(self, contrast_grid, density_grid, meta):
        self.cg = contrast_grid
        self.dg = density_grid
        self.xmin, self.ymin, self.res = meta['xmin'], meta['ymin'], meta['res']

    def w2p(self, x, y):
        return (x - self.xmin) / self.res, (y - self.ymin) / self.res

    def _sample(self, xs, ys, min_density):
        px, py = self.w2p(xs, ys)
        coords = np.stack([px.ravel(), py.ravel()])
        c = map_coordinates(self.cg, coords, order=1, mode='constant',
                            cval=0.0).reshape(xs.shape)
        d = map_coordinates(self.dg, coords, order=1, mode='constant',
                            cval=0.0).reshape(xs.shape)
        covered = d > min_density
        cov_frac = covered.mean(axis=1)
        c_sum = np.where(covered, c, 0).sum(axis=1)
        c_cnt = np.maximum(covered.sum(axis=1), 1)
        return c_sum / c_cnt, cov_frac

    def point_contrast(self, x, y):
        px, py = self.w2p(np.array([x]), np.array([y]))
        c = map_coordinates(self.cg, np.stack([px, py]), order=1,
                            mode='constant', cval=0.0)
        return c[0]
