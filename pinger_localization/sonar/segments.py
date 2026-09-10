"""
Exact n-segment polyline fit over an ordered chain of trace samples (vendored).

Vendored (verbatim) from sonar_pipeline_experiment/fit_segments.py -- this is
the "splitting" step that produces the segmented pipe model whose VERTICES the
node publishes as poses.  Pure numpy.

An n-segment fit returns `n+1` vertices; vertices are a subset of the ordered
input samples and both ends are always included (the model spans the whole
trace, seed-side -> far end).  `robust_trim > 0` ignores the worst-fitting
fraction of samples per chord, so short bright-noise excursions the ridge DP
briefly locked onto do not steer the segment breakpoints.
"""

import numpy as np

# Robust-fit knob: per chord, ignore the worst-fitting fraction of trace
# samples (matches fit_segments.ROBUST_TRIM).
ROBUST_TRIM = 0.15


def seg_cost_table(pts, robust_trim=0.0):
    """C[i,j] (i<j): cost of approximating pts[i..j] with chord i->j.

    Uses clipped point-to-SEGMENT distance.  When `robust_trim` > 0, the
    worst-fitting fraction of samples (the largest residuals) are dropped from
    each chord sum before summing."""
    m = len(pts)
    C = np.zeros((m, m))
    for i in range(m - 1):
        a = pts[i]
        for j in range(i + 1, m):
            d = pts[j] - a
            L2 = d @ d
            if L2 < 1e-12:
                C[i, j] = 0.0
                continue
            w = pts[i:j + 1] - a
            t = np.clip((w @ d) / L2, 0.0, 1.0)
            r2 = ((w - t[:, None] * d) ** 2).sum(axis=1)
            if robust_trim > 0 and len(r2) > 4:
                k = int(np.ceil(len(r2) * robust_trim))
                r2 = np.sort(r2)[:-k]
            C[i, j] = float(r2.sum())
    return C


def fit_from_cost(poly, C, nseg):
    """Globally optimal n-segment polyline through ordered `poly` given a
    precomputed cost table.  Vertices are a subset of trace samples (both ends
    always included).  Returns (model Nx2, vertex indices) with nseg+1 rows."""
    m = len(poly)
    if nseg >= m:
        return poly.copy(), np.arange(m)
    INF = np.inf
    dp = np.full((nseg + 1, m), INF)
    par = np.zeros((nseg + 1, m), dtype=np.int64)
    for j in range(1, m):                      # 1 segment: 0..j
        dp[1, j] = C[0, j]
    for s in range(2, nseg + 1):
        for j in range(s, m):                  # need >= s points under 0..j
            seg = dp[s - 1, s - 1:j] + C[s - 1:j, j]
            k = int(np.argmin(seg))
            dp[s, j], par[s, j] = seg[k], (s - 1) + k
    j = m - 1
    idx = [j]
    for s in range(nseg, 1, -1):
        j = par[s, j]
        idx.append(j)
    idx = [0] + idx[::-1]          # dp[1][i] starts the first segment at 0
    return poly[idx], np.asarray(idx)
