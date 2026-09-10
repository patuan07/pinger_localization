"""
Seed-traced pipe detector for 2D sonar point clouds (vendored).

Vendored from sonar_pipeline_experiment/trace.py (real-time oriented engine:
crop a small box around the known seed, find the straight-lead heading, unroll
a strip along it, extract the pipe ridge with a Viterbi DP, bound by the
signal, emit a dense polyline). Pure numpy / scipy.

Only change from the experiment copy: `prepare_roi` and `detect` take an
in-memory cloud ARRAY instead of a path, and the import of voter.py's grid
helpers points at the local `grid` module. Data coordinates throughout:
X = range, Y = lateral (identical to /oculus/pointcloud in frame auv5/sonar).

The trace runs OUTWARD from `seed` (the pipe terminus) toward +X.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from pinger_localization.sonar.grid import RANGE_BIN, Scorer, build_grids

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# ROI crop around the seed (data coords, meters).
ROI_XBACK_M    = 0.6    # behind the seed
ROI_XFWD_M     = 4.2    # outward (the pipe extends 2.6-3.3 m)
ROI_YHALF_M    = 2.4    # lateral half-width
ROI_MIN_PTS    = 500    # below this the ROI is empty -> report None

# Straight-lead heading search.
LEAD_LEN_M     = 1.25   # ray length for the lead probe
LEAD_NSAMP     = 13     # samples along each probe ray
NEAR_COV_GATE  = 0.4    # min coverage over the first ~0.4 m of the ray
NEAR_PROBE_M   = 0.4    # how far out the near-coverage probe reaches
DENSITY_MIN    = 0.3    # "there is a real return here" density bar

# Unrolled strip + ridge DP.
STRIP_SMAX_M   = 3.8    # max down-range along the lead heading
STRIP_DS_M     = 0.03   # along-track step (= 1 grid pixel)
STRIP_U_HALF_M = 0.8    # corridor half-width (GT stays within ~0.8 m of axis)
STRIP_DU_M     = 0.04   # lateral step
DP_SMOOTH_LAM  = 0.3    # quadratic lateral-jump penalty (m^-2)

# Signal bounds (start / far-end) + acceptance.
FLOOR_CONTRAST = 0.05   # "on the ridge" contrast bar (voter ISOLATION_THR style)
TRIM_GAP_M     = 0.25   # allow dim gaps this long before declaring the end
TRIM_REL       = 0.40   # start/end bar as a fraction of the trace's own median
                        # brightness (the seed and the strip cap are only coarse
                        # locators -- the trace should span where the sonar
                        # return actually is, and end where it falls off)
START_MIN_RUN_M = 0.15  # require this much sustained signal before the trace
                        # "starts" (skips a dim near-seed lead the seed pins)
MIN_TRACE_LEN_M = 1.0   # below this the trace is too short to trust

# Relaxed retry when the seed-side coverage gate finds no lead.
RELAX_NEAR_GATE = 0.6   # multiplier (mirrors voter.RELAX_ANCHOR_FACTOR)


def estimate_bin_medians(X, I, stride=8, min_per_bin=5):
    """Per-range-bin median intensity (the range normalization), computed on a
    stride subsample of the FULL cloud so the ROI crop cannot bias it."""
    s = slice(None, None, stride)
    Xs, Is = X[s], I[s]
    bins = np.arange(X.min(), X.max() + RANGE_BIN, RANGE_BIN)
    bi = np.clip(np.digitize(Xs, bins) - 1, 0, len(bins) - 1)
    med = np.empty(len(bins))
    fallback = np.median(Is)
    for b in range(len(bins)):
        m = (bi == b)
        med[b] = np.median(Is[m]) if m.sum() > min_per_bin else fallback
    return bins, med


def prepare_roi(cloud, seed):
    """Range-normalize one cloud array and build a contrast grid over the box
    around the seed.

    `cloud` is an (N, >=4) array [x, y, z, intensity] (x=range, y=lateral).
    Returns (scorer, X_roi, Y_roi) or (None, None, None) when the ROI is too
    empty to matter."""
    seed = np.asarray(seed, dtype=float)
    X = cloud[:, 0].astype(np.float64)
    Y = cloud[:, 1].astype(np.float64)
    I = cloud[:, 3].astype(np.float64)

    bins, med = estimate_bin_medians(X, I)
    m = ((X >= seed[0] - ROI_XBACK_M) & (X <= seed[0] + ROI_XFWD_M) &
         (Y >= seed[1] - ROI_YHALF_M) & (Y <= seed[1] + ROI_YHALF_M))
    if m.sum() < ROI_MIN_PTS:
        return None, None, None
    Xr, Yr, Ir = X[m], Y[m], I[m]
    bi = np.clip(np.digitize(Xr, bins) - 1, 0, len(bins) - 1)
    contrast = Ir - med[bi]
    cg, dg, meta = build_grids(Xr, Yr, contrast)
    return Scorer(cg, dg, meta), Xr, Yr


def _sample_points(scorer, xs, ys):
    """Raw bilinear samples of (contrast, density) grids at data-coord points."""
    px, py = scorer.w2p(xs, ys)
    coords = np.stack([px.ravel(), py.ravel()])
    c = map_coordinates(scorer.cg, coords, order=1, mode='constant',
                        cval=0.0).reshape(xs.shape)
    d = map_coordinates(scorer.dg, coords, order=1, mode='constant',
                        cval=0.0).reshape(xs.shape)
    return c, d


def _ray_means(scorer, seed, angles_deg, length, nsamp):
    """Mean covered contrast + coverage fraction of outward rays from `seed`.
    angle_deg = 0 points along +X (range), matching data coordinates."""
    th = np.radians(np.asarray(angles_deg, dtype=float))
    t = np.linspace(0.0, 1.0, nsamp)
    xs = seed[0] + np.cos(th[:, None]) * length * t[None, :]
    ys = seed[1] + np.sin(th[:, None]) * length * t[None, :]
    return scorer._sample(xs, ys, min_density=DENSITY_MIN)


def find_straight_lead(scorer, seed, near_gate=NEAR_COV_GATE):
    """Heading of the ~1.25 m straight lead out of the seed.

    Coarse fan over the outward half-plane (step 3 deg), gated on coverage
    within NEAR_PROBE_M of the seed (a bright echo not attached to the seed
    cannot win), then a +-3 deg refine at 0.5 deg.
    Returns (lead_deg, score, near_cov) or (None, None, None)."""
    seed = np.asarray(seed, dtype=float)
    coarse = np.arange(-89.0, 90.0, 3.0)
    mean_c, _ = _ray_means(scorer, seed, coarse, LEAD_LEN_M, LEAD_NSAMP)
    _, cov_near = _ray_means(scorer, seed, coarse, NEAR_PROBE_M, LEAD_NSAMP)
    ok = cov_near > near_gate
    if not ok.any():
        return None, None, None
    best = coarse[np.argmax(np.where(ok, mean_c, -np.inf))]

    fine = np.linspace(best - 3.0, best + 3.0, 13)
    mean_f, covf = _ray_means(scorer, seed, fine, LEAD_LEN_M, LEAD_NSAMP)
    okf = covf > near_gate
    if not okf.any():
        return None, None, None
    i = np.argmax(np.where(okf, mean_f, -np.inf))
    return float(fine[i]), float(mean_f[i]), float(covf[i])


def build_strip(scorer, seed, lead_deg):
    """Sample the contrast grid on a straight strip along `lead_deg`.

    Strip axes: `arc` = down-range along the lead heading [0, STRIP_SMAX_M],
    `offset` = lateral [+-STRIP_U_HALF_M].  Returns
    (contrast_strip [Ns x M], offsets_m, arcs_m, basis).  Density-masked."""
    seed = np.asarray(seed, dtype=float)
    th = np.radians(lead_deg)
    ux, uy = np.cos(th), np.sin(th)
    vx, vy = -np.sin(th), np.cos(th)
    arcs = np.arange(0.0, STRIP_SMAX_M + 1e-9, STRIP_DS_M)
    offs = np.arange(-STRIP_U_HALF_M, STRIP_U_HALF_M + 1e-9, STRIP_DU_M)
    A, O = np.meshgrid(arcs, offs, indexing='ij')          # Ns x M
    xs = seed[0] + A * ux + O * vx
    ys = seed[1] + A * uy + O * vy
    c, d = _sample_points(scorer, xs, ys)
    cs = np.where(d > DENSITY_MIN, c, 0.0)
    return cs, offs, arcs, (ux, uy, vx, vy)


def dp_ridge(contrast_strip, offsets_m):
    """Viterbi over the strip: the brightest smooth path of contrast.

    State = lateral offset index per down-range column.  Transition cost is
    quadratic in the lateral jump (meters).  Column 0 is pinned to the seed's
    lateral position (offset ~0): the seed is the pipe TERMINUS.
    Returns the offset index per column (len = Ns)."""
    Ns, M = contrast_strip.shape
    lam = DP_SMOOTH_LAM
    o = np.asarray(offsets_m, dtype=float)
    s0 = int(np.argmin(np.abs(o)))
    prev = np.full(M, -np.inf)
    prev[s0] = contrast_strip[0, s0]
    back = np.zeros((Ns - 1, M), dtype=np.int64)
    for s in range(1, Ns):
        # trans[i, j] = prev[i] - lam * (o_j - o_i)^2 ; take best previous i per j
        trans = prev[:, None] - lam * (o[None, :] - o[:, None]) ** 2
        best = trans.max(axis=0)
        back[s - 1] = trans.argmax(axis=0)
        prev = contrast_strip[s] + best
    path = np.empty(Ns, dtype=np.int64)
    path[-1] = int(np.argmax(prev))
    for s in range(Ns - 2, -1, -1):
        path[s] = back[s, path[s + 1]]
    return path


def trim_far_end(path_contrast, arcs_m, floor=FLOOR_CONTRAST, gap_m=TRIM_GAP_M):
    """Last down-range index to trust, walking back from the far end.  Kept for
    reference / comparison; signal_bounds() is the version trace_pipe uses."""
    Ns = len(path_contrast)
    gap_cols = max(int(round(gap_m / (arcs_m[1] - arcs_m[0]))), 1)
    sig = path_contrast > floor
    idx = np.where(sig)[0]
    if not len(idx):
        return 0
    last = int(idx[-1])
    return min(Ns - 1, last + gap_cols)


def signal_bounds(path_contrast, arcs_m, rel=TRIM_REL, floor=FLOOR_CONTRAST,
                  gap_m=TRIM_GAP_M, min_run_m=START_MIN_RUN_M):
    """(start, end) strip-column indices where the RIDGE signal is actually on.

    `start` = the first sustained run (>= min_run_m) that crosses the bar, so a
    dim lead pinned to the seed is skipped.  `end` = the last column before the
    ridge dies for >= gap_m (block semantics -- no forward extension into a dim
    tail).  A ridge that stays bright to the cap is kept."""
    d_arc = arcs_m[1] - arcs_m[0]
    gap = max(int(round(gap_m / d_arc)), 1)
    min_run = max(int(round(min_run_m / d_arc)), 1)
    N = len(path_contrast)
    on = path_contrast > floor
    if on.sum() < 3:                 # too dim to judge brightness -> keep whole
        return 0, N - 1
    bar = max(floor, rel * float(np.median(path_contrast[on])))
    alive = path_contrast >= bar

    # --- start: first alive run long enough that the signal is really on -----
    start = 0
    i = 0
    while i < N:
        if alive[i]:
            j = i
            while j < N and alive[j]:
                j += 1
            if j - i >= min_run:
                start = i
                break
            i = j
        else:
            i += 1

    # --- end: last alive column before the first dead run >= gap -------------
    last_alive = -1
    i = 0
    while i < N:
        if alive[i]:
            last_alive = i
            i += 1
        else:
            j = i
            while j < N and not alive[j]:
                j += 1
            if j - i >= gap and last_alive >= 0:
                break                   # signal fell off; close at last_alive
            i = j
    end = N - 1 if last_alive < 0 else last_alive
    return start, max(start, end)


def _remove_cost(pts, k):
    """Cost of dropping vertex k = its distance from the k-1..k+1 segment."""
    a, b = pts[k - 1], pts[k + 1]
    seg = b - a
    L = np.linalg.norm(seg)
    w = pts[k] - a
    if L < 1e-9:
        return float(np.linalg.norm(w))
    t = float(np.clip((w @ seg) / L, 0.0, L))
    return float(np.linalg.norm(w - t * (seg / L)))


def canonical_vertices(polyline, max_pts=4):
    """The trace -> a coarse <=`max_pts`-vertex polyline (display/schema only).

    NOTE: lossy serialization used only for voter-schema overlays -- NOT the
    source of the node's published poses (those come from segments.py)."""
    pts = np.asarray(polyline, dtype=float)
    n = len(pts)
    if n <= 2:
        sel = pts.copy()
    else:
        keep = {0, n - 1}
        stack = [(0, n - 1)]
        tol = 0.02
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            seg = pts[j] - pts[i]
            L = np.linalg.norm(seg)
            sub = pts[i + 1:j] - pts[i]
            if L > 1e-9:
                t = np.clip((sub @ seg) / L, 0.0, L)
                perp = np.linalg.norm(sub - t[:, None] * (seg / L), axis=1)
            else:
                perp = np.linalg.norm(sub, axis=1)
            m = i + 1 + int(np.argmax(perp))
            if perp.max() > tol:
                keep.add(m)
                stack.append((i, m))
                stack.append((m, j))
        sel = pts[sorted(keep)]
    while len(sel) > max_pts:                       # drop the least-important
        costs = np.array([_remove_cost(sel, k) for k in range(1, len(sel) - 1)])
        drop = int(np.argmin(costs)) + 1
        sel = np.delete(sel, drop, axis=0)
    while len(sel) < max_pts:                       # pad to exactly max_pts
        segs = np.linalg.norm(np.diff(sel, axis=0), axis=1)
        i = int(np.argmax(segs))
        mid = 0.5 * (sel[i] + sel[i + 1])
        sel = np.vstack([sel[:i + 1], mid, sel[i + 1:]])
    return sel


def chain_from_polyline(polyline, seed, mean_contrast, lead_deg):
    """Pack a trace into the voter.py 4-vertex chain dict (schema parity)."""
    seed = np.asarray(seed, dtype=float)
    v = canonical_vertices(polyline)
    return dict(
        p_anchor_far=(float(v[0, 0]), float(v[0, 1])),
        p_join1=(float(v[1, 0]), float(v[1, 1])),
        p_join2=(float(v[2, 0]), float(v[2, 1])),
        p_far_end=(float(v[3, 0]), float(v[3, 1])),
        anchor_score=float(mean_contrast),
        far_score=float(mean_contrast),
        min_score=float(mean_contrast),
        seed_side='anchor_end',
        seed_term=float(np.linalg.norm(v[0] - seed)),
        pre_gate_rank=1,
        centroid=(float(np.mean(v[:, 0])), float(np.mean(v[:, 1]))),
        lead_deg=float(lead_deg),
    )


def trace_pipe(scorer, seed):
    """Full seed-traced detection on an already-built ROI scorer.

    Returns (chain|None, polyline (Nx2)|None, info dict).  info['status'] is
    'ok', 'no_lead' (nothing bright attached to the seed), or 'short' (trace
    too short / dim)."""
    seed = np.asarray(seed, dtype=float)

    def _run(gate):
        lead = find_straight_lead(scorer, seed, near_gate=gate)
        if lead[0] is None:
            return None, None, dict(status='no_lead')
        lead_deg, score, near_cov = lead
        cs, offs, arcs, basis = build_strip(scorer, seed, lead_deg)
        path = dp_ridge(cs, offs)
        path_contrast = cs[np.arange(len(cs)), path]
        s0, end = signal_bounds(path_contrast, arcs)
        ux, uy, vx, vy = basis
        sl = slice(s0, end + 1)
        poly = np.column_stack([
            seed[0] + arcs[sl] * ux + offs[path[sl]] * vx,
            seed[1] + arcs[sl] * uy + offs[path[sl]] * vy,
        ])
        mean_c = float(np.mean(path_contrast[sl]))
        length = float(arcs[end] - arcs[s0])
        if length < MIN_TRACE_LEN_M:
            return None, poly, dict(status='short', length=length,
                                    lead_deg=lead_deg, mean_contrast=mean_c)
        chain = chain_from_polyline(poly, seed, mean_c, lead_deg)
        info = dict(status='ok', lead_deg=lead_deg, length=length,
                    mean_contrast=mean_c, near_cov=near_cov,
                    lead_score=score)
        return chain, poly, info

    chain, poly, info = _run(NEAR_COV_GATE)
    if chain is None and info['status'] == 'no_lead':
        chain, poly, info = _run(NEAR_COV_GATE * RELAX_NEAR_GATE)
        if info['status'] == 'ok':
            info['relaxed'] = True
    return chain, poly, info


def detect(cloud, seed):
    """Convenience: prepare the ROI then trace an in-memory cloud array.

    Returns (chain|None, polyline|None, info dict, roi_scorer|None)."""
    roi = prepare_roi(cloud, seed)
    if roi[0] is None:
        return None, None, dict(status='empty_roi'), None
    scorer, Xr, Yr = roi
    chain, poly, info = trace_pipe(scorer, seed)
    return chain, poly, info, scorer


# ----------------------------------------------------------------------------
# metric helpers (shared with the experiment's offline comparison scripts)
# ----------------------------------------------------------------------------
def densify(pts, step=0.1):
    """Resample a polyline every ~`step` m (for measuring pipe-following)."""
    pts = np.asarray(pts, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.array(out)


def polyline_distance(poly, pts):
    """Distance from each of `pts` (Mx2) to the nearest point on `poly` (Kx2)."""
    poly = np.asarray(poly, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if len(poly) < 2:
        return np.linalg.norm(pts - poly[0], axis=1) if len(poly) == 1 else \
            np.full(len(pts), np.inf)
    seg = np.diff(poly, axis=0)
    L2 = (seg ** 2).sum(axis=1)
    d = np.full(len(pts), np.inf)
    for j in range(len(seg)):
        v = seg[j]
        w = pts - poly[j]
        c1 = (w * v).sum(axis=1)
        c2 = L2[j]
        t = np.clip(c1 / max(c2, 1e-12), 0.0, 1.0)
        proj = poly[j] + t[:, None] * v
        d = np.minimum(d, np.linalg.norm(pts - proj, axis=1))
    return d


def gt_pipe_metric(poly, seed, gt):
    """Trace-based metric: how well `poly` follows the labeled pipe.
    Returns (mean_dist, max_dist, far_reach) in meters, or None if empty."""
    if poly is None or len(poly) == 0:
        return None
    gt = np.asarray(gt, dtype=float)
    seed = np.asarray(seed, dtype=float)
    ref = densify(np.vstack([seed, gt]))
    dd = polyline_distance(np.asarray(poly, dtype=float), ref)
    far = np.linalg.norm(np.asarray(poly[-1]) - seed)
    return float(dd.mean()), float(dd.max()), float(far)


def trace_is_correct(metric, mean_thr=0.10, max_thr=0.30,
                     min_reach=2.4, max_reach=4.0):
    """Binary correctness under the compare-metric."""
    if metric is None:
        return False
    mean_d, max_d, far = metric
    return (mean_d <= mean_thr and max_d <= max_thr and
            min_reach <= far <= max_reach)
