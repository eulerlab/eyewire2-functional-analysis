"""Receptive field (RF) estimation from continuous stimuli (e.g. mouse-cam movies) and RF postprocessing.

Conventions follow djimaging (eulerlab): the RF time axis ``rf_time`` is relative to the response,
negative values are stimulus frames preceding the response. STRFs have shape ``(n_t, ny, nx[, n_channels])``.
"""
import itertools
import warnings

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

def compute_response(trace: np.ndarray, tracetime: np.ndarray, frame_times: np.ndarray,
                     kind: str = 'gradient') -> np.ndarray:
    """Transform a calcium trace and resample it at the stimulus frame times.

    Args:
        trace: Fluorescence trace, shape ``(n_samples,)``.
        tracetime: Sample times of ``trace``.
        frame_times: Stimulus frame times to resample to.
        kind: ``'gradient'`` (positive part of the derivative, as djimaging ``fit_kind='gradient'``) or ``'trace'``.

    Returns:
        np.ndarray: Response at ``frame_times``.
    """
    if kind == 'gradient':
        response = np.clip(np.diff(trace), 0, None)
        response_time = tracetime[1:] - np.diff(tracetime) / 2.  # derivative is centered between samples
    elif kind == 'trace':
        response, response_time = trace, tracetime
    else:
        raise NotImplementedError(kind)

    assert response_time[0] <= frame_times[0] and frame_times[-1] <= response_time[-1] + np.diff(tracetime).mean()
    return np.interp(frame_times, response_time, response)


def standardize_stim(stim: np.ndarray, frame_mask: np.ndarray | None = None) -> np.ndarray:
    """Subtract the per-pixel mean and divide by the per-channel standard deviation.

    Args:
        stim: Stimulus of shape ``(n_frames, ny, nx, n_channels)``.
        frame_mask: Frames used to compute the mean and std. Defaults to all frames.
    """
    ref = stim if frame_mask is None else stim[frame_mask]
    stim = stim - ref.mean(axis=0, keepdims=True)
    return stim / (ref - ref.mean(axis=0, keepdims=True)).std(axis=(0, 1, 2), keepdims=True)


def get_rf_shifts(dt: float, dur_past_s: float, dur_future_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Frame shifts and the corresponding RF time axis in seconds."""
    shifts = np.arange(-int(round(dur_past_s / dt)), int(round(dur_future_s / dt)) + 1)
    return shifts, shifts * dt


def bspline_basis(n: int, df: int, degree: int = 3) -> np.ndarray:
    """B-spline basis of shape ``(n, df)`` with uniformly spaced knots on ``0, ..., n - 1``."""
    inner = np.linspace(0, n - 1, df - degree + 1)
    knots = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    return BSpline.design_matrix(np.arange(n, dtype=float), knots, degree).toarray()


def compute_basis_features(stim: np.ndarray, shifts: np.ndarray, basis_t: np.ndarray, basis_y: np.ndarray,
                           basis_x: np.ndarray) -> np.ndarray:
    """Design matrix of a stimulus ``(n_frames, ny, nx, n_channels)`` projected onto a separable STRF basis.

    Feature ``(k, i, l, c)`` at frame ``t`` is ``sum_{j,y,x} B_t[j, k] B_y[y, i] B_x[x, l] stim[t + shifts[j], y, x, c]``,
    stimulus frames outside the movie are zero. Use ``basis_to_strf`` to map weights back to an STRF.

    Returns:
        np.ndarray: Features of shape ``(n_frames, n_bt * n_by * n_bx * n_channels)``.
    """
    n_frames = stim.shape[0]
    s = np.einsum('tyxc,yi,xl->tilc', stim, basis_y, basis_x, optimize=True).reshape(n_frames, -1).astype(np.float32)
    feats = np.zeros((n_frames, basis_t.shape[1], s.shape[1]), dtype=np.float32)
    for j, shift in enumerate(shifts):
        lagged = np.zeros_like(s)
        if shift <= 0:
            lagged[-shift:] = s[:n_frames + shift]
        else:
            lagged[:n_frames - shift] = s[shift:]
        feats += basis_t[j][None, :, None].astype(np.float32) * lagged[:, None, :]
    return feats.reshape(n_frames, -1)


def basis_to_strf(w: np.ndarray, basis_t: np.ndarray, basis_y: np.ndarray, basis_x: np.ndarray) -> np.ndarray:
    """Map basis weights ``(n_rfs, n_features)`` to STRFs ``(n_rfs, n_shifts, ny, nx, n_channels)``."""
    w = w.reshape(w.shape[0], basis_t.shape[1], basis_y.shape[1], basis_x.shape[1], -1)
    return np.einsum('jk,yi,xl,nkilc->njyxc', basis_t, basis_y, basis_x, w, optimize=True)


def split_strf(strf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Split an STRF ``(n_t, ny, nx)`` into a spatial and temporal RF using SVD.

    The sign is chosen such that the sRF peak is positive, i.e. the tRF carries the polarity.
    The tRF is normalized to a maximum absolute value of 1 and the sRF scaled such that ``trf x srf``
    is the rank-1 approximation of ``strf``.

    Returns:
        tuple: ``(srf, trf, explained_variance)``.
    """
    n_t = strf.shape[0]
    # NaN/inf (e.g. a diverged fit) makes the SVD fail to converge; an all-zero STRF (e.g. fully shrunk by the
    # regularizer) has no defined sRF / tRF. Return NaNs for both.
    if not np.all(np.isfinite(strf)) or not np.any(strf):
        return np.full(strf.shape[1:], np.nan), np.full(n_t, np.nan), np.nan
    u, s, vt = np.linalg.svd(strf.reshape(n_t, -1), full_matrices=False)
    srf = vt[0].reshape(strf.shape[1:]) * s[0]
    trf = u[:, 0]

    sign = np.sign(srf.flat[np.argmax(np.abs(srf))])
    scale = np.max(np.abs(trf))
    return srf * sign * scale, trf * sign / scale, s[0] ** 2 / np.sum(s ** 2)


def _gauss2d(xy, amp, x0, y0, sx, sy, theta, offset):
    x, y = xy
    a = np.cos(theta) ** 2 / (2 * sx ** 2) + np.sin(theta) ** 2 / (2 * sy ** 2)
    b = -np.sin(2 * theta) / (4 * sx ** 2) + np.sin(2 * theta) / (4 * sy ** 2)
    c = np.sin(theta) ** 2 / (2 * sx ** 2) + np.cos(theta) ** 2 / (2 * sy ** 2)
    return offset + amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))


def _unit_gauss2d(xx, yy, x0, y0, sx, sy, theta):
    return _gauss2d((xx, yy), 1., x0, y0, sx, sy, theta, 0.)


def srf_component(params: dict, shape: tuple[int, int], which: str = 'center') -> np.ndarray:
    """Unit-peak Gaussian image of the ``center`` or ``surround`` of a ``fit_srfs`` fit, e.g. to draw its 2-sigma ellipse
    as the contour at ``exp(-2)``. All NaN if the fit failed or has no surround."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    sx, sy = (params['sx'], params['sy']) if which == 'center' else (params.get('sx_s', np.nan),
                                                                      params.get('sy_s', np.nan))
    if not np.all(np.isfinite([params['x0'], params['y0'], sx, sy, params['theta']])):
        return np.full(shape, np.nan)
    return _unit_gauss2d(xx, yy, params['x0'], params['y0'], sx, sy, params['theta'])


def draw_srf_fit(ax, params: dict, shape: tuple[int, int], color, transform=None, xs=None, ys=None, lw: float = 0.6,
                 center_ls: str = '-', min_surround_vol: float = 0.1) -> None:
    """Draw the 2-sigma ellipses of a ``fit_srfs`` fit on ``ax``: the center (``center_ls``) and, if its volume relative
    to the center (``w * k ** 2``) is at least ``min_surround_vol``, the surround (dotted).

    Args:
        transform: Applied to the component images before drawing (e.g. ``movie_to_retina``).
        xs, ys: Coordinates of the pixel centers of the (transformed) image (e.g. from ``pixel_centers``); pixel indices
            if None.
    """
    surround_vol = params.get('surround_rel_amp', 0.) * params.get('surround_ratio', 0.) ** 2
    for which, ls in (('center', center_ls), ('surround', ':')):
        if which == 'surround' and not surround_vol >= min_surround_vol:
            continue
        g = srf_component(params, shape, which)
        g = transform(g) if transform is not None else g
        if np.all(np.isfinite(g)) and g.max() > np.exp(-2):
            args = (xs, ys, g) if xs is not None else (g,)
            ax.contour(*args, levels=[np.exp(-2)], colors=color, linewidths=lw, linestyles=ls)


def fit_srfs(srfs: np.ndarray, kind: str = 'dog', center_prior_sd: float = 4., center_prior_weight: float = 0.05,
             coupling_weight: float = 0.05, surround_ratio: tuple[float, float] = (2., 10.),
             sigma_init: tuple[float, ...] = (1.5, 4.)) -> tuple[np.ndarray, list[dict]]:
    """Fit a difference of Gaussians (or a Gaussian) to the sRF of each channel jointly, with soft priors on the centers.

    Model per channel (pixel ``i`` at coordinate ``i``): ``amp * (G_c - w * G_s)`` with unit-peak Gaussians ``G``.
    The center ``G_c`` has its own position, angle and ``sx, sy``. The surround ``G_s`` is concentric with the center
    and has the same angle and aspect ratio, scaled by ``k`` in ``surround_ratio``. ``0 <= w <= 1``, so the surround
    always has the opposite sign of the center and the sRF has the center's sign at the center (as djimaging
    ``srf_dog_model``). ``amp`` is signed, i.e. the center can be the negative part of the sRF (e.g. a ring-shaped sRF).
    For ``kind='gauss'`` the surround is omitted.

    The least-squares cost is the squared residual relative to the total sum of squares of all channels' sRFs (i.e. the
    fraction of variance unexplained, about the mean of zero) plus two soft penalties in the same units:

    - center prior: ``center_prior_weight * (r / center_prior_sd) ** 2`` per channel, with ``r`` the distance of the
      channel's center from the window center (the stimulus is centered on the recording field);
    - coupling: ``coupling_weight * d ** 2 / (s_i ** 2 + s_j ** 2)`` per pair of channels, with ``d`` the distance between
      their centers and ``s = sqrt(sx * sy)`` their center sigmas, i.e. separations are penalized relative to the size.

    The fit is started from the window center and from the largest |value| of the strongest channel, with each sigma in
    ``sigma_init``; the initial center sign of each channel is the sign of the sRF weighted with the initial center
    Gaussian. The start with the lowest penalized cost is kept.

    The lower bound of ``surround_ratio`` must not be close to 1: two similar Gaussians subtracted give a ring, i.e. the
    DoG can then fit a blob with a much larger center and a thin surround ring.

    Args:
        srfs: sRFs of shape ``(n_channels, ny, nx)``.
        kind: ``'dog'`` or ``'gauss'``.
        center_prior_sd: SD of the center prior in pixels.
        center_prior_weight: Weight of the center prior (0: no prior).
        coupling_weight: Weight of the coupling between the channel centers (0: independent channels).
        surround_ratio: Bounds of the surround / center sigma ratio ``k``.
        sigma_init: Initial center sigmas (pixels) to start from.

    Returns:
        tuple: ``(srf_fits, params)`` with the fits of shape ``(n_channels, ny, nx)`` and per channel a dict with
        ``amp`` (signed center peak), ``polarity`` (center sign), ``x0, y0, sx, sy, theta`` of the center (pixels,
        radians) and the explained variance ``r2``; for the DoG also ``amp_s`` (signed surround peak), ``sx_s, sy_s``,
        ``surround_ratio`` (``k``) and ``surround_rel_amp`` (``w``). NaN if the fit failed.
    """

    if kind not in ('dog', 'gauss'):
        raise ValueError(kind)
    dog = kind == 'dog'
    n_ch, ny, nx = srfs.shape
    names = ('amp', 'polarity', 'x0', 'y0', 'sx', 'sy', 'theta', 'r2') + (
        ('amp_s', 'sx_s', 'sy_s', 'surround_ratio', 'surround_rel_amp') if dog else ())
    norm = np.sqrt(np.sum(srfs ** 2)) if np.all(np.isfinite(srfs)) else 0.
    if norm == 0:  # e.g. NaN sRFs from ``split_strf`` of a failed STRF
        return np.full(srfs.shape, np.nan), [{k: np.nan for k in names} for _ in range(n_ch)]

    yy, xx = np.mgrid[:ny, :nx].astype(float)
    xc, yc = (nx - 1) / 2, (ny - 1) / 2
    z = srfs / norm
    n_p = 8 if dog else 6  # x0, y0, theta, sx, sy, amp[, k, w]
    s_max = float(max(nx, ny))
    lb = np.tile([-0.5, -0.5, -np.inf, 0.3, 0.3, -np.inf] + ([surround_ratio[0], 0.] if dog else []), n_ch)
    ub = np.tile([nx - 0.5, ny - 0.5, np.inf, s_max, s_max, np.inf] + ([surround_ratio[1], 1.] if dog else []), n_ch)
    pairs = list(itertools.combinations(range(n_ch), 2))
    sqrt_wc, sqrt_wk = np.sqrt(center_prior_weight), np.sqrt(coupling_weight)

    def channel_model(p):
        g = _unit_gauss2d(xx, yy, p[0], p[1], p[3], p[4], p[2])
        if dog:
            g = g - p[7] * _unit_gauss2d(xx, yy, p[0], p[1], p[6] * p[3], p[6] * p[4], p[2])
        return p[5] * g

    def residuals(theta):
        ps = theta.reshape(n_ch, n_p)
        res = [(channel_model(p) - z[c]).ravel() for c, p in enumerate(ps)]
        res.append(sqrt_wc * (ps[:, :2] - [xc, yc]).ravel() / center_prior_sd)
        for i, j in pairs:
            res.append(sqrt_wk * (ps[i, :2] - ps[j, :2]) / np.sqrt(ps[i, 3] * ps[i, 4] + ps[j, 3] * ps[j, 4]))
        return np.concatenate(res)

    ch0 = int(np.argmax(np.abs(z).max(axis=(1, 2))))
    iy, ix = np.unravel_index(np.argmax(np.abs(z[ch0])), (ny, nx))
    starts = {(xc, yc), (float(ix), float(iy))}
    best = None
    for (x0, y0), s0 in itertools.product(starts, sigma_init):
        g0 = _unit_gauss2d(xx, yy, x0, y0, s0, s0, 0.)
        p0 = np.concatenate([[x0, y0, 0., s0, s0, (np.sign(np.sum(g0 * z[c])) or 1.) * max(np.abs(z[c]).max(), 1e-3)]
                             + ([3., 0.2] if dog else []) for c in range(n_ch)])
        p0 = np.clip(p0, lb + 1e-6, ub - 1e-6)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sol = least_squares(residuals, p0, bounds=(lb, ub), x_scale='jac', max_nfev=2000)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best = sol
    if best is None:
        return np.full(srfs.shape, np.nan), [{k: np.nan for k in names} for _ in range(n_ch)]

    fits, params = np.zeros(srfs.shape), []
    for c, p in enumerate(best.x.reshape(n_ch, n_p)):
        fits[c] = channel_model(p) * norm
        d = dict(amp=p[5] * norm, polarity=int(np.sign(p[5])), x0=p[0], y0=p[1], sx=p[3], sy=p[4],
                 theta=(p[2] + np.pi / 2) % np.pi - np.pi / 2, r2=1. - np.var(srfs[c] - fits[c]) / np.var(srfs[c]))
        if dog:
            d.update(amp_s=-p[7] * p[5] * norm, sx_s=p[6] * p[3], sy_s=p[6] * p[4], surround_ratio=p[6],
                     surround_rel_amp=p[7])
        params.append(d)
    return fits, params


def movie_to_retina(img: np.ndarray) -> np.ndarray:
    """Reorient a movie-space image ``(rows, cols)`` to the retinal frame, to be shown with ``origin='lower'``:
    x = nasal (columns), y = dorsal (rows). From ``data/stimuli/README.md``: array row -> temporal (so nasal = -row),
    array column -> ventral (so dorsal = -column), i.e. the transpose with both axes reversed. Checked in
    ``scripts/analysis/rf_offsets/REPORT.md``: every other flip / rotation fits worse, and the moving-bar RF centers
    agree with the mouse-cam RF centers per ROI."""
    return img.T[::-1, ::-1]
