"""Receptive field (RF) estimation from continuous stimuli (e.g. mouse-cam movies) and RF postprocessing.

Conventions follow djimaging (eulerlab): the RF time axis ``rf_time`` is relative to the response,
negative values are stimulus frames preceding the response. STRFs have shape ``(n_t, ny, nx[, n_channels])``.
"""
import warnings

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import curve_fit


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


def _weighted_stim_average(x: np.ndarray, w: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """``avg[c, j] = sum_t w[t, c] * x[t + shifts[j]] / sum_t w[t, c]`` for x of shape ``(n_frames, n_pixels)``."""
    n_frames = x.shape[0]
    avg = np.zeros((w.shape[1], shifts.size, x.shape[1]), dtype=np.float32)
    for j, s in enumerate(shifts):
        if s <= 0:
            avg[:, j] = w[-s:].T @ x[:n_frames + s]
        else:
            avg[:, j] = w[:n_frames - s].T @ x[s:]
    return avg / w.sum(axis=0)[:, None, None]


def compute_sta(stim: np.ndarray, responses: np.ndarray, shifts: np.ndarray,
                frame_masks: np.ndarray | None = None) -> np.ndarray:
    """Response-weighted stimulus average for many responses and frame subsets at once.

    For each frame subset (mask) the average stimulus of that subset is subtracted, i.e.
    ``sta[m, c, j] = sum_{t in m} y[t, c] x[t + s_j] / sum_{t in m} y[t, c] - sum_{t in m} x[t + s_j] / |m|``.
    This matters for natural movies, where the mean stimulus of a subset (e.g. the train clips) is not zero.

    Args:
        stim: Stimulus, shape ``(n_frames, ...)``.
        responses: Non-negative responses, shape ``(n_frames, n_responses)``.
        shifts: Frame shifts, see ``get_rf_shifts``.
        frame_masks: Boolean frame subsets, shape ``(n_frames, n_masks)``. Defaults to all frames.

    Returns:
        np.ndarray: STAs of shape ``(n_masks, n_responses, n_shifts, ...)``.
    """
    n_frames = stim.shape[0]
    assert responses.shape[0] == n_frames, (responses.shape, stim.shape)
    if frame_masks is None:
        frame_masks = np.ones((n_frames, 1), dtype=bool)
    x = stim.reshape(n_frames, -1).astype(np.float32)
    y = responses.astype(np.float32)
    m = frame_masks.astype(np.float32)
    n_masks, n_responses = m.shape[1], y.shape[1]

    w = (m[:, :, None] * y[:, None, :]).reshape(n_frames, n_masks * n_responses)
    sta = _weighted_stim_average(x, w, shifts).reshape(n_masks, n_responses, shifts.size, -1)
    sta -= _weighted_stim_average(x, m, shifts)[:, None]
    return sta.reshape((n_masks, n_responses, shifts.size) + stim.shape[1:])


def predict_linear(stim: np.ndarray, rfs: np.ndarray, shifts: np.ndarray, frame_idxs: np.ndarray) -> np.ndarray:
    """Linear prediction ``r[t, c] = sum_j <rfs[c, j], stim[t + shifts[j]]>`` at ``frame_idxs``.

    Returns:
        np.ndarray: Predictions of shape ``(frame_idxs.size, n_rfs)``.
    """
    n_frames = stim.shape[0]
    x = stim.reshape(n_frames, -1).astype(np.float32)
    w = rfs.reshape(rfs.shape[0], shifts.size, -1).astype(np.float32)
    pred = np.zeros((frame_idxs.size, rfs.shape[0]), dtype=np.float32)
    for j, s in enumerate(shifts):
        pred += x[np.clip(frame_idxs + s, 0, n_frames - 1)] @ w[:, j].T
    return pred


def compute_stim_covariances(stim: np.ndarray, shifts: np.ndarray,
                             frame_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Temporal and spatial(-chromatic) stimulus covariance for a separable approximation of the full
    spatiotemporal covariance, ``C ~ C_t (x) C_s``.

    Args:
        stim: Standardized stimulus, shape ``(n_frames, ...)``.
        shifts: Frame shifts, see ``get_rf_shifts``.
        frame_mask: Frames used to estimate the covariances. Defaults to all frames.

    Returns:
        tuple: ``(c_t, c_s)`` with ``c_t`` the pixel-averaged autocorrelation between shifts ``(n_shifts, n_shifts)``
        (unit diagonal) and ``c_s`` the pixel covariance ``(n_pixels, n_pixels)``.
    """
    n_frames = stim.shape[0]
    x = stim.reshape(n_frames, -1).astype(np.float32)
    idxs = np.arange(n_frames) if frame_mask is None else np.where(frame_mask)[0]
    c_s = x[idxs].T @ x[idxs] / idxs.size

    acov = []
    for lag in range(shifts.max() - shifts.min() + 1):
        i = idxs[idxs + lag < n_frames]
        acov.append(np.mean(x[i] * x[i + lag]))
    acov = np.array(acov)
    c_t = acov[np.abs(shifts[:, None] - shifts[None, :])] / acov[0]
    return c_t, c_s


def regularized_inverse(evals: np.ndarray, evecs: np.ndarray, reg: float) -> np.ndarray:
    """``(C + reg * mean(evals) * I)^-1`` from the eigendecomposition of a covariance ``C``."""
    return (evecs / (evals + reg * evals.mean())) @ evecs.T


def whiten_sta(sta: np.ndarray, c_t_inv: np.ndarray, c_s_inv: np.ndarray) -> np.ndarray:
    """Whitened STA ``C_t^-1 STA C_s^-1`` for STAs of shape ``(n_rfs, n_shifts, ...)``."""
    n_rfs, n_shifts = sta.shape[:2]
    sta_w = np.einsum('ij,njp,pq->niq', c_t_inv, sta.reshape(n_rfs, n_shifts, -1), c_s_inv, optimize=True)
    return sta_w.reshape(sta.shape)


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


def centered_gram(x: np.ndarray, y: np.ndarray, idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``X'X`` and ``X'Y`` of the rows ``idxs`` after centering ``x`` and ``y`` on these rows (float64)."""
    x_mean = x[idxs].mean(axis=0, dtype=np.float64)
    y_mean = y[idxs].mean(axis=0, dtype=np.float64)
    xtx = (x[idxs].T @ x[idxs]).astype(np.float64) - idxs.size * np.outer(x_mean, x_mean)
    xty = (x[idxs].T @ y[idxs]).astype(np.float64) - idxs.size * np.outer(x_mean, y_mean)
    return xtx, xty


def ridge_path(xtx: np.ndarray, xty: np.ndarray, regs: np.ndarray) -> np.ndarray:
    """Ridge solutions ``(X'X + reg * mean(eig(X'X)) * I)^-1 X'Y`` for all ``regs``.

    Returns:
        np.ndarray: Weights of shape ``(n_regs, n_features, n_responses)``.
    """
    evals, evecs = np.linalg.eigh(xtx)
    vty = evecs.T @ xty
    return np.stack([evecs @ (vty / (evals + reg * evals.mean())[:, None]) for reg in regs])


def elastic_net_path(xtx: np.ndarray, xty: np.ndarray, n_samples: int, alphas_rel: np.ndarray,
                     l1_ratio: float = 1., n_iter: int = 300) -> np.ndarray:
    """Elastic net ``1 / (2n) ||y - Xw||^2 + alpha (l1_ratio ||w||_1 + (1 - l1_ratio) / 2 ||w||^2)`` (lasso for
    ``l1_ratio=1``) from the Gram matrix with FISTA, for all responses at once.

    Alphas are relative to ``alpha_max = max|X'y| / (n l1_ratio)`` of each response (all-zero solution) and are
    traversed from large to small with warm starts. With a fixed ``n_iter`` the solutions may not be fully converged
    for small alphas, which acts as additional (early-stopping) regularization.

    Returns:
        np.ndarray: Weights of shape ``(n_alphas, n_features, n_responses)``.
    """
    g, b = (xtx / n_samples).astype(np.float32), (xty / n_samples).astype(np.float32)
    lipschitz_g = np.linalg.eigvalsh(g)[-1]
    alpha_max = np.abs(b).max(axis=0) / l1_ratio
    w = np.zeros_like(b)
    ws = []
    for alpha_rel in np.sort(alphas_rel)[::-1]:
        alpha = alpha_rel * alpha_max
        alpha_l2 = alpha * (1 - l1_ratio)
        lipschitz = lipschitz_g + alpha_l2  # per response
        thresh = alpha * l1_ratio / lipschitz
        z, t = w.copy(), 1.
        for _ in range(n_iter):
            v = z - (g @ z - b + alpha_l2 * z) / lipschitz
            w_new = np.sign(v) * np.clip(np.abs(v) - thresh, 0, None)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = w_new + (t - 1) / t_new * (w_new - w)
            w, t = w_new, t_new
        ws.append(w.copy())
    return np.stack(ws)[np.argsort(np.argsort(alphas_rel)[::-1])]


def compute_repeat_qidx(repeats: np.ndarray) -> float:
    """Quality index of Baden et al. 2016 for responses of shape ``(n_repeats, n_t)``."""
    return np.var(np.mean(repeats, axis=0)) / np.mean(np.var(repeats, axis=1))


def split_strf(strf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Split an STRF ``(n_t, ny, nx)`` into a spatial and temporal RF using SVD.

    The sign is chosen such that the sRF peak is positive, i.e. the tRF carries the polarity.
    The tRF is normalized to a maximum absolute value of 1 and the sRF scaled such that ``trf x srf``
    is the rank-1 approximation of ``strf``.

    Returns:
        tuple: ``(srf, trf, explained_variance)``.
    """
    n_t = strf.shape[0]
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


def fit_srf_gauss(srf: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit a 2D Gaussian (plus offset) to the positive peak of an sRF.

    Returns:
        tuple: ``(srf_fit, params)`` with params ``amp, x0, y0, sx, sy, theta, offset`` (in pixels) and the
        explained variance ``r2``. Parameters are NaN if the fit failed.
    """
    ny, nx = srf.shape
    yy, xx = np.mgrid[:ny, :nx]
    iy, ix = np.unravel_index(np.argmax(srf), srf.shape)
    p0 = (srf.max(), ix, iy, 1.5, 1.5, 0., 0.)
    bounds = ([0., -0.5, -0.5, 0.3, 0.3, -np.pi / 2, -np.inf], [np.inf, nx - 0.5, ny - 0.5, nx, ny, np.pi / 2, np.inf])
    names = ('amp', 'x0', 'y0', 'sx', 'sy', 'theta', 'offset')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(_gauss2d, (xx.ravel(), yy.ravel()), srf.ravel(), p0=p0, bounds=bounds, maxfev=5000)
    except RuntimeError:
        return np.full_like(srf, np.nan), dict(**{k: np.nan for k in names}, r2=np.nan)

    srf_fit = _gauss2d((xx, yy), *popt)
    params = dict(zip(names, popt))
    params['r2'] = 1. - np.var(srf - srf_fit) / np.var(srf)
    return srf_fit, params


def compute_trf_metrics(trf: np.ndarray, rf_time: np.ndarray) -> dict:
    """Polarity, main peak lag and transience index of a tRF, using only ``rf_time <= 0``.

    The main peak is the largest absolute value. Similar to djimaging, the transience index is
    ``2|pre| / (|main| + |pre|)``, where ``pre`` is the largest opposite-sign value preceding the main peak (0 if none).
    """
    past = rf_time <= 0
    trf, rf_time = trf[past], rf_time[past]

    i_main = np.argmax(np.abs(trf))
    main = np.abs(trf[i_main])
    polarity = int(np.sign(trf[i_main]))
    pre = np.max(-polarity * trf[:i_main], initial=0.)
    transience = 2 * pre / (main + pre)

    return dict(polarity=polarity, peak_lag_s=-rf_time[i_main], transience=transience)


def gauss_diameter(sx: float, sy: float, pixel_size_um: float) -> float:
    """Diameter of the circle with the area of the 2-sigma ellipse, as djimaging ``compute_gauss_srf_area``."""
    return 4. * np.sqrt(sx * sy) * pixel_size_um
