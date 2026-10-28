"""Spline-based linear-Gaussian GLM receptive fields with an L1 (lasso) / L2 penalty, fitted with RFEst (JAX).

Adapted from djimaging (eulerlab, ``djimaging.utils.receptive_fields.glm_utils.ReceptiveFieldGLM``), copied here so
that this repo does not depend on djimaging. Differences: single ROI, no permutation test / plotting / DataJoint,
and no forced CPU platform, so JAX runs on a GPU whenever one is available.

Model: ``y = intercept + sum_c X_c @ S @ b_c``, with one lagged design matrix ``X_c`` per stimulus channel (e.g. green
and UV) and a cubic-regression spline basis ``S`` over (t, y, x). A single elastic-net penalty
``beta * (alpha ||b||_1 + (1 - alpha) ||b||_2)`` is shared by the coefficients of all channels (``alpha=1`` is lasso).
RFEst optimizes it with Adam, so coefficients are shrunk but not set exactly to zero.

RF time axis convention as in ``receptive_fields.py``: STRFs have shape ``(n_t, ny, nx, n_channels)``, ``rf_time`` is
relative to the response and negative values are stimulus frames preceding the response.
"""
import warnings

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import rfest
    from jax.example_libraries import optimizers
    from rfest.loss import loss_penalty
    from rfest.splines import build_spline_matrix
    from rfest.utils import build_design_matrix
except ImportError:  # pragma: no cover
    jax = jnp = rfest = None


def fit_spline_glm(stim: np.ndarray, response: np.ndarray, n_past: int, df: tuple[int, int, int],
                   betas=(0.001, 0.01, 0.1), alpha: float = 1., n_future: int = 0, frac_dev: float = 0.2,
                   step_size: float = 0.1, max_iters: int = 1000, min_iters: int = 50, tolerance: int = 5,
                   atol: float = 1e-3, init_method: str = 'mle', seed: int = 42, verbose: int = 0) -> tuple:
    """Fit a spline GLM and pick the penalty strength ``beta`` on a held-out block at the end of the data.

    Args:
        stim: Stimulus, shape ``(n_frames, ny, nx, n_channels)``. Should be standardized.
        response: Response at the stimulus frame times, shape ``(n_frames,)``.
        n_past: Number of lags into the past (including lag 0).
        df: Degrees of freedom of the spline basis along (t, y, x); clipped to the filter size.
        betas: Candidate penalty strengths. The one with the best correlation on the dev block is kept.
        alpha: L1/L2 mix, 1 = lasso, 0 = L2.
        n_future: Number of lags into the future.
        frac_dev: Fraction of frames at the end held out to select ``beta``. If 0 or only one beta is given,
            no selection is done and the first beta is used.
        step_size, max_iters, min_iters, tolerance, atol: Adam settings (see ``rfest.GLM.fit``).
        init_method: ``'mle'`` or ``'random'`` initialization of the coefficients.
        seed: Random seed for the initialization.
        verbose: Verbosity of RFEst.

    Returns:
        tuple: ``(strf, info)``. ``strf`` has shape ``(n_t, ny, nx, n_channels)`` and ``info`` holds
        ``rf_time`` (in frames), ``beta``, ``intercept``, ``dev_ccs`` (per beta, NaN without dev block),
        ``train_cc`` and the fitted ``model``.
    """
    assert rfest is not None, 'rfest (and jax) are required'
    stim = np.asarray(stim, dtype=np.float32)
    response = np.asarray(response, dtype=np.float32)
    assert stim.ndim == 4 and stim.shape[0] == response.size, (stim.shape, response.shape)
    n_frames, ny, nx, n_ch = stim.shape
    n_t = n_past + n_future
    dims = (n_t, ny, nx)
    shift = -n_future
    df_clipped = tuple(int(min(d, n)) for d, n in zip(df, dims))
    if df_clipped != tuple(df):
        warnings.warn(f'df clipped from {tuple(df)} to {df_clipped}')
    df = df_clipped
    betas = np.sort(np.atleast_1d(betas).astype(float))
    names = [f'ch{c}' for c in range(n_ch)]
    stim_flat = stim.reshape(n_frames, ny * nx, n_ch)

    n_train = int(round(n_frames * (1. - frac_dev))) if (frac_dev > 0 and betas.size > 1) else n_frames
    has_dev = n_train < n_frames

    def build(n_fit: int, with_dev: bool):
        model = rfest.GLM(distr='gaussian', output_nonlinearity='none', dtype=jnp.float32)
        model.burn_in = n_past
        for c, name in enumerate(names):
            model.add_design_matrix(stim_flat[:n_fit, :, c], dims=dims, df=df, smooth='cr', name=name, shift=shift)
            if with_dev:
                model.add_design_matrix(stim_flat[n_fit:, :, c], name=name, kind='dev')
        model.initialize(num_subunits=1, dt=1., method=init_method, compute_ci=False, random_seed=seed,
                         y=response[:n_fit], verbose=verbose)
        return model

    def fit(model, y, beta):
        model.fit(y={'train': y}, metric='mse', num_iters=max_iters, min_iters=min_iters, step_size=step_size,
                  atol=atol, tolerance=tolerance, alpha=alpha, beta=beta, verbose=verbose,
                  return_model='best_train_cost')

    dev_ccs = np.full(betas.size, np.nan)
    if has_dev:
        y_dev = response[n_train:][n_past:]
        for i, beta in enumerate(betas):
            model = build(n_train, with_dev=True)
            fit(model, response[:n_train], beta)
            y_pred = np.asarray(model.forwardpass(model.p['opt'], kind='dev')).ravel()
            dev_ccs[i] = _corrcoef(y_dev, y_pred)
        best = int(np.nanargmax(dev_ccs))
    else:
        best = 0

    model = build(n_train, with_dev=False)
    fit(model, response[:n_train], betas[best])
    y_pred = np.asarray(model.forwardpass(model.p['opt'], kind='train')).ravel()

    strf = np.stack([np.asarray(model.w['opt'][name]).reshape(dims) for name in names], axis=-1)
    info = dict(rf_time=np.arange(-n_past + 1, n_future + 1), beta=float(betas[best]),
                intercept=float(sum(np.asarray(v).item() for v in model.p['opt']['intercept'].values())),  # per-filter + global
                dev_ccs=dev_ccs, train_cc=_corrcoef(response[:n_train][n_past:], y_pred), model=model)
    return strf, info


def fit_spline_glm_batch(stim: np.ndarray, responses: np.ndarray, n_past: int, df: tuple[int, int, int],
                         betas=(0.001, 0.01, 0.1), alpha: float = 1., n_future: int = 0, frac_dev: float = 0.2,
                         step_size: float = 0.1, n_iters: int = 1000, roi_chunk: int | None = None) -> tuple:
    """Fit the spline GLM of ``fit_spline_glm`` for many ROIs and all betas at once (``jax.vmap``).

    All ROIs share the stimulus, so the spline-projected design matrix is built once; only the response differs.
    The Adam optimization from RFEst is replicated for every (beta, ROI) pair in a single jitted loop with a fixed
    number of iterations (no per-fit early stopping). Unlike ``fit_spline_glm``, the design matrix is built from the
    continuous stimulus, so the dev block has real stimulus history in its first frames. The best beta is picked per
    ROI by the correlation on the dev block (the last ``frac_dev`` of the frames); models are fitted on the training
    block only.

    Args:
        stim: Stimulus, shape ``(n_frames, ny, nx, n_channels)``. Should be standardized.
        responses: Responses at the stimulus frame times, shape ``(n_frames, n_rois)``.
        n_past, df, betas, alpha, n_future, frac_dev, step_size: As in ``fit_spline_glm``.
        n_iters: Number of Adam iterations.
        roi_chunk: Fit this many ROIs at a time to limit GPU memory (default: all at once).

    Returns:
        tuple: ``(strfs, info)``. ``strfs`` has shape ``(n_rois, n_t, ny, nx, n_channels)``; ``info`` holds
        ``rf_time``, ``beta`` and ``intercept`` and ``train_cc`` (each ``(n_rois,)``) and ``dev_ccs``
        (``(n_betas, n_rois)``).
    """
    assert rfest is not None, 'rfest (and jax) are required'
    stim = np.asarray(stim, dtype=np.float32)
    responses = np.asarray(responses, dtype=np.float32)
    assert stim.ndim == 4 and responses.ndim == 2 and stim.shape[0] == responses.shape[0], \
        (stim.shape, responses.shape)
    n_frames, ny, nx, n_ch = stim.shape
    n_rois = responses.shape[1]
    n_t = n_past + n_future
    dims = (n_t, ny, nx)
    df_clipped = tuple(int(min(d, n)) for d, n in zip(df, dims))
    if df_clipped != tuple(df):
        warnings.warn(f'df clipped from {tuple(df)} to {df_clipped}')
    betas = np.sort(np.atleast_1d(betas).astype(float))

    # Design matrix projected on the spline basis, channels side by side: (n_frames - n_past, n_ch * n_basis)
    spline = np.asarray(build_spline_matrix(dims=dims, df=df_clipped, smooth='cr', dtype=np.float32))
    n_basis = spline.shape[1]
    xs = np.concatenate([
        build_design_matrix(stim[..., c].reshape(n_frames, -1), n_t, shift=-n_future, dtype=np.float32)[n_past:] @ spline
        for c in range(n_ch)], axis=1)
    y_all = responses[n_past:]

    n_train = int(round((n_frames - n_past) * (1. - frac_dev))) if (frac_dev > 0 and betas.size > 1) \
        else n_frames - n_past
    x_tr, y_tr, x_dev, y_dev = xs[:n_train], y_all[:n_train], xs[n_train:], y_all[n_train:]

    # Initialization: least squares with intercept (as rfest 'mle'), shared design -> one solve for all ROIs
    x1 = np.hstack([np.ones((n_train, 1), np.float32), x_tr]).astype(np.float64)
    sol = np.linalg.lstsq(x1.T @ x1, x1.T @ y_tr.astype(np.float64), rcond=None)[0].astype(np.float32)
    w0, b0 = sol[1:], sol[0]  # (n_ch * n_basis, n_rois), (n_rois,)

    x_tr_j = jnp.asarray(x_tr)

    def cost(params, y, beta):
        w, b = params
        loss = jnp.mean((y - (x_tr_j @ w + b)) ** 2)
        return loss + loss_penalty(w, alpha, beta)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    grad = jax.vmap(jax.vmap(jax.grad(cost), in_axes=((1, 0), 1, None), out_axes=(1, 0)),  # ROIs
                    in_axes=((0, 0), None, 0), out_axes=(0, 0))  # betas

    @jax.jit
    def run(w_init, b_init, y, betas_j):
        def step(i, state):
            g = grad(get_params(state), y, betas_j)
            return opt_update(i, g, state)

        return get_params(jax.lax.fori_loop(0, n_iters, step, opt_init((w_init, b_init))))

    chunk = n_rois if roi_chunk is None else roi_chunk
    ws, bs = [], []
    for i in range(0, n_rois, chunk):
        sl = slice(i, i + chunk)
        n_c = w0[:, sl].shape[1]
        w_init = np.broadcast_to(w0[:, sl], (betas.size,) + w0[:, sl].shape).copy()
        b_init = np.broadcast_to(b0[sl], (betas.size, n_c)).copy()
        w, b = run(jnp.asarray(w_init), jnp.asarray(b_init), jnp.asarray(y_tr[:, sl]), jnp.asarray(betas, jnp.float32))
        ws.append(np.asarray(w)), bs.append(np.asarray(b))
    w, b = np.concatenate(ws, axis=2), np.concatenate(bs, axis=1)  # (n_betas, n_ch * n_basis, n_rois), (n_betas, n_rois)

    pred_tr = np.einsum('nk,bkr->bnr', x_tr, w) + b[:, None]
    train_cc = _corrcoef_cols(y_tr, pred_tr)  # (n_betas, n_rois)
    if n_train < y_all.shape[0]:
        dev_ccs = _corrcoef_cols(y_dev, np.einsum('nk,bkr->bnr', x_dev, w) + b[:, None])
        best = np.nanargmax(np.where(np.isfinite(dev_ccs), dev_ccs, -np.inf), axis=0)
    else:
        dev_ccs = np.full((betas.size, n_rois), np.nan)
        best = np.zeros(n_rois, dtype=int)

    roi_idx = np.arange(n_rois)
    w_best = w[best, :, roi_idx].reshape(n_rois, n_ch, n_basis)  # (n_rois, n_ch, n_basis)
    strfs = np.einsum('pk,rck->rpc', spline, w_best).reshape((n_rois,) + dims + (n_ch,))
    info = dict(rf_time=np.arange(-n_past + 1, n_future + 1), beta=betas[best], intercept=b[best, roi_idx],
                train_cc=train_cc[best, roi_idx], dev_ccs=dev_ccs)
    return strfs, info


def predict_spline_glm(strf: np.ndarray, intercept: float, stim: np.ndarray, n_past: int,
                       n_future: int = 0) -> np.ndarray:
    """Linear prediction of a fitted STRF, aligned with ``response[n_past:]`` (the burn-in frames are dropped).

    Args:
        strf: Shape ``(n_t, ny, nx, n_channels)`` with ``n_t = n_past + n_future``.
        intercept: Intercept of the fit.
        stim: Stimulus, shape ``(n_frames, ny, nx, n_channels)``.
        n_past: Number of lags into the past.
        n_future: Number of lags into the future.

    Returns:
        np.ndarray: Prediction, shape ``(n_frames - n_past,)``.
    """
    n_t = n_past + n_future
    assert strf.shape[0] == n_t and strf.shape[1:] == stim.shape[1:], (strf.shape, stim.shape)
    n_frames = stim.shape[0]
    flat = np.asarray(stim, dtype=np.float32).reshape(n_frames, -1)
    padded = np.vstack([np.zeros((n_t - n_future - 1, flat.shape[1]), np.float32), flat,
                        np.zeros((n_future, flat.shape[1]), np.float32)])
    y_pred = np.full(n_frames, float(intercept))
    for lag in range(n_t):
        # lag index 0 is the oldest frame; the design matrix uses the same ordering as rfest
        y_pred += padded[lag:lag + n_frames] @ strf[lag].reshape(-1)
    return y_pred[n_past:]


def _corrcoef_cols(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Correlation per (beta, ROI): ``y`` is ``(n, n_rois)``, ``y_pred`` is ``(n_betas, n, n_rois)``."""
    yc = y - y.mean(0)
    pc = y_pred - y_pred.mean(1, keepdims=True)
    denom = np.sqrt((yc ** 2).sum(0) * (pc ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, (yc * pc).sum(1) / denom, 0.)


def _corrcoef(y: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y) == 0 or np.std(y_pred) == 0:
        return 0.
    return float(np.corrcoef(y, y_pred)[0, 1])
