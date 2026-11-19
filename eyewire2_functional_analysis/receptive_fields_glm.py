"""Group-lasso GLM receptive fields with a ReLU output non-linearity, fitted with FISTA in JAX.

Model: ``y ~ relu(X w + b)`` with ``X`` a pre-computed design matrix (e.g. the stimulus projected onto a separable
B-spline basis, see ``receptive_fields.compute_basis_features``). Cost: ``mean((y - y_pred) ** 2) + beta * sum_g
||w_g||_2`` over groups of ``group_size`` consecutive features (e.g. the channels of one spatio-temporal basis
function), so a group is zero or non-zero as a whole and the penalty does not favor one feature of a group (a plain
lasso, ``group_size = 1``, tends to keep only the stronger of correlated features, e.g. of the color channels).

Solver: FISTA (``fit_fista``), i.e. proximal gradient steps (block soft-thresholding, so groups are exactly zero where
the penalty wins) with Nesterov momentum and adaptive restart, with a diagonal (Jacobi) preconditioner.

The model, the intercept handling and the initialization follow the GLM of RFEst (https://github.com/berenslab/RFEst,
MIT license) as extended in djimaging (eulerlab), with the group penalty in place of its L1 penalty and FISTA in place
of its Adam optimizer, and without the dependency on rfest.

Several responses (e.g. all ROIs of a recording field) sharing one design matrix are fitted jointly: the weights of all
responses form one matrix ``W`` (n_features x n_responses), so every step is a matrix-matrix product instead of a
matrix-vector product (which is limited by memory bandwidth, i.e. about as slow as the batched product). Each response
has its own penalty, step state and stopping criterion, and the cost of each response only depends on its own
parameters, so every fit is the same as fitting the responses one by one.

Requires JAX (on Linux with a GPU: ``pip install -e .[gpu]``); JAX uses a GPU if one is available.
"""
import functools
import time

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, lax
except ImportError:
    jax = jnp = None

# Number of FISTA iterations per call of the compiled function. The stopping criterion is evaluated within the compiled
# function; whether all responses have stopped is checked after each chunk.
_CHUNK_SIZE = 25
_DTYPE = np.float32


def to_device(x):
    """Design matrix as a float32 JAX array (on the GPU if available), to transfer it only once for several fits."""
    if jax is None:
        raise ImportError('jax is required, install it with `pip install jax` (GPU: `pip install -e .[gpu]`).')
    return jnp.asarray(x, _DTYPE)


def _forwardpass(p, x):
    """``relu(X W + b)``: ``(n_samples, n_responses)``."""
    out = x @ p['stimulus'] + p['intercept']['stimulus'] + p['intercept']['global']
    return jnp.where(out > 0., out, 0.).astype(_DTYPE)


def _group_norm(w, group_size):
    """Euclidean norm of each group of ``group_size`` consecutive rows of ``w`` ``(n_features, n_responses)``,
    repeated to the shape of ``w``; ``|w|`` for ``group_size = 1``."""
    if group_size == 1:
        return jnp.abs(w)
    norms = jnp.sqrt(jnp.sum(w.reshape(-1, group_size, w.shape[-1]) ** 2, axis=1))
    return jnp.repeat(norms, group_size, axis=0)


def _costs(p, x, y, beta, group_size):
    """Per-response MSE plus group-lasso penalty ``beta * sum_g ||w_g||_2``."""
    mse = jnp.mean((y - _forwardpass(p, x)) ** 2, axis=0)
    penalty = jnp.sum(_group_norm(p['stimulus'], group_size), axis=0) / group_size  # each group norm repeated
    return mse + beta * penalty


def compute_beta_max(x, y, group_size):
    """``beta_max = 2 max_g ||X_g'(y - mean(y))||_2 / n`` per response: the smallest penalty that gives all-zero
    weights for the linear model (only approximate with the ReLU). ``y``: ``(n_samples, n_responses)``."""
    x = to_device(x)
    xty = (x - x.mean(axis=0)).T @ (y - y.mean(axis=0))
    return 2 * np.asarray(jnp.max(_group_norm(xty, group_size), axis=0), np.float64) / x.shape[0]


def fista_steps(x, precondition='jacobi', step_scale=1., n_iter=100, margin=1.05, seed=0):
    """FISTA step sizes per weight ``(n_features,)`` and for the intercept: ``step_scale / (c d)``.

    ``d`` is the diagonal metric (``'none'``: all ones; ``'jacobi'``: the variances ``mean(x_j^2)`` of the features and
    1 for the intercept, i.e. FISTA on standardized features with the penalty re-weighted accordingly, the same
    model) and ``c`` the Lipschitz constant of the MSE gradient of the linear model in that metric, ``c = 2 / n *
    lambda_max(D^-1/2 [X 1]' [X 1] D^-1/2)`` (power iteration, times ``margin`` since it under-estimates). ``c d``
    also bounds the curvature of the MSE of the ReLU model, whose Gauss-Newton matrix ``2 / n [X 1]' M [X 1]`` has
    ``0 <= M <= I``. ``step_scale > 1`` is not guaranteed to converge.
    """
    x = to_device(x)
    n, n_feat = x.shape
    if precondition == 'jacobi':
        d_w = jnp.mean(x ** 2, axis=0)
        d_w = jnp.maximum(d_w, 1e-6 * jnp.max(d_w))
    elif precondition == 'none':
        d_w = jnp.ones(n_feat, _DTYPE)
    else:
        raise ValueError(f'Unknown precondition {precondition!r}.')
    s_w = 1. / jnp.sqrt(d_w)  # intercept: d_b = 1

    def body(_, v):
        u = x @ (v[:-1] * s_w) + v[-1]
        v = jnp.concatenate([(x.T @ u) * s_w, u.sum()[None]])
        return v / jnp.linalg.norm(v)

    v = jnp.asarray(np.random.default_rng(seed).standard_normal(n_feat + 1), _DTYPE)
    v = lax.fori_loop(0, n_iter, body, v / jnp.linalg.norm(v))
    u = x @ (v[:-1] * s_w) + v[-1]
    c = margin * 2. * float(u @ u) / n
    return np.asarray(step_scale / (c * d_w), np.float64), step_scale / c


def _prox(vw, beta, step_w, group_size):
    """Proximal step of ``beta sum_g ||w_g||_2``: block soft-thresholding of the group norms (``step_w`` must be the
    same within each group)."""
    nv = _group_norm(vw, group_size)
    return vw * jnp.maximum(1. - step_w * beta / jnp.where(nv > 0., nv, 1.), 0.)


def _gradient_mapping(w, vw, gw, beta, step_w, group_size):
    """``(w - prox(vw)) / step`` with ``vw = w - step gw``, computed from the gradient (``gw + beta vw / ||vw_g||``
    where the prox keeps the group, ``w / step`` where it zeroes it) instead of the difference, which is 0 once the
    update is below float32 resolution of ``w`` although the gradient is not."""
    nv = _group_norm(vw, group_size)
    unit = vw / jnp.where(nv > 0., nv, 1.)
    return jnp.where(nv > step_w * beta, gw + beta * unit, w / step_w)


@functools.lru_cache(maxsize=None)
def _get_fista_chunk_runner(num_iters, gtol, tolerance, min_iters, group_size):
    """`run_chunk` performs `_CHUNK_SIZE` FISTA steps (with adaptive restart) and updates the per-response stopping
    state. Compiled once per settings and input shapes."""

    def body(data, s, i):
        x, y, beta, step_w, step_b = data['x'], data['y'], data['beta'], data['step_w'], data['step_b']
        n = x.shape[0]
        # gradient of the MSE at the extrapolated point z
        out = x @ s['zw'] + s['zb']
        r = jnp.where(out > 0., out - y, 0.)
        gw, gb = (2. / n) * (x.T @ r), (2. / n) * r.sum(axis=0)
        # proximal step: block soft-thresholding of the weights, the intercept is not penalized
        vw = s['zw'] - step_w * gw
        w = _prox(vw, beta, step_w, group_size)
        b = s['zb'] - step_b * gb
        # gradient mapping G = (z - prox step) / step, in gradient units: zero iff z is optimal (KKT conditions)
        g_map_w, g_map_b = _gradient_mapping(s['zw'], vw, gw, beta, step_w, group_size), gb
        # adaptive restart (O'Donoghue & Candes 2015): drop the momentum if it points against the proximal step
        restart = (jnp.sum(g_map_w * (w - s['w']), axis=0) + g_map_b * (b - s['b'])) > 0.
        t = jnp.where(restart, 1., (1. + jnp.sqrt(1. + 4. * s['t'] ** 2)) / 2.)
        mom = jnp.where(restart, 0., (s['t'] - 1.) / t)
        zw, zb = w + mom * (w - s['w']), b + mom * (b - s['b'])
        # stop once the optimality residual (relative to beta_max) is below gtol in `tolerance` consecutive steps
        res = jnp.maximum(jnp.max(jnp.abs(g_map_w), axis=0), jnp.abs(g_map_b)) / data['g_scale']
        live = s['active'] & (i < num_iters)
        n_small = jnp.where(live, jnp.where(res <= gtol, s['n_small'] + 1, 0), s['n_small'])
        stop = live & (n_small >= tolerance) & (i + 1 >= min_iters)
        return dict(
            w=jnp.where(live, w, s['w']), b=jnp.where(live, b, s['b']), zw=jnp.where(live, zw, s['zw']),
            zb=jnp.where(live, zb, s['zb']), t=jnp.where(live, t, s['t']), n_small=n_small,
            res=jnp.where(live, res, s['res']), n_restart=s['n_restart'] + (live & restart),
            active=s['active'] & ~stop, n_iter=jnp.where(stop, i + 1, s['n_iter'])), None

    @jit
    def run_chunk(s, i0, data):
        s, _ = lax.scan(functools.partial(body, data), s, i0 + jnp.arange(_CHUNK_SIZE))
        return s

    return run_chunk


def fit_fista(x, y, w0, b0, beta, steps, g_scale, group_size, num_iters=5000, gtol=1e-4, tolerance=3, min_iters=10):
    """Minimize the cost of each response with FISTA (accelerated proximal gradient with adaptive restart) from
    ``(w0, b0)``. Groups of weights are exactly zero where the penalty wins.

    A fit stops once its optimality residual, the largest absolute entry of the gradient mapping (the proximal gradient
    step divided by the step size, in units of the gradient; zero at the optimum), divided by ``g_scale``, was at most
    ``gtol`` in ``tolerance`` consecutive steps (and after at least ``min_iters`` steps). Unlike the parameter change,
    this does not depend on the step size.

    Args:
        x: Design matrix ``(n_samples, n_features)``.
        y: Responses ``(n_samples, n_responses)``.
        w0, b0: Initial weights ``(n_features, n_responses)`` and intercepts ``(n_responses,)``.
        beta: Penalty per response ``(n_responses,)``.
        steps: Step sizes ``(step_w (n_features,), step_b)``, see ``fista_steps``; ``step_w`` must be the same within
            each group.
        g_scale: Gradient scale per response ``(n_responses,)``, e.g. ``beta_max``.
        group_size: Number of consecutive features per group.

    Returns:
        tuple: Weights, intercepts and a list of info dicts per response (``n_iter``, ``n_restart``, ``res`` = last
        relative optimality residual, ``cost`` = final train cost, ``stop`` = ``'train_stop'`` or ``'maxiter_stop'``).
    """
    run_chunk = _get_fista_chunk_runner(num_iters, gtol, tolerance, min_iters, group_size)
    data = dict(x=x, y=jnp.asarray(y, _DTYPE), beta=jnp.asarray(beta, _DTYPE),
                step_w=jnp.asarray(steps[0], _DTYPE)[:, None], step_b=jnp.asarray(steps[1], _DTYPE),
                g_scale=jnp.asarray(g_scale, _DTYPE))
    w0, b0 = jnp.asarray(w0, _DTYPE), jnp.asarray(b0, _DTYPE)
    n_resp = data['y'].shape[1]
    s = dict(w=w0, b=b0, zw=w0, zb=b0, t=jnp.ones(n_resp, _DTYPE), n_small=jnp.zeros(n_resp, jnp.int32),
             res=jnp.full(n_resp, np.inf, _DTYPE), n_restart=jnp.zeros(n_resp, jnp.int32),
             active=jnp.ones(n_resp, bool), n_iter=jnp.full(n_resp, num_iters, jnp.int32))

    for i_chunk in range(int(np.ceil(num_iters / _CHUNK_SIZE))):
        s = run_chunk(s, i_chunk * _CHUNK_SIZE, data)
        if not bool(jnp.any(s['active'])):
            break

    p = dict(stimulus=s['w'], intercept=dict(stimulus=jnp.zeros_like(s['b']), **{'global': s['b']}))
    costs = np.asarray(_costs(p, x, data['y'], data['beta'], group_size))
    active, n_iter, n_restart, res = (np.asarray(s[k]) for k in ('active', 'n_iter', 'n_restart', 'res'))
    infos = [dict(n_iter=int(n_iter[r]), n_restart=int(n_restart[r]), res=float(res[r]), cost=float(costs[r]),
                  stop='maxiter_stop' if active[r] else 'train_stop') for r in range(n_resp)]
    return s['w'], s['b'], infos


def fit_group_lasso_path(x, y, alphas_rel, group_size, eps_mean=1e-3, precondition='jacobi', step_scale=1.,
                         **kwargs):
    """Group-lasso GLM ``y ~ relu(X w + b)`` for all penalties ``alphas_rel`` (relative to ``beta_max``), large to
    small.

    The first (largest) penalty starts from zero weights and ``b = max(mean(y), eps_mean)`` (so the zero-weight model
    predicts the mean response and the ReLU is not dead), the others warm-start from the previous penalty.
    ``beta_max`` (``compute_beta_max``) is the smallest penalty that gives all-zero weights for the linear model (only
    approximate with the ReLU). Several responses are fitted jointly (see module docstring), each with its own
    ``beta_max``. FISTA step sizes come from ``fista_steps`` with ``precondition`` and ``step_scale``; within each group
    the smallest step of the group is used (block soft-thresholding needs one step per group; a smaller step is still a
    valid step). Optimality residuals are relative to ``beta_max``.

    Args:
        x: Design matrix ``(n_samples, n_features)``, float32 (numpy or jax array, see ``to_device``).
        y: Response ``(n_samples,)`` or responses ``(n_samples, n_responses)``, scaled (e.g. by their std) but NOT
            centered, so that their mean is positive.
        alphas_rel: Penalties relative to ``beta_max``.
        group_size: Number of consecutive features per group (``1``: plain lasso).
        **kwargs: Passed to ``fit_fista``.

    Returns:
        tuple: weights ``(n_alphas, n_features)`` and intercepts ``(n_alphas,)`` in the order of ``alphas_rel``, and a
        list of per-fit info dicts (see ``fit_fista``, plus ``alpha_idx`` and ``time_s``, the time of the joint fit).
        For 2D ``y``, weights ``(n_responses, n_alphas, n_features)``, intercepts ``(n_responses, n_alphas)`` and one
        list of info dicts per response.
    """
    x = to_device(x)
    alphas_rel = np.asarray(alphas_rel)
    if x.shape[1] % group_size:
        raise ValueError(f'{x.shape[1]} features are not divisible into groups of {group_size}.')
    y = np.asarray(y)  # beta_max and the initial intercept are computed before the cast to float32
    single = y.ndim == 1
    y = y.reshape(y.shape[0], -1)
    n_feat, n_resp = x.shape[1], y.shape[1]
    beta_max = compute_beta_max(x, y, group_size)
    w = jnp.zeros((n_feat, n_resp), _DTYPE)
    b = jnp.asarray(np.maximum(y.mean(axis=0), eps_mean), _DTYPE)
    y = y.astype(_DTYPE)

    step_w, step_b = fista_steps(x, precondition=precondition, step_scale=step_scale)
    steps = (np.repeat(step_w.reshape(-1, group_size).min(axis=1), group_size), step_b)  # one step per group

    weights, intercepts = np.zeros((n_resp, alphas_rel.size, n_feat), _DTYPE), np.zeros((n_resp, alphas_rel.size))
    infos = [[] for _ in range(n_resp)]
    for a in np.argsort(alphas_rel)[::-1]:  # large penalty first
        t0 = time.time()
        w, b, info = fit_fista(x, y, w, b, alphas_rel[a] * beta_max, steps, beta_max, group_size, **kwargs)
        weights[:, a], intercepts[:, a] = np.asarray(w).T, np.asarray(b, float)
        dt = time.time() - t0
        for r in range(n_resp):
            infos[r].append(dict(alpha_idx=int(a), **info[r], time_s=dt))
    if single:
        return weights[0], intercepts[0], infos[0]
    return weights, intercepts, infos


def predict(x, weights, intercepts):
    """``relu(X w + b)`` for weights ``(n_models, n_features)`` and intercepts ``(n_models,)``: ``(n_samples, n_models)``."""
    return np.maximum(np.asarray(x) @ np.asarray(weights).T + np.asarray(intercepts), 0.)
