"""L1-penalized (lasso) or elastic-net GLM receptive fields with a ReLU output non-linearity, fitted with FISTA or Adam
in JAX.

Model: ``y ~ relu(X w + b)`` with ``X`` a pre-computed design matrix (e.g. the stimulus projected onto a separable
B-spline basis, see ``receptive_fields.compute_basis_features``). Cost: ``mean((y - y_pred) ** 2) + beta * ||w||_1``
(lasso) or, with ``l1_ratio < 1`` (elastic net, FISTA only), ``mean((y - y_pred) ** 2) + beta * (l1_ratio * ||w||_1 +
(1 - l1_ratio) / 2 * ||w||_2^2)`` (the scikit-learn parametrization).

The default solver is FISTA (``fit_fista``): proximal gradient steps (soft-thresholding, so weights are exactly zero
where the penalty wins; for the elastic net followed by the exact shrinkage of the L2 term) with Nesterov momentum and
adaptive restart. It needs far fewer iterations than Adam and converges to a lower cost. The Adam solver
(``fit_adam``, lasso only) reproduces RFEst, see below.

Based on RFEst (https://github.com/berenslab/RFEst, MIT license), via the RFEst GLM extension of
djimaging (eulerlab), which subclasses ``rfest.GLM`` but overrides its design matrix handling, initialization and
optimization. This module is that extension trimmed to the code path used here (pre-computed design matrix, fitted
intercept, ReLU output, no dev set, no CI), without the dependency on rfest: forward pass, cost (RFEst's ``loss_mse`` and
``loss_penalty`` with ``alpha=1``, re-implemented below), Adam and stopping criteria are unchanged, so the Adam results
are identical to ``rfest_glm.GLM(distr='gaussian', output_nonlinearity='relu')`` up to floating point differences. Adam
does not give exact zeros, weights are only shrunk towards zero.

Several responses (e.g. all ROIs of a recording field) sharing one design matrix are fitted jointly: the weights of all
responses form one matrix ``W`` (n_features x n_responses), so every step is a matrix-matrix product instead of a
matrix-vector product (which is limited by memory bandwidth, i.e. about as slow as the batched product). Each response
has its own penalty, cost and stopping criterion, and since Adam acts element-wise and the cost of each response only
depends on its own parameters, every fit is the same as fitting the responses one by one.

Requires the optional ``rf`` dependencies (``pip install -e .[rf]``); JAX uses a GPU if one is available.
"""
import functools
import time

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, lax, value_and_grad
    from jax.example_libraries import optimizers
except ImportError:
    jax = jnp = None

# Number of Adam iterations per call of the compiled function. The stopping criteria are evaluated within the compiled
# function; whether all responses have stopped is checked after each chunk.
_CHUNK_SIZE = 25
_DTYPE = np.float32


def to_device(x):
    """Design matrix as a float32 JAX array (on the GPU if available), to transfer it only once for several fits."""
    if jax is None:
        raise ImportError('jax is required, install with `pip install -e .[rf]`.')
    return jnp.asarray(x, _DTYPE)


def _forwardpass(p, x):
    """``relu(X W + b)``: ``(n_samples, n_responses)``."""
    out = x @ p['stimulus'] + p['intercept']['stimulus'] + p['intercept']['global']
    return jnp.where(out > 0., out, 0.).astype(_DTYPE)


def _costs(p, x, y, beta, beta_l2=None):
    """Per-response MSE plus L1 penalty, as ``rfest.loss.loss_mse(y, r) + rfest.loss.loss_penalty(w, alpha=1., beta)``,
    plus ``beta_l2 / 2 * ||w||_2^2`` if given (elastic net, with ``beta`` the L1 part).
    Returns the summed cost (for the gradient) and the per-response costs."""
    mse = jnp.mean((y - _forwardpass(p, x)) ** 2, axis=0)
    l1 = jnp.sum(jnp.abs(p['stimulus']), axis=0)
    costs = mse + beta * l1
    if beta_l2 is not None:
        costs = costs + 0.5 * beta_l2 * jnp.sum(p['stimulus'] ** 2, axis=0)
    return costs.sum(), costs


@functools.lru_cache(maxsize=None)
def _get_chunk_runner(step_size, num_iters, min_iters, tolerance, atol):
    """Adam (init, run_chunk, get_params); `run_chunk` performs `_CHUNK_SIZE` steps and updates, per response, the
    stopping state and the parameters with the lowest train cost. Compiled once per settings and input shapes.

    Stopping as in RFEst: at step ``i``, with ``cost[i]`` the train cost before step ``i`` and ``params[i]`` the
    parameters after step ``i``, a fit stops if ``i > min_iters`` and the cost decreased by less than ``atol`` in each
    of the last ``tolerance - 1`` steps (``cost[i - tolerance:i]``), and returns ``params[argmin(cost[:i + 1])]``.
    """
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    def body(data, carry, i):
        opt_state, s = carry
        (_, cost), grad = value_and_grad(_costs, has_aux=True)(get_params(opt_state), data['x'], data['y'],
                                                               data['beta'])
        opt_state = opt_update(i, grad, opt_state)
        params = get_params(opt_state)

        live = s['active'] & (i < num_iters)
        if tolerance:
            stop = live & (i > min_iters) & (s['n_small'] >= tolerance - 1)
        else:
            stop = jnp.zeros_like(live)
        better = live & (cost < s['best_cost'])
        small = (s['prev_cost'] - cost) < atol
        s = dict(
            active=s['active'] & ~stop,
            n_iter=jnp.where(stop, i + 1, s['n_iter']),
            n_small=jnp.where(live, jnp.where(small, s['n_small'] + 1, 0), s['n_small']),
            prev_cost=jnp.where(live, cost, s['prev_cost']),
            best_cost=jnp.where(better, cost, s['best_cost']),
            best_iter=jnp.where(better, i, s['best_iter']),
            best_params=jax.tree_util.tree_map(lambda new, old: jnp.where(better, new, old), params,
                                               s['best_params']))
        return (opt_state, s), None

    @jit
    def run_chunk(opt_state, s, i0, data):
        (opt_state, s), _ = lax.scan(functools.partial(body, data), (opt_state, s), i0 + jnp.arange(_CHUNK_SIZE))
        return opt_state, s

    return opt_init, run_chunk, get_params


def fit_adam(x, y, p0, beta, num_iters=3000, step_size=1e-3, tolerance=20, atol=1e-6, min_iters=300):
    """Minimize the cost of each response with Adam from ``p0``; a fit stops early once its train cost decreased by
    less than ``atol`` in each of the last ``tolerance`` steps (after ``min_iters``).

    Args:
        x: Design matrix ``(n_samples, n_features)``.
        y: Responses ``(n_samples, n_responses)``.
        p0: Initial parameters: ``stimulus`` ``(n_features, n_responses)``, ``intercept`` dict with ``stimulus`` and
            ``global`` ``(n_responses,)``.
        beta: Penalty per response ``(n_responses,)``.

    Returns:
        tuple: Per response the parameters with the lowest train cost (same structure as ``p0``) and a list of info
        dicts (``n_iter``, ``best_iter``, ``cost``, ``stop`` = ``'train_stop'`` or ``'maxiter_stop'``).
    """
    opt_init, run_chunk, get_params = _get_chunk_runner(step_size, num_iters, min_iters, tolerance, atol)
    opt_state = opt_init(p0)
    data = dict(x=x, y=jnp.asarray(y, _DTYPE), beta=jnp.asarray(beta, _DTYPE))
    n_resp = data['y'].shape[1]
    s = dict(active=jnp.ones(n_resp, bool), n_iter=jnp.full(n_resp, num_iters, jnp.int32),
             n_small=jnp.zeros(n_resp, jnp.int32), prev_cost=jnp.full(n_resp, np.nan, _DTYPE),
             best_cost=jnp.full(n_resp, np.inf, _DTYPE), best_iter=jnp.zeros(n_resp, jnp.int32), best_params=p0)

    for i_chunk in range(int(np.ceil(num_iters / _CHUNK_SIZE))):
        opt_state, s = run_chunk(opt_state, s, i_chunk * _CHUNK_SIZE, data)
        if not bool(jnp.any(s['active'])):
            break

    active, n_iter, best_iter, best_cost = (np.asarray(s[k]) for k in ('active', 'n_iter', 'best_iter', 'best_cost'))
    infos = [dict(n_iter=int(n_iter[r]), best_iter=int(best_iter[r]), cost=float(best_cost[r]),
                  stop='maxiter_stop' if active[r] else 'train_stop') for r in range(n_resp)]
    return s['best_params'], infos


def compute_beta_max(x, y):
    """``beta_max = 2 max|X'(y - mean(y))| / n`` per response: the smallest penalty that gives all-zero weights for the
    linear model (only approximate with the ReLU). ``y``: ``(n_samples, n_responses)``."""
    x = to_device(x)
    xty = (x - x.mean(axis=0)).T @ (y - y.mean(axis=0))
    return 2 * np.asarray(jnp.max(jnp.abs(xty), axis=0), np.float64) / x.shape[0]


def fista_steps(x, precondition='none', step_scale=1., n_iter=100, margin=1.05, seed=0):
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


def _prox(vw, beta, beta_l2, step_w):
    """Proximal step of ``beta ||w||_1 + beta_l2 / 2 ||w||_2^2``: soft-thresholding, then shrinkage by ``1 + step
    beta_l2`` (exact for the elastic net, so the step sizes of the MSE alone still apply)."""
    return jnp.sign(vw) * jnp.maximum(jnp.abs(vw) - step_w * beta, 0.) / (1. + step_w * beta_l2)


def _gradient_mapping(w, vw, gw, beta, beta_l2, step_w):
    """``(w - prox(vw)) / step`` with ``vw = w - step gw``, computed from the gradient (``(gw + beta_l2 w + beta
    sign(vw)) / (1 + step beta_l2)`` where the prox keeps the weight, ``w / step`` where it zeroes it) instead of the
    difference, which is 0 once the update is below float32 resolution of ``w`` although the gradient is not."""
    return jnp.where(jnp.abs(vw) > step_w * beta,
                     (gw + beta_l2 * w + beta * jnp.sign(vw)) / (1. + step_w * beta_l2), w / step_w)


@functools.lru_cache(maxsize=None)
def _get_fista_chunk_runner(num_iters, gtol, tolerance, min_iters):
    """`run_chunk` performs `_CHUNK_SIZE` FISTA steps (with adaptive restart) and updates the per-response stopping
    state. Compiled once per settings and input shapes."""

    def body(data, s, i):
        x, y, beta, beta_l2 = data['x'], data['y'], data['beta'], data['beta_l2']
        step_w, step_b = data['step_w'], data['step_b']
        n = x.shape[0]
        # gradient of the MSE at the extrapolated point z
        out = x @ s['zw'] + s['zb']
        r = jnp.where(out > 0., out - y, 0.)
        gw, gb = (2. / n) * (x.T @ r), (2. / n) * r.sum(axis=0)
        # proximal step: soft-thresholding (and L2 shrinkage) of the weights, the intercept is not penalized
        vw = s['zw'] - step_w * gw
        w = _prox(vw, beta, beta_l2, step_w)
        b = s['zb'] - step_b * gb
        # gradient mapping G = (z - prox step) / step, in gradient units: zero iff z is optimal (KKT conditions)
        g_map_w, g_map_b = _gradient_mapping(s['zw'], vw, gw, beta, beta_l2, step_w), gb
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


def fit_fista(x, y, w0, b0, beta, steps, g_scale, beta_l2=None, num_iters=5000, gtol=1e-4, tolerance=3,
              min_iters=10):
    """Minimize the cost of each response with FISTA (accelerated proximal gradient with adaptive restart) from
    ``(w0, b0)``. Weights are exactly zero where the penalty wins.

    A fit stops once its optimality residual, the largest absolute entry of the gradient mapping (the proximal gradient
    step divided by the step size, in units of the gradient; zero at the optimum), divided by ``g_scale``, was at most
    ``gtol`` in ``tolerance`` consecutive steps (and after at least ``min_iters`` steps). Unlike the parameter change,
    this does not depend on the step size.

    Args:
        x: Design matrix ``(n_samples, n_features)``.
        y: Responses ``(n_samples, n_responses)``.
        w0, b0: Initial weights ``(n_features, n_responses)`` and intercepts ``(n_responses,)``.
        beta: (L1) penalty per response ``(n_responses,)``.
        steps: Step sizes ``(step_w (n_features,), step_b)``, see ``fista_steps``.
        g_scale: Gradient scale per response ``(n_responses,)``, e.g. ``beta_max``.
        beta_l2: L2 penalty per response ``(n_responses,)`` (elastic net: cost term ``beta_l2 / 2 ||w||_2^2``), or None
            (lasso).

    Returns:
        tuple: Weights, intercepts and a list of info dicts per response (``n_iter``, ``n_restart``, ``res`` = last
        relative optimality residual, ``cost`` = final train cost, ``stop`` = ``'train_stop'`` or ``'maxiter_stop'``).
    """
    run_chunk = _get_fista_chunk_runner(num_iters, gtol, tolerance, min_iters)
    l2 = beta_l2 is not None
    data = dict(x=x, y=jnp.asarray(y, _DTYPE), beta=jnp.asarray(beta, _DTYPE),
                beta_l2=jnp.asarray(beta_l2 if l2 else np.zeros_like(np.asarray(beta)), _DTYPE),
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
    costs = np.asarray(_costs(p, x, data['y'], data['beta'], data['beta_l2'] if l2 else None)[1])
    active, n_iter, n_restart, res = (np.asarray(s[k]) for k in ('active', 'n_iter', 'n_restart', 'res'))
    infos = [dict(n_iter=int(n_iter[r]), n_restart=int(n_restart[r]), res=float(res[r]), cost=float(costs[r]),
                  stop='maxiter_stop' if active[r] else 'train_stop') for r in range(n_resp)]
    return s['w'], s['b'], infos


def optimality_residual(x, y, weights, intercepts, beta, g_scale, steps, beta_l2=None):
    """Relative optimality residual of any solution (e.g. also of Adam): the largest absolute entry of the gradient
    mapping ``(w - prox(w - step g)) / step`` at the solution (as the FISTA stopping criterion, but at the solution
    instead of the extrapolated point), divided by ``g_scale``. Zero iff the lasso optimality conditions hold; unlike
    the plain violation of these conditions, it is continuous in ``w`` (a weight of 1e-12 that should be zero only
    counts as 1e-12 / step). Weights ``(n_features, n_models)``, intercepts, ``beta`` and ``g_scale``
    ``(n_models,)``, ``y`` ``(n_samples, n_models)``, ``steps`` from ``fista_steps`` (use the same for all solutions
    that are compared). ``beta_l2``: L2 penalty for the elastic net, as in ``fit_fista``."""
    x = to_device(x)
    w, b = jnp.asarray(weights, _DTYPE), jnp.asarray(intercepts, _DTYPE)
    beta = jnp.asarray(beta, _DTYPE)
    beta_l2 = jnp.zeros_like(beta) if beta_l2 is None else jnp.asarray(beta_l2, _DTYPE)
    step_w, step_b = jnp.asarray(steps[0], _DTYPE)[:, None], jnp.asarray(steps[1], _DTYPE)
    out = x @ w + b
    r = jnp.where(out > 0., out - jnp.asarray(y, _DTYPE), 0.)
    gw, gb = (2. / x.shape[0]) * (x.T @ r), (2. / x.shape[0]) * r.sum(axis=0)
    g_map_w = _gradient_mapping(w, w - step_w * gw, gw, beta, beta_l2, step_w)
    return np.asarray(jnp.maximum(jnp.max(jnp.abs(g_map_w), axis=0), jnp.abs(gb)) / jnp.asarray(g_scale, _DTYPE),
                      np.float64)


def fit_lasso_path(x, y, alphas_rel, solver='fista', eps_mean=1e-3, step_size=3e-3, step_size_finetune=1e-3,
                   precondition='none', step_scale=1., l1_ratio=1., **kwargs):
    """Lasso (or elastic-net) GLM ``y ~ relu(X w + b)`` for all penalties ``alphas_rel`` (relative to ``beta_max``),
    large to small.

    The first (largest) penalty starts from zero weights and ``b = max(mean(y), eps_mean)`` (so the zero-weight model
    predicts the mean response and the ReLU is not dead), the others warm-start from the previous penalty.
    ``beta_max`` (``compute_beta_max``) is the smallest penalty that gives all-zero weights for the linear model (only
    approximate with the ReLU). Several responses are fitted jointly (see module docstring), each with its own
    ``beta_max``.

    With ``l1_ratio < 1`` (FISTA only), the penalty ``beta = alpha_rel * beta_max / l1_ratio`` is split into the L1
    part ``l1_ratio * beta`` and the L2 part ``(1 - l1_ratio) * beta`` (cost term ``(1 - l1_ratio) * beta / 2
    ||w||_2^2``), so that ``alpha_rel = 1`` still gives all-zero weights (as ``alpha_max`` in scikit-learn's
    ``ElasticNet``).

    Two solvers minimize the same cost: ``'fista'`` (``fit_fista``, step sizes from ``fista_steps`` with
    ``precondition`` and ``step_scale``, residuals relative to ``beta_max``; gives exact zeros) and ``'adam'``
    (``fit_adam``, as RFEst; ``step_size`` for the first penalty, ``step_size_finetune`` for the warm-started ones).

    Args:
        x: Design matrix ``(n_samples, n_features)``, float32 (numpy or jax array, see ``to_device``).
        y: Response ``(n_samples,)`` or responses ``(n_samples, n_responses)``, scaled (e.g. by their std) but NOT
            centered, so that their mean is positive.
        alphas_rel: Penalties relative to ``beta_max``.
        solver: ``'fista'`` or ``'adam'``.
        l1_ratio: Share of the L1 penalty, ``1`` = lasso, ``0 < l1_ratio < 1`` = elastic net (FISTA only).
        **kwargs: Passed to ``fit_fista`` or ``fit_adam``.

    Returns:
        tuple: weights ``(n_alphas, n_features)`` and intercepts ``(n_alphas,)`` in the order of ``alphas_rel``, and a
        list of per-fit info dicts (see ``fit_fista`` / ``fit_adam``, plus ``alpha_idx`` and ``time_s``, the time of the
        joint fit). For 2D ``y``, weights ``(n_responses, n_alphas, n_features)``, intercepts
        ``(n_responses, n_alphas)`` and one list of info dicts per response.
    """
    x = to_device(x)
    alphas_rel = np.asarray(alphas_rel)
    if not 0. < l1_ratio <= 1.:
        raise ValueError(f'l1_ratio must be in (0, 1], got {l1_ratio}.')
    if l1_ratio < 1. and solver != 'fista':
        raise ValueError('The elastic net (l1_ratio < 1) is only implemented for FISTA.')
    y = np.asarray(y)  # beta_max and the initial intercept are computed before the cast to float32 (as in the sweep)
    single = y.ndim == 1
    y = y.reshape(y.shape[0], -1)
    n, n_feat = x.shape
    n_resp = y.shape[1]
    beta_max = compute_beta_max(x, y)
    p = dict(stimulus=jnp.zeros((n_feat, n_resp), _DTYPE),
             intercept=dict(stimulus=jnp.zeros(n_resp, _DTYPE),
                            **{'global': jnp.asarray(np.maximum(y.mean(axis=0), eps_mean), _DTYPE)}))
    y = y.astype(_DTYPE)

    if solver == 'fista':
        steps = fista_steps(x, precondition=precondition, step_scale=step_scale)
        w, b = p['stimulus'], p['intercept']['global']
    elif solver != 'adam':
        raise ValueError(f'Unknown solver {solver!r}.')

    weights, intercepts = np.zeros((n_resp, alphas_rel.size, n_feat), _DTYPE), np.zeros((n_resp, alphas_rel.size))
    infos = [[] for _ in range(n_resp)]
    for k, a in enumerate(np.argsort(alphas_rel)[::-1]):  # large penalty first
        t0 = time.time()
        beta = alphas_rel[a] * beta_max
        if solver == 'fista':
            # lasso: beta_l2 None, so the cost is computed exactly as before
            beta_l1, beta_l2 = (beta, None) if l1_ratio == 1. else (beta, (1. - l1_ratio) / l1_ratio * beta)
            w, b, info = fit_fista(x, y, w, b, beta_l1, steps, beta_max, beta_l2=beta_l2, **kwargs)
            weights[:, a], intercepts[:, a] = np.asarray(w).T, np.asarray(b, float)
        else:
            p, info = fit_adam(x, y, p, beta, step_size=step_size if k == 0 else step_size_finetune, **kwargs)
            weights[:, a] = np.asarray(p['stimulus']).T
            intercepts[:, a] = (np.asarray(p['intercept']['stimulus'], float)
                                + np.asarray(p['intercept']['global'], float))
        dt = time.time() - t0
        for r in range(n_resp):
            infos[r].append(dict(alpha_idx=int(a), **info[r], time_s=dt))
    if single:
        return weights[0], intercepts[0], infos[0]
    return weights, intercepts, infos


def predict(x, weights, intercepts):
    """``relu(X w + b)`` for weights ``(n_models, n_features)`` and intercepts ``(n_models,)``: ``(n_samples, n_models)``."""
    return np.maximum(np.asarray(x) @ np.asarray(weights).T + np.asarray(intercepts), 0.)
