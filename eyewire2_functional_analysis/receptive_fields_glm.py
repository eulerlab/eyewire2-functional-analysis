"""L1-penalized (lasso) GLM receptive fields with a ReLU output non-linearity, fitted with Adam in JAX.

Model: ``y ~ relu(X w + b)`` with ``X`` a pre-computed design matrix (e.g. the stimulus projected onto a separable
B-spline basis, see ``receptive_fields.compute_basis_features``). Cost: ``mean((y - y_pred) ** 2) + beta * ||w||_1``.

Based on RFEst (https://github.com/berenslab/RFEst, MIT license), via the RFEst GLM extension of
djimaging (eulerlab), which subclasses ``rfest.GLM`` but overrides its design matrix handling, initialization and
optimization. This module is that extension trimmed to the code path used here (pre-computed design matrix, fitted
intercept, ReLU output, no dev set, no CI), without the dependency on rfest: forward pass, cost (RFEst's ``loss_mse`` and
``loss_penalty`` with ``alpha=1``, re-implemented below), Adam and stopping criteria are unchanged, so the results
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


def _costs(p, x, y, beta):
    """Per-response MSE plus L1 penalty, as ``rfest.loss.loss_mse(y, r) + rfest.loss.loss_penalty(w, alpha=1., beta)``.
    Returns the summed cost (for the gradient) and the per-response costs."""
    mse = jnp.mean((y - _forwardpass(p, x)) ** 2, axis=0)
    l1 = jnp.sum(jnp.abs(p['stimulus']), axis=0)
    costs = mse + beta * l1
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


def fit_lasso_path(x, y, alphas_rel, eps_mean=1e-3, step_size=3e-3, step_size_finetune=1e-3, **kwargs):
    """Lasso GLM ``y ~ relu(X w + b)`` for all penalties ``alphas_rel`` (relative to ``beta_max``), large to small.

    The first (largest) penalty starts from zero weights and ``b = max(mean(y), eps_mean)`` (so the zero-weight model
    predicts the mean response and the ReLU is not dead), the others warm-start from the previous penalty with
    ``step_size_finetune``. ``beta_max = 2 max|X'y| / n`` (centered) is the smallest penalty that gives all-zero weights
    for the linear model (only approximate with the ReLU). Several responses are fitted jointly (see module docstring),
    each with its own ``beta_max``.

    Args:
        x: Design matrix ``(n_samples, n_features)``, float32 (numpy or jax array, see ``to_device``).
        y: Response ``(n_samples,)`` or responses ``(n_samples, n_responses)``, scaled (e.g. by their std) but NOT
            centered, so that their mean is positive.
        alphas_rel: Penalties relative to ``beta_max``.
        **kwargs: Passed to ``fit_adam``.

    Returns:
        tuple: weights ``(n_alphas, n_features)`` and intercepts ``(n_alphas,)`` in the order of ``alphas_rel``, and a
        list of per-fit info dicts (see ``fit_adam``, plus ``alpha_idx`` and ``time_s``, the time of the joint fit).
        For 2D ``y``, weights ``(n_responses, n_alphas, n_features)``, intercepts ``(n_responses, n_alphas)`` and one
        list of info dicts per response.
    """
    x = to_device(x)
    alphas_rel = np.asarray(alphas_rel)
    y = np.asarray(y)  # beta_max and the initial intercept are computed before the cast to float32 (as in the sweep)
    single = y.ndim == 1
    y = y.reshape(y.shape[0], -1)
    n, n_feat = x.shape
    n_resp = y.shape[1]
    xty = (x - x.mean(axis=0)).T @ (y - y.mean(axis=0))
    beta_max = 2 * np.asarray(jnp.max(jnp.abs(xty), axis=0), np.float64) / n
    p = dict(stimulus=jnp.zeros((n_feat, n_resp), _DTYPE),
             intercept=dict(stimulus=jnp.zeros(n_resp, _DTYPE),
                            **{'global': jnp.asarray(np.maximum(y.mean(axis=0), eps_mean), _DTYPE)}))
    y = y.astype(_DTYPE)

    weights, intercepts = np.zeros((n_resp, alphas_rel.size, n_feat), _DTYPE), np.zeros((n_resp, alphas_rel.size))
    infos = [[] for _ in range(n_resp)]
    for k, a in enumerate(np.argsort(alphas_rel)[::-1]):  # large penalty first
        t0 = time.time()
        p, info = fit_adam(x, y, p, beta=alphas_rel[a] * beta_max,
                           step_size=step_size if k == 0 else step_size_finetune, **kwargs)
        weights[:, a] = np.asarray(p['stimulus']).T
        intercepts[:, a] = np.asarray(p['intercept']['stimulus'], float) + np.asarray(p['intercept']['global'], float)
        dt = time.time() - t0
        for r in range(n_resp):
            infos[r].append(dict(alpha_idx=int(a), **info[r], time_s=dt))
    if single:
        return weights[0], intercepts[0], infos[0]
    return weights, intercepts, infos


def predict(x, weights, intercepts):
    """``relu(X w + b)`` for weights ``(n_models, n_features)`` and intercepts ``(n_models,)``: ``(n_samples, n_models)``."""
    return np.maximum(np.asarray(x) @ np.asarray(weights).T + np.asarray(intercepts), 0.)
