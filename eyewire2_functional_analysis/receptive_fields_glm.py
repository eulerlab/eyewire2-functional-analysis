"""L1-penalized (lasso) GLM receptive fields with a ReLU output non-linearity, fitted with Adam in JAX.

Model: ``y ~ relu(X w + b)`` with ``X`` a pre-computed design matrix (e.g. the stimulus projected onto a separable
B-spline basis, see ``receptive_fields.compute_basis_features``). Cost: ``mean((y - y_pred) ** 2) + beta * ||w||_1``.

Based on RFEst (https://github.com/berenslab/RFEst, MIT license), via the RFEst GLM extension of
djimaging (eulerlab), which subclasses ``rfest.GLM`` but overrides its design matrix handling, initialization and
optimization. This module is that extension trimmed to the code path used here (pre-computed design matrix, fitted
intercept, ReLU output, no dev set, no CI), without the dependency on rfest: forward pass, cost (RFEst's ``loss_mse`` and
``loss_penalty`` with ``alpha=1``, re-implemented below), Adam chunks and stopping criteria are unchanged, so the results
are identical to ``rfest_glm.GLM(distr='gaussian', output_nonlinearity='relu')``. Adam does not give exact zeros, weights
are only shrunk towards zero.

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

# Number of Adam iterations per call of the compiled function. The stopping criteria are checked after each chunk, so up
# to `_CHUNK_SIZE - 1` iterations past the stopping point are computed and discarded.
_CHUNK_SIZE = 25
_DTYPE = np.float32


def _forwardpass(p, x):
    filters_output = jnp.array([jnp.squeeze(x @ p['stimulus']) + p['intercept']['stimulus']]).sum(0)
    out = filters_output + p['intercept']['global']
    return 1. * jnp.where(out > 0., out, 0.).astype(_DTYPE)


def _cost(p, x, y, beta):
    """MSE plus L1 penalty, as ``rfest.loss.loss_mse(y, r) + rfest.loss.loss_penalty(w, alpha=1., beta)``."""
    mse = jnp.mean((y - _forwardpass(p, x)) ** 2)
    l1 = jnp.linalg.norm(jnp.hstack([p['stimulus'].flatten()]), 1)
    return jnp.squeeze(mse + beta * l1)


@functools.lru_cache(maxsize=None)
def _get_chunk_runner(step_size):
    """Adam (init, run_chunk, get_params); `run_chunk` performs `_CHUNK_SIZE` steps and returns per step the train cost
    before the step and the parameters after the step. Compiled once per step size and input shapes."""
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    def body(data, opt_state, i):
        cost, grad = value_and_grad(_cost)(get_params(opt_state), data['x'], data['y'], data['beta'])
        opt_state = opt_update(i, grad, opt_state)
        return opt_state, (cost, get_params(opt_state))

    @jit
    def run_chunk(opt_state, i0, data):
        return lax.scan(functools.partial(body, data), opt_state, i0 + jnp.arange(_CHUNK_SIZE))

    return opt_init, run_chunk, get_params


def fit_adam(x, y, p0, beta, num_iters=3000, step_size=1e-3, tolerance=20, atol=1e-6, min_iters=300):
    """Minimize the cost with Adam from ``p0``; stop early once the train cost decreased by less than ``atol`` in each
    of the last ``tolerance`` steps (after ``min_iters``). Returns the parameters with the lowest train cost and an info
    dict (``n_iter``, ``best_iter``, ``cost``, ``stop`` = ``'train_stop'`` or ``'maxiter_stop'``)."""
    opt_init, run_chunk, get_params = _get_chunk_runner(step_size)
    opt_state = opt_init(p0)
    data = dict(x=x, y=jnp.asarray(y), beta=beta)

    n_chunks = int(np.ceil(num_iters / _CHUNK_SIZE))
    cost = np.full(n_chunks * _CHUNK_SIZE, np.nan)
    params_chunks = []
    stop = None
    for i_chunk in range(n_chunks):
        i0 = i_chunk * _CHUNK_SIZE
        opt_state, (c, params) = run_chunk(opt_state, i0, data)
        params_chunks.append(params)
        cost[i0:i0 + _CHUNK_SIZE] = np.asarray(c)
        for i in range(i0, min(i0 + _CHUNK_SIZE, num_iters)):
            if tolerance and i > min_iters and np.all(-np.diff(cost[i - tolerance:i]) < atol):
                stop = 'train_stop'
                break
        if stop is not None:
            break
    else:
        i = num_iters - 1
        stop = 'maxiter_stop'

    best = int(np.argmin(cost[:i + 1]))
    p = jax.tree_util.tree_map(lambda v: v[best % _CHUNK_SIZE], params_chunks[best // _CHUNK_SIZE])
    return p, dict(n_iter=i + 1, best_iter=best, cost=float(cost[best]), stop=stop)


def fit_lasso_path(x, y, alphas_rel, eps_mean=1e-3, step_size=3e-3, step_size_finetune=1e-3, **kwargs):
    """Lasso GLM ``y ~ relu(X w + b)`` for all penalties ``alphas_rel`` (relative to ``beta_max``), large to small.

    The first (largest) penalty starts from zero weights and ``b = max(mean(y), eps_mean)`` (so the zero-weight model
    predicts the mean response and the ReLU is not dead), the others warm-start from the previous penalty with
    ``step_size_finetune``. ``beta_max = 2 max|X'y| / n`` (centered) is the smallest penalty that gives all-zero weights
    for the linear model (only approximate with the ReLU).

    Args:
        x: Design matrix ``(n_samples, n_features)``, float32 (numpy or jax array).
        y: Response ``(n_samples,)``, scaled (e.g. by its std) but NOT centered, so that its mean is positive.
        alphas_rel: Penalties relative to ``beta_max``.
        **kwargs: Passed to ``fit_adam``.

    Returns:
        tuple: weights ``(n_alphas, n_features)`` and intercepts ``(n_alphas,)`` in the order of ``alphas_rel``, and a
        list of per-fit info dicts (see ``fit_adam``, plus ``alpha_idx`` and ``time_s``).
    """
    if jax is None:
        raise ImportError('jax is required, install with `pip install -e .[rf]`.')
    alphas_rel = np.asarray(alphas_rel)
    x = jnp.asarray(x, _DTYPE)
    y = np.asarray(y)  # beta_max and the initial intercept are computed before the cast to float32 (as in the sweep)
    n, n_feat = x.shape
    beta_max = 2 * float(jnp.max(jnp.abs((x - x.mean(axis=0)).T @ (y - y.mean())))) / n
    p = dict(stimulus=jnp.zeros((n_feat, 1), jnp.float32),
             intercept=dict(stimulus=jnp.float32(0.), **{'global': jnp.float32(max(float(y.mean()), eps_mean))}))
    y = y.astype(_DTYPE)

    weights, intercepts = np.zeros((alphas_rel.size, n_feat), _DTYPE), np.zeros(alphas_rel.size)
    infos = []
    for k, a in enumerate(np.argsort(alphas_rel)[::-1]):  # large penalty first
        t0 = time.time()
        p, info = fit_adam(x, y, p, beta=float(alphas_rel[a] * beta_max),
                           step_size=step_size if k == 0 else step_size_finetune, **kwargs)
        weights[a] = np.asarray(p['stimulus'])[:, 0]
        intercepts[a] = float(p['intercept']['stimulus']) + float(p['intercept']['global'])
        infos.append(dict(alpha_idx=int(a), **info, time_s=time.time() - t0))
        p = dict(stimulus=p['stimulus'], intercept=dict(p['intercept']))
    return weights, intercepts, infos


def predict(x, weights, intercepts):
    """``relu(X w + b)`` for weights ``(n_models, n_features)`` and intercepts ``(n_models,)``: ``(n_samples, n_models)``."""
    return np.maximum(np.asarray(x) @ np.asarray(weights).T + np.asarray(intercepts), 0.)
