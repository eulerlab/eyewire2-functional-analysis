"""Benchmark the lasso-path solvers of ``fit_mc_rfs.py`` on one field: timing, iterations, optimality and agreement
with a tightly converged FISTA reference, for FISTA (stopping tolerance, preconditioning, step size) and Adam (step
sizes) configs (``CONFIGS``, edit as needed).

Usage::

    python bench_mc_rf_solvers.py --field GCL0 [--n-rois 32] [--no-adam]

Prints one line per config; all fits are saved to ``--out`` (pickle). Columns:

- ``s``: wall time of the joint fit of all ROIs (compilation excluded, by a warm-up fit at one penalty).
- ``iters``: median over ROIs of the iterations summed over the path; ``maxit``: fraction of fits that hit the maximum.
- ``res``: optimality residual (largest violation of the lasso optimality conditions relative to ``beta_max``,
  ``glm.optimality_residual``, the same measure for all solvers), max over penalties, median and max over ROIs. Check
  that the reference has the smallest residuals, otherwise raise ``--ref-iters``.
- ``dcost``: relative excess train cost vs. the reference, median and max over ROIs x penalties.
- ``same_best``: fraction of ROIs with the same best penalty as the reference; ``dcc``: median (best test cc - reference
  best test cc); ``rf_r``: median correlation of the weights with the reference at the reference's best penalty.
"""
import argparse
import os
import pickle
import time

import numpy as np

from eyewire2_functional_analysis import data_loader, receptive_fields as rf
from eyewire2_functional_analysis import receptive_fields_glm as glm
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--field', default='GCL0')
parser.add_argument('--n-rois', type=int, default=None, help='Only the first n ROIs of the field (default: all).')
parser.add_argument('--no-adam', action='store_true', help='Skip the (slow) Adam configs.')
parser.add_argument('--ref-iters', type=int, default=20000, help='Max. iterations per penalty of the reference.')
parser.add_argument('--data-2p', default=None, help='data-2p folder (default: data_loader.DATA_2P).')
parser.add_argument('--out', default=None, help='Output pickle (default: bench_<field>.pkl in the working directory).')
args = parser.parse_args()

# Same stimulus / basis / penalties as fit_mc_rfs.py (keep in sync)
BIN_XY, DUR_PAST_S, DUR_FUTURE_S, DF_T, DF_S = 2, 1.2, 0.3, 15, 14
TRAIN_BLOCKS, TEST_BLOCKS = ('train1', 'train2'), ('test1', 'test2', 'test3')
LASSO_ALPHAS = np.logspace(-3, 0, 7)  # weaker penalties: rarely best, chance-level test cc, slow to converge

FISTA = dict(num_iters=5000, gtol=1e-4, tolerance=3)
ADAM = dict(num_iters=3000, min_iters=300, atol=1e-6, tolerance=20, step_size=1e-4, step_size_finetune=1e-3)
# gtol ~1e-5 is about the float32 noise floor of the gradient (fits there run to the maximum, so the reference is
# simply "as converged as float32 allows")
REFERENCE = ('reference', 'fista', dict(FISTA, num_iters=args.ref_iters, gtol=1e-5, precondition='jacobi'))
CONFIGS = [
    ('fista gtol 1e-3', 'fista', dict(FISTA, gtol=1e-3)),
    ('fista gtol 1e-4', 'fista', FISTA),
    ('fista step x2', 'fista', dict(FISTA, step_scale=2.)),
    ('fista jac gtol 1e-3', 'fista', dict(FISTA, gtol=1e-3, precondition='jacobi')),
    ('fista jac gtol 1e-4', 'fista', dict(FISTA, precondition='jacobi')),
    ('fista jac gtol 3e-5', 'fista', dict(FISTA, gtol=3e-5, precondition='jacobi')),
    ('fista jac step x2', 'fista', dict(FISTA, precondition='jacobi', step_scale=2.)),
    ('fista jac 2000 it', 'fista', dict(FISTA, gtol=1e-3, precondition='jacobi', num_iters=2000)),
    ('adam 1e-4/1e-3 (old)', 'adam', ADAM),
    ('adam 1e-3/1e-3', 'adam', dict(ADAM, step_size=1e-3)),
]
if args.no_adam:
    CONFIGS = [c for c in CONFIGS if c[1] != 'adam']

# Data
data_2p = args.data_2p or data_loader.DATA_2P
df = data_loader.load_df_rois(data_2p).reset_index(drop=True)
df = df[df.field == args.field].iloc[:args.n_rois]
print(f'{args.field}: {len(df)} ROIs', flush=True)

blocks = mc.get_frame_blocks()
train_idxs = np.where(np.isin(blocks, TRAIN_BLOCKS))[0]
test_idxs = np.stack([np.where(blocks == b)[0] for b in TEST_BLOCKS])
shifts, rf_time = rf.get_rf_shifts(1. / mc.MC_FRAME_RATE_HZ, DUR_PAST_S, DUR_FUTURE_S)
n_eval_skip = int(np.sum(rf_time <= 0))

stim = rf.standardize_stim(mc.load_mc_movie(
    df.mc.iloc[0], os.path.join(data_2p, 'stimuli', 'mouse_cam_movies', 'mc_arrays'), bin_xy=BIN_XY))
bases = (rf.bspline_basis(shifts.size, DF_T), rf.bspline_basis(stim.shape[1], DF_S),
         rf.bspline_basis(stim.shape[2], DF_S))
feats = rf.compute_basis_features(stim, shifts, *bases)
x_train = glm.to_device(feats[train_idxs])
x_test = feats[test_idxs]  # (3, n_t, n_feat)


def get_response(row):
    tracetime = row.mc_pp_trace_t0 + np.arange(row.mc_pp_trace.size) * row.mc_pp_trace_dt
    return rf.compute_response(row.mc_pp_trace, tracetime, mc.get_frame_times(row.mc_triggertimes), kind='gradient')


responses = np.stack([get_response(row) for _, row in df.iterrows()], axis=1)  # (n_frames, n_rois)
y_train = responses[train_idxs].astype(np.float64)
y_train /= y_train.std(axis=0)
y_test = responses[test_idxs][:, n_eval_skip:]  # (3, n_t, n_rois)
beta_max = glm.compute_beta_max(x_train, y_train)
n_rois, n_alphas = len(df), LASSO_ALPHAS.size
# per (ROI, penalty) model, ROI-major as weights.reshape(-1, n_feat)
y_models = np.repeat(y_train, n_alphas, axis=1).astype(np.float32)
beta_models = (beta_max[:, None] * LASSO_ALPHAS[None]).reshape(-1)
g_scale_models = np.repeat(beta_max, n_alphas)
res_steps = glm.fista_steps(x_train)  # one (unpreconditioned) step for the residuals of all configs


def test_cc(weights, intercepts):
    """Mean test correlation over the repeats, (n_rois, n_alphas); NaN for constant predictions."""
    n_feat = weights.shape[2]
    pred = np.maximum(x_test @ weights.reshape(-1, n_feat).T + intercepts.reshape(-1), 0.)[:, n_eval_skip:]
    pred = pred.reshape(*pred.shape[:2], n_rois, n_alphas)  # (3, n_t, n_rois, n_alphas)
    p, y = pred - pred.mean(axis=1, keepdims=True), (y_test - y_test.mean(axis=1, keepdims=True))[..., None]
    with np.errstate(invalid='ignore', divide='ignore'):
        cc = np.sum(p * y, axis=1) / np.sqrt(np.sum(p ** 2, axis=1) * np.sum(y ** 2, axis=1))
    return np.nanmean(cc, axis=0) if np.isfinite(cc).any() else cc[0]


def run(solver, kw):
    glm.fit_lasso_path(x_train, y_train, [1.], solver=solver, eps_mean=1e-3, **kw)  # compile
    t0 = time.time()
    w, b, infos = glm.fit_lasso_path(x_train, y_train, LASSO_ALPHAS, solver=solver, eps_mean=1e-3, **kw)
    dt = time.time() - t0
    infos = [sorted(i, key=lambda d: d['alpha_idx']) for i in infos]
    cost = np.array([[d['cost'] for d in i] for i in infos])
    n_iter = np.array([[d['n_iter'] for d in i] for i in infos])
    maxit = np.array([[d['stop'] == 'maxiter_stop' for d in i] for i in infos])
    res = glm.optimality_residual(x_train, y_models, w.reshape(-1, w.shape[2]).T, b.reshape(-1), beta_models,
                                  g_scale_models, res_steps).reshape(n_rois, n_alphas)
    cc = test_cc(w, b)
    best = np.array([np.nanargmax(c) if np.isfinite(c).any() else -1 for c in cc])
    return dict(time_s=dt, weights=w, intercepts=b, cost=cost, n_iter=n_iter, maxit=maxit, res=res, cc=cc, best=best)


def corr_rows(a, b):
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sum(a * b, axis=1) / np.sqrt(np.sum(a ** 2, axis=1) * np.sum(b ** 2, axis=1))


results = {}
print(f'{"config":<22}{"s":>7}{"iters":>8}{"maxit":>7}{"res med":>9}{"res max":>9}{"dcost med":>11}{"dcost max":>11}'
      f'{"same_best":>11}{"dcc":>8}{"rf_r":>7}', flush=True)
for name, solver, kw in [REFERENCE] + CONFIGS:
    r = results[name] = dict(run(solver, kw), solver=solver, kw=kw)
    ref = results[REFERENCE[0]]
    rel = (r['cost'] - ref['cost']) / np.abs(ref['cost'])
    res = r['res'].max(axis=1)
    rows = np.arange(n_rois)
    ok = ref['best'] >= 0
    best_cc = np.array([c[i] if i >= 0 else np.nan for c, i in zip(r['cc'], r['best'])])
    ref_cc = np.array([c[i] if i >= 0 else np.nan for c, i in zip(ref['cc'], ref['best'])])
    rf_r = corr_rows(r['weights'][rows[ok], ref['best'][ok]], ref['weights'][rows[ok], ref['best'][ok]])
    print(f'{name:<22}{r["time_s"]:>7.1f}{np.median(r["n_iter"].sum(axis=1)):>8.0f}{r["maxit"].mean():>7.2f}'
          f'{np.median(res):>9.1e}{np.max(res):>9.1e}{np.nanmedian(rel):>11.1e}{np.nanmax(rel):>11.1e}'
          f'{np.mean(r["best"] == ref["best"]):>11.2f}{np.nanmedian(best_cc - ref_cc):>8.3f}{np.nanmedian(rf_r):>7.3f}',
          flush=True)

out = args.out or f'bench_{args.field}.pkl'
with open(out, 'wb') as f:
    pickle.dump(dict(field=args.field, roi_ids=df.roi_id.to_numpy(), alphas=LASSO_ALPHAS, beta_max=beta_max,
                     results=results), f)
print(f'Saved {out}')
