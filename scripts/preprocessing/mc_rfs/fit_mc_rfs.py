"""Mouse-cam (MC) receptive fields of all 2P ROIs: lasso GLM penalty sweep with a ReLU output non-linearity.

Per ROI, a lasso GLM ``y ~ relu(X w + b)`` on the separable B-spline basis (t x y x x x channel) of the MC movie is
fitted with FISTA for every penalty of ``LASSO_ALPHAS`` (relative to the smallest penalty giving an all-zero RF, large to
small, warm-started; ``receptive_fields_glm.fit_lasso_path``, a trimmed, dependency-free version of the RFEst GLM
(https://github.com/berenslab/RFEst) as extended in djimaging). All ROIs of a field are fitted jointly (up to
``--batch-size``), which is much faster on a GPU than one by one and gives the same fits up to floating point noise:

- **Response**: positive gradient of ``mc_pp_trace`` at the movie frame times, divided by its train std for the fit.
- **Train** on both training sequences (``train1`` + ``train2``, 2 x 54 clips), **no cross-validation**.
- **Test** on the three repeats of the test sequence (``test1``-``test3``, 5 clips = 25 s each). Per penalty: correlation
  and MSE between prediction and response per repeat (the first ``DUR_PAST_S`` of each repeat are skipped), and the
  fraction of non-zero basis weights.
- The penalty with the best mean test correlation is selected per ROI (i.e. selected on the test data, so the test
  correlation of the selected penalty is optimistic). Its STRF is split per channel into sRF and tRF (SVD) and the sRFs
  of both channels are fitted jointly with a DoG (``receptive_fields.fit_srfs``).

The output is one row per ROI, documented in ``df_eyewire2_roi_level_mc_rf.md`` (copied next to the parquet).

Usage (one job per field, then merge)::

    python fit_mc_rfs.py --field GCL0        # fit (results cached per ROI, re-running resumes)
    python fit_mc_rfs.py --merge             # write the parquet once all fields are done

``--solver adam`` (with both commands) fits with Adam instead (RFEst, not converged to the lasso optimum) and writes
``df_eyewire2_roi_level_mc_rf_adam.parquet``. ``--solver enet`` fits an elastic net (L1 + L2 penalty, FISTA,
``--l1-ratio``, default 0.5) and writes ``df_eyewire2_roi_level_mc_rf_enet{l1_ratio}.parquet``.
``compare_mc_rf_solvers.py`` plots the solvers side by side.

Without ``--field`` and ``--merge``, all fields are fitted and then merged.
"""
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--field', action='append', help='Field(s) to fit, e.g. GCL0 (default: all).')
parser.add_argument('--merge', action='store_true', help='Only merge the cached results into the parquet.')
parser.add_argument('--data-2p', default=None, help='data-2p folder (default: data_loader.DATA_2P).')
parser.add_argument('--out-dir', default=None, help='Output folder (default: <repo>/data/mc_rfs).')
parser.add_argument('--n-cpus', type=int, default=None, help='Limit the number of CPU threads.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Max. number of ROIs of a field fitted jointly (lower it if the GPU runs out of memory).')
parser.add_argument('--solver', choices=('fista', 'adam', 'enet'), default='fista',
                    help='fista: lasso, FISTA; adam: lasso, Adam (RFEst); enet: elastic net, FISTA. Results are cached '
                         'separately, the parquet gets the suffix _adam / _enet{l1_ratio}.')
parser.add_argument('--l1-ratio', type=float, default=0.5,
                    help='--solver enet: share of the L1 penalty (0 < l1_ratio < 1, scikit-learn convention).')
args = parser.parse_args()

if args.n_cpus:
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = str(args.n_cpus)
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.n_cpus])

import hashlib  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eyewire2_functional_analysis import data_loader, io, receptive_fields as rf  # noqa: E402
from eyewire2_functional_analysis import receptive_fields_glm as glm  # noqa: E402
from eyewire2_functional_analysis.stimulus import mouse_cam as mc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_2P = args.data_2p or data_loader.DATA_2P
MC_ARRAY_DIR = os.path.join(DATA_2P, 'stimuli', 'mouse_cam_movies', 'mc_arrays')
OUT_DIR = args.out_dir or os.path.join(data_loader.DATA_ROOT, 'mc_rfs')
CACHE_DIR = os.path.join(HERE, 'results', 'cache')
OUT_NAME = 'df_eyewire2_roi_level_mc_rf'
KEY_COLS = ['experimenter', 'date', 'exp_num', 'field', 'roi_id']

# Stimulus / basis
BIN_XY = 2
PIXEL_SIZE_UM = mc.MC_PIXEL_SIZE_UM * BIN_XY
DT_FRAME = 1. / mc.MC_FRAME_RATE_HZ
DUR_PAST_S, DUR_FUTURE_S = 1.2, 0.3
TRAIN_BLOCKS = ('train1', 'train2')
TEST_BLOCKS = ('test1', 'test2', 'test3')
DF_T, DF_S = 15, 14
# Lasso path (penalties relative to alpha_max) and solver settings, per --solver
SOLVER_SETTINGS = dict(
    # FISTA (glm.fit_fista). Weaker penalties: rarely best, chance-level test cc, slow to converge.
    # bench_mc_rf_solvers.py (GCL0): gtol is never reached (fits stop at num_iters), 2000 iters too few, Jacobi
    # preconditioning best agreement with a converged reference on the selected penalty
    fista=(np.logspace(-3, 0, 7), dict(num_iters=5000, gtol=1e-3, tolerance=3, precondition='jacobi', eps_mean=1e-3)),
    # Adam (glm.fit_adam, as RFEst): the last Adam settings before the switch to FISTA (eb4d16c). Not converged to the
    # lasso optimum, i.e. additionally regularized by early stopping.
    adam=(np.logspace(-4, 0, 8), dict(num_iters=3000, min_iters=300, atol=1e-6, tolerance=20, step_size=1e-4,
                                      step_size_finetune=1e-3, eps_mean=1e-3)),
    # Elastic net (FISTA as above, glm.fit_lasso_path with l1_ratio): at the same alpha, the same L1 penalty as the
    # lasso plus an L2 penalty ((1 - l1_ratio) / l1_ratio times the L1 one), which keeps weak but consistent weights
    # (e.g. the weaker channel) that the lasso zeroes. One decade lower penalties than the lasso, since the L2 term
    # regularizes in addition and makes the weak penalties converge faster.
    enet=(np.logspace(-4, 0, 9), dict(num_iters=5000, gtol=1e-3, tolerance=3, precondition='jacobi', eps_mean=1e-3,
                                      l1_ratio=args.l1_ratio)),
)
SOLVER = args.solver
if SOLVER == 'enet' and not 0. < args.l1_ratio < 1.:
    parser.error(f'--l1-ratio must be in (0, 1) for the elastic net, got {args.l1_ratio}')
GLM_SOLVER = 'adam' if SOLVER == 'adam' else 'fista'
LASSO_ALPHAS, FIT_KW = SOLVER_SETTINGS[SOLVER]
OUT_NAME = OUT_NAME + {'fista': '', 'adam': '_adam', 'enet': f'_enet{args.l1_ratio:g}'}[SOLVER]
# Non-zero weights: exact for FISTA (lasso and elastic net); for Adam (no exact zeros) |w| > NONZERO_THRESH * max|w|
NONZERO_THRESH = 1e-2
# sRF fit: DoG per channel, soft prior pulling the centers towards the window center (= field center), soft penalty on
# the green-UV center distance relative to the center sizes
SRF_FIT_KIND = 'dog'
CENTER_PRIOR_SD_UM = 100.
CENTER_PRIOR_WEIGHT, COUPLING_WEIGHT = 0.05, 0.05
DOG_PARAMS = ('amp', 'polarity', 'x0', 'y0', 'sx', 'sy', 'theta', 'r2', 'amp_s', 'sx_s', 'sy_s', 'surround_ratio',
              'surround_rel_amp')

SETTINGS_KEY = hashlib.md5(repr((
    BIN_XY, DUR_PAST_S, DUR_FUTURE_S, TRAIN_BLOCKS, TEST_BLOCKS, DF_T, DF_S, list(LASSO_ALPHAS), SOLVER, sorted(FIT_KW.items()),
    NONZERO_THRESH, SRF_FIT_KIND, CENTER_PRIOR_SD_UM, CENTER_PRIOR_WEIGHT, COUPLING_WEIGHT, 'v2')).encode()).hexdigest()[:8]

blocks = mc.get_frame_blocks()
train_idxs = np.where(np.isin(blocks, TRAIN_BLOCKS))[0]
test_idxs = np.stack([np.where(blocks == b)[0] for b in TEST_BLOCKS])  # (3 repeats, n_frames_per_repeat)
shifts, rf_time = rf.get_rf_shifts(DT_FRAME, DUR_PAST_S, DUR_FUTURE_S)
n_eval_skip = int(np.sum(rf_time <= 0))  # the prediction needs the stimulus history


def cache_path(row):
    return os.path.join(CACHE_DIR, SETTINGS_KEY, f'{row.date}_{row.exp_num}_{row.field}_{row.roi_id}.pkl')


def get_response(row):
    """Positive-gradient response at the movie frame times."""
    tracetime = row.mc_pp_trace_t0 + np.arange(row.mc_pp_trace.size) * row.mc_pp_trace_dt
    return rf.compute_response(row.mc_pp_trace, tracetime, mc.get_frame_times(row.mc_triggertimes), kind='gradient')


def colwise_cc(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sum(a * b, axis=0) / np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))


def describe_strf(strf):
    """sRF / tRF per channel (SVD) and a joint DoG fit of the sRFs. sRF, tRF and the DoG amplitudes are flipped such
    that the fitted center is positive, i.e. the tRF carries the polarity of the center."""
    split = [rf.split_strf(strf[..., i]) for i in range(len(mc.MC_CHANNELS))]
    _, params = rf.fit_srfs(np.stack([s[0] for s in split]), kind=SRF_FIT_KIND,
                            center_prior_sd=CENTER_PRIOR_SD_UM / PIXEL_SIZE_UM,
                            center_prior_weight=CENTER_PRIOR_WEIGHT, coupling_weight=COUPLING_WEIGHT)
    out = {}
    for i, ch in enumerate(mc.MC_CHANNELS):
        sign = -1. if params[i]['polarity'] == -1 else 1.
        p = dict(params[i], amp=sign * params[i]['amp'], amp_s=sign * params[i]['amp_s'])
        out.update({f'mc_rf_{ch}_srf': sign * split[i][0], f'mc_rf_{ch}_trf': sign * split[i][1],
                    f'mc_rf_{ch}_svd_ev': split[i][2], **{f'mc_rf_{ch}_dog_{k}': float(p[k]) for k in DOG_PARAMS}})
    return out


def fit_rois(rows, x_train, feats, bases):
    """Lasso paths of several ROIs of one field (fitted jointly on the GPU, every fit is the same as fitting the ROIs one
    by one), their test scores and the RF of the best penalty. Returns a list of (row dict, cache extras) per ROI."""
    responses = [get_response(row) for row in rows]
    y_train = np.stack([r[train_idxs] for r in responses], axis=1).astype(np.float64)  # (n_train, n_rois)
    y_std = y_train.std(axis=0)
    weights, intercepts, infos = glm.fit_lasso_path(x_train, y_train / y_std, LASSO_ALPHAS, solver=GLM_SOLVER, **FIT_KW)
    return [evaluate_roi(responses[r], weights[r], intercepts[r], y_std[r], infos[r], feats, bases)
            for r in range(len(rows))]


def evaluate_roi(response, weights, intercepts, y_std, infos, feats, bases):
    """Test scores and the RF of the best penalty of one ROI. Returns (row dict, cache extras)."""
    infos = sorted(infos, key=lambda d: d['alpha_idx'])

    resp = np.stack([response[idxs] for idxs in test_idxs])[:, n_eval_skip:]  # (3, n_t)
    pred = np.stack([glm.predict(feats[idxs], weights, intercepts) for idxs in test_idxs]) * y_std  # (3, n_t, n_a)
    pred = pred[:, n_eval_skip:].transpose(2, 0, 1)  # (n_a, 3, n_t)
    cc = np.stack([colwise_cc(p.T, resp.T) for p in pred])  # (n_a, 3), NaN if the prediction is constant
    mse = np.mean((pred - resp[None]) ** 2, axis=2)  # (n_a, 3)
    if GLM_SOLVER == 'fista':  # exact zeros
        frac_nonzero = np.mean(weights != 0, axis=1)
    else:  # Adam only shrinks weights towards zero
        frac_nonzero = np.mean(np.abs(weights) > NONZERO_THRESH * np.abs(weights).max(axis=1, keepdims=True), axis=1)

    cc_mean = np.array([np.nanmean(c) if np.any(np.isfinite(c)) else np.nan for c in cc])  # over repeats
    best = int(np.nanargmax(cc_mean)) if np.any(np.isfinite(cc_mean)) else -1
    strf = rf.basis_to_strf(weights[[max(best, 0)]] * y_std, *bases)[0]
    if best < 0:
        strf = np.full_like(strf, np.nan)

    out = dict(
        mc_rf_alphas=LASSO_ALPHAS.copy(), mc_rf_cc=cc_mean, mc_rf_mse=mse.mean(axis=1), mc_rf_cc_repeats=cc,
        mc_rf_mse_repeats=mse, mc_rf_frac_nonzero=frac_nonzero,
        mc_rf_n_iter=np.array([d['n_iter'] for d in infos]),
        mc_rf_converged=np.array([d['stop'] != 'maxiter_stop' for d in infos]),
        mc_rf_opt_res=np.array([d.get('res', np.nan) for d in infos]),
        mc_rf_best_alpha_idx=best, mc_rf_best_alpha=LASSO_ALPHAS[best] if best >= 0 else np.nan,
        mc_rf_best_cc=cc_mean[best] if best >= 0 else np.nan, mc_rf_best_mse=mse[best].mean() if best >= 0 else np.nan,
        mc_rf_trf_t0=float(rf_time[0]), mc_rf_trf_dt=DT_FRAME, mc_rf_pixel_size_um=PIXEL_SIZE_UM,
        **describe_strf(strf))
    extras = dict(weights=weights, intercepts=intercepts, y_std=y_std, infos=infos)
    return out, extras


def fit_field(df_field):
    todo = [row for _, row in df_field.iterrows() if not os.path.exists(cache_path(row))]
    print(f'{df_field.field.iloc[0]}: {len(df_field) - len(todo)}/{len(df_field)} ROIs cached', flush=True)
    if not todo:
        return
    stim = rf.standardize_stim(mc.load_mc_movie(df_field.mc.iloc[0], MC_ARRAY_DIR, bin_xy=BIN_XY))
    bases = (rf.bspline_basis(shifts.size, DF_T), rf.bspline_basis(stim.shape[1], DF_S),
             rf.bspline_basis(stim.shape[2], DF_S))
    feats = rf.compute_basis_features(stim, shifts, *bases)  # (n_frames, n_features) float32
    x_train = glm.to_device(feats[train_idxs])  # transferred once per field
    os.makedirs(os.path.join(CACHE_DIR, SETTINGS_KEY), exist_ok=True)
    for i0 in range(0, len(todo), args.batch_size):
        batch = todo[i0:i0 + args.batch_size]
        t0 = time.time()
        results = fit_rois(batch, x_train, feats, bases)
        for i, (row, (out, extras)) in enumerate(zip(batch, results)):
            path = cache_path(row)
            with open(path + '.tmp', 'wb') as f:
                pickle.dump(dict(row=out, **extras), f)
            os.replace(path + '.tmp', path)
            print(f'  ROI {row.roi_id} ({i0 + i + 1}/{len(todo)}): best cc {out["mc_rf_best_cc"]:.3f} '
                  f'(alpha idx {out["mc_rf_best_alpha_idx"]})', flush=True)
        print(f'  {len(batch)} ROIs in {time.time() - t0:.0f} s', flush=True)


def merge(df_rois):
    missing = [f'{r.field} {r.roi_id}' for _, r in df_rois.iterrows() if not os.path.exists(cache_path(r))]
    if missing:
        raise FileNotFoundError(f'{len(missing)} ROIs not fitted yet (settings {SETTINGS_KEY}), e.g. {missing[:5]}')
    rows = []
    for _, r in df_rois.iterrows():
        with open(cache_path(r), 'rb') as f:
            rows.append(dict(r[KEY_COLS], **pickle.load(f)['row']))
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'{OUT_NAME}.parquet')
    io.serialize_numpy_arrays(df).to_parquet(out_path, engine='fastparquet', index=False)
    shutil.copy(os.path.join(HERE, 'df_eyewire2_roi_level_mc_rf.md'), os.path.join(OUT_DIR, f'{OUT_NAME}.md'))
    print(f'Wrote {out_path}: {df.shape[0]} ROIs x {df.shape[1]} columns')


if __name__ == '__main__':
    df_rois = data_loader.load_df_rois(DATA_2P).reset_index(drop=True)
    print(f'Settings key {SETTINGS_KEY}, {len(df_rois)} ROIs', flush=True)
    if not args.merge:
        fields = args.field or list(df_rois.field.unique())
        for field in fields:
            fit_field(df_rois[df_rois.field == field])
    if args.merge or not args.field:
        merge(df_rois)
