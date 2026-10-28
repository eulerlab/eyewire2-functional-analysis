# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: eyewire2-functional-analysis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparing RF estimation methods on the mouse-cam movie (example cells)
#
# Same data, response and train/test split as `mouse_cam_rfs.py`. Methods:
# - `sta`: baseline STA on the positive gradient of `mc_pp_trace`.
# - `sta_clipXX`: STA with the response clipped at its XXth percentile (per ROI, train frames),
#   so that a few strong transients (e.g. at scene cuts) do not dominate the average.
# - `wsta`: whitened STA, ``C_t^-1 STA C_s^-1`` with a separable (temporal x spatio-chromatic) approximation
#   of the stimulus covariance and ridge-regularized inverses.
# - `ridge` / `lasso` / `enet`: linear-Gaussian GLM with L2 / L1 / L1+L2 penalty on a separable B-spline basis (t x y x x x channel).
# - `ald` / `asd`: rfest evidence optimization (automatic locality / smoothness determination) of the chromatic sRF
#   (2 x 14 x 14 at 50 µm), using the per-channel tRFs of the ridge fit to project the stimulus in time
#   (two-stage, separable per channel). The green-UV prior covariance is a free 2x2 matrix.
#
# Hyperparameters are chosen per ROI by 2-fold cross-validation (train1 <-> train2, correlation of the prediction).
# Quality measures as before: `split_cc` (STRFs from train1 vs. train2, past lags) and `test_cc`
# (linear prediction vs. mean test response). Note that `split_cc` is inflated for the smooth basis/prior methods,
# `test_cc` is the fairer comparison.
#
# ## Status (2026-10-27)
#
# First run on the 12 example cells (`figures/methods/`, `results/mc_rf_methods_examples.pkl`), median `test_cc`:
# sta 0.23 | sta_clip99 0.21 | sta_clip95 0.17 | sta_clip90 0.16 | wsta 0.33 | ridge 0.34 | lasso 0.37.
# - Clipping the response makes the STA worse at every level, so the strong transients carry signal.
# - wsta and ridge give localized sRFs and biphasic tRFs (the STA tRFs are slow ramps, blurred by the movie's
#   autocorrelation). Ridge chose reg ~1-3 (grid edge is far away); wsta chose reg_s = 100 = grid edge in that run.
# - Lasso has the best test_cc, but its sRFs are single B-spline bumps (~230 µm for every cell), so the sRF size is
#   set by the basis, not the cell.
# - ALD/ASD in that run gave salt-and-pepper sRFs (split_cc ~0): the evidence treats the 16k frames as independent.
#   Fixed afterwards with `EB_SAMPLE_FACTOR`, tested on GCL0 rois 5/68/10/21 (test_cc):
#   ALD k=1: 0.49/0.10/0.16/0.28 | k=6: 0.47/0.36/0.32/0.31 | k=20: 0.52/0.36/0.32/0.33 | ridge: 0.48/0.32/0.35/0.28.
# - rfest's evidence was replaced by a Cholesky-only version (identical values; JAX's CPU LU/inv/pinv are very slow).
#
# The current script (ALD/ASD with k=20, new `enet` method, wsta grid up to 1e3) has NOT been run end-to-end yet;
# the figures/pickle on disk are from the first run. Runtime on an 8-core laptop: ~30-45 min per field.
# lasso and enet take ~15 min each, nearly independent of the number of ROIs (5880 x 5880 Gram matrix, FISTA);
# ALD/ASD take ~1-2 min per ROI and prior (3 folds). Memory: features ~430 MB + 3 Gram matrices ~280 MB each per field.
#
# ## Next steps
#
# 1. Run this script on the examples (cluster) and check: ALD/ASD sRFs smooth and local? enet sRFs not
#    basis-limited (otherwise drop lasso/enet or use a finer spatial basis)? wsta no longer at the grid edge?
# 2. Pick `EB_SAMPLE_FACTOR` properly: CV over e.g. {10, 20, 50}, or derive it from the autocorrelation of
#    response x stimulus (response alone gives ~5-10 frames, k=20 was better than 6).
# 3. Optional: ALD at full 25 µm resolution (2 x 28 x 28 coefficients, ~64x the cost of 14 x 14) or on a crop around
#    the ridge RF center; joint spatiotemporal ALD instead of the two-stage fit.
# 4. Choose a method, then go large scale: move the chosen code (e.g. `ChromaticEB`, `fit_basis_glm`) into
#    `receptive_fields.py`, use it in `mouse_cam_rfs.py` instead of the STA for all ROIs, and redefine `rf_ok`
#    (split_cc thresholds are not comparable across methods; e.g. test_cc vs. the wrong-sequence control).
# 5. Re-check tRF polarity against the chirp supergroups (several STA polarities flip after decorrelation,
#    e.g. the chirp-OFF cells GCL0 21/25 become OFF with ridge/lasso).
# 6. If "GLM with L1 loss" meant a robust (absolute-error) loss rather than an L1 penalty: not tested yet.

# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from eyewire2_functional_analysis import data_loader, paths, receptive_fields as rf, style
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

style.set_rc_params()

# %%
HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FIG_DIR = os.path.join(HERE, 'figures', 'methods')
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(FIG_DIR, exist_ok=True)

DATA_2P = os.path.join(paths.PROJECT_ROOT, '..', 'huggingface', 'eyewire2-data', 'data-2p')
if not os.path.exists(DATA_2P):
    DATA_2P = data_loader.DATA_2P
MC_ARRAY_DIR = os.path.join(DATA_2P, 'stimuli', 'mouse_cam_movies', 'mc_arrays')

BIN_XY = 2
PIXEL_SIZE_UM = mc.MC_PIXEL_SIZE_UM * BIN_XY
DT_FRAME = 1. / mc.MC_FRAME_RATE_HZ
DUR_PAST_S, DUR_FUTURE_S = 1.2, 0.3
TRAIN_BLOCKS = ('train1', 'train2')
TEST_BLOCKS = ('test1', 'test2', 'test3')
CH_COLORS = {'green': 'forestgreen', 'uv': 'darkviolet'}

# Examples: best STAs, STAs with a polarity opposite to the chirp supergroup, and some medium-quality STAs
EXAMPLES = [
    ('GCL0', 5), ('GCL0', 68), ('GCL0', 10), ('GCL1', 46), ('GCL2', 54), ('GCL3', 26),
    ('GCL4', 63), ('GCL0', 9), ('GCL0', 21), ('GCL0', 25), ('GCL1', 47), ('GCL1', 33),
]

CLIP_QS = (99, 95, 90)
WSTA_REGS = np.logspace(-2, 3, 6)  # relative to the mean eigenvalue, for both C_t and C_s
DF_T, DF_S = 15, 14  # B-spline basis functions in time and per spatial dimension
RIDGE_REGS = np.logspace(-4, 3, 15)
LASSO_ALPHAS = np.logspace(-3, -0.5, 8)  # relative to alpha_max
L1_RATIOS = {'lasso': 1., 'enet': 0.5}
EB_BIN = 2  # additional spatial binning for ALD / ASD (rfest scales cubically with the number of coefficients)
# The evidence assumes independent samples, but response and movie are autocorrelated (response alone: ~5-10 frames).
# Without correction, the noise is underestimated and the prior too weak (salt-and-pepper sRFs).
EB_SAMPLE_FACTOR = 20

# %% [markdown]
# # Data

# %%
df_rois = data_loader.load_df_rois(DATA_2P).reset_index(drop=True)
df_ex = pd.DataFrame(EXAMPLES, columns=['field', 'roi_id']).merge(
    df_rois[['field', 'roi_id', 'supergroup', 'group_id']], on=['field', 'roi_id'], validate='one_to_one')

blocks = mc.get_frame_blocks()
fold_idxs = {'train1': np.where(blocks == 'train1')[0], 'train2': np.where(blocks == 'train2')[0],
             'train': np.where(np.isin(blocks, TRAIN_BLOCKS))[0]}
folds = ('train1', 'train2', 'train')
val_fold = {'train1': 'train2', 'train2': 'train1'}
test_idxs = np.stack([np.where(blocks == b)[0] for b in TEST_BLOCKS])
shifts, rf_time = rf.get_rf_shifts(DT_FRAME, DUR_PAST_S, DUR_FUTURE_S)
past = rf_time <= 0
n_eval_skip = np.sum(past)


def get_responses(df_field):
    """Positive-gradient responses at the movie frame times, shape (n_frames, n_rois)."""
    frame_times = mc.get_frame_times(df_field.mc_triggertimes.iloc[0])
    responses = []
    for _, row in df_field.iterrows():
        tracetime = row.mc_pp_trace_t0 + np.arange(row.mc_pp_trace.size) * row.mc_pp_trace_dt
        responses.append(rf.compute_response(row.mc_pp_trace, tracetime, frame_times, kind='gradient'))
    return np.stack(responses, axis=1)


def colwise_cc(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    return np.sum(a * b, axis=0) / np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))


def evaluate(stim, responses, strfs):
    """split_cc and test_cc for STRFs {fold: (n_rois, n_shifts, ...)}."""
    a, b = strfs['train1'][:, past], strfs['train2'][:, past]
    split_cc = colwise_cc(a.reshape(len(a), -1).T, b.reshape(len(b), -1).T)
    pred = np.mean([rf.predict_linear(stim, strfs['train'], shifts, idxs) for idxs in test_idxs], axis=0)
    test_cc = colwise_cc(pred[n_eval_skip:], responses[test_idxs].mean(axis=0)[n_eval_skip:])
    return dict(split_cc=split_cc, test_cc=test_cc)


def select_by_cv(val_ccs):
    """Index of the best hyperparameter per ROI from val_ccs (n_params, n_rois) and the corresponding cc."""
    best = np.nanargmax(val_ccs, axis=0)
    return best, val_ccs[best, np.arange(val_ccs.shape[1])]


# %% [markdown]
# # Methods


# %%
def fit_sta(stim, responses, clip_q=None):
    if clip_q is not None:
        responses = np.minimum(responses, np.percentile(responses[fold_idxs['train']], clip_q, axis=0))
    masks = np.zeros((stim.shape[0], len(folds)), dtype=bool)
    for i, fold in enumerate(folds):
        masks[fold_idxs[fold], i] = True
    return dict(zip(folds, rf.compute_sta(stim, responses, shifts, masks))), {}


def fit_wsta(stim, responses):
    """Whitened STA, (reg_t, reg_s) chosen by 2-fold CV."""
    stas, _ = fit_sta(stim, responses)
    regs = [(rt, rs) for rt in WSTA_REGS for rs in WSTA_REGS]
    cands = {}
    for fold in folds:
        c_t, c_s = rf.compute_stim_covariances(stim, shifts, np.isin(np.arange(stim.shape[0]), fold_idxs[fold]))
        eig_t, eig_s = np.linalg.eigh(c_t), np.linalg.eigh(c_s.astype(np.float64))
        cands[fold] = np.stack([rf.whiten_sta(stas[fold], rf.regularized_inverse(*eig_t, rt),
                                              rf.regularized_inverse(*eig_s, rs)) for rt, rs in regs])
    val_ccs = np.mean([[colwise_cc(rf.predict_linear(stim, c, shifts, fold_idxs[val_fold[fold]]),
                                   responses[fold_idxs[val_fold[fold]]]) for c in cands[fold]]
                       for fold in ('train1', 'train2')], axis=0)
    best, val_cc = select_by_cv(val_ccs)
    roi_idx = np.arange(responses.shape[1])
    strfs = {fold: cands[fold][best, roi_idx] for fold in folds}
    return strfs, dict(val_cc=val_cc, reg_t=[regs[i][0] for i in best], reg_s=[regs[i][1] for i in best])


def get_bases(stim):
    n_t, ny, nx = shifts.size, stim.shape[1], stim.shape[2]
    return rf.bspline_basis(n_t, DF_T), rf.bspline_basis(ny, DF_S), rf.bspline_basis(nx, DF_S)


def fit_basis_glm(feats, grams, bases, responses, kind):
    """Ridge or lasso on basis features, hyperparameter chosen by 2-fold CV. grams: {fold: (X'X, X'Y)}."""
    weights = {}
    for fold in folds:
        idxs = fold_idxs[fold]
        xtx, xty = grams[fold]
        if kind == 'ridge':
            weights[fold] = rf.ridge_path(xtx, xty, RIDGE_REGS)
        elif kind in ('lasso', 'enet'):
            weights[fold] = rf.elastic_net_path(xtx, xty, idxs.size, LASSO_ALPHAS, l1_ratio=L1_RATIOS[kind])
        else:
            raise NotImplementedError(kind)
    val_ccs = np.mean([[colwise_cc(feats[fold_idxs[val_fold[fold]]] @ w, responses[fold_idxs[val_fold[fold]]])
                        for w in weights[fold]] for fold in ('train1', 'train2')], axis=0)
    best, val_cc = select_by_cv(val_ccs)
    roi_idx = np.arange(responses.shape[1])
    strfs = {fold: rf.basis_to_strf(weights[fold][best, :, roi_idx], *bases) for fold in folds}
    params = RIDGE_REGS if kind == 'ridge' else LASSO_ALPHAS
    info = dict(val_cc=val_cc, reg=params[best])
    if kind in ('lasso', 'enet'):
        info['frac_nonzero'] = np.mean(weights['train'][best, :, roi_idx] != 0, axis=1)
    return strfs, info


# %% [markdown]
# ## ALD / ASD (rfest)
#
# rfest builds the dense prior covariance over all RF coefficients, so the RF is reduced to the chromatic sRF
# by fixing the per-channel tRFs from the ridge fit, and estimated at 50 µm resolution (2 x 14 x 14 = 392 coefficients;
# 28 x 28 takes ~4 s per optimization step). Sufficient statistics are divided by `EB_SAMPLE_FACTOR`
# (effective number of independent samples).

# %%
import jax  # noqa: E402
import jax.scipy.linalg  # noqa: E402
from rfest.EvidenceOpt import EmpiricalBayes  # noqa: E402
from rfest.priors import _dense_prior  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update('jax_debug_nans', False)  # enabled globally by rfest.GLM, slows everything down


class ChromaticEB(EmpiricalBayes):
    """Evidence optimization for sRFs of shape (2, ny, nx): free 2x2 channel covariance x ALD/ASD prior in y and x."""

    def __init__(self, X, y, dims, prior='ald', sample_factor=1.):
        super().__init__(X, y, dims, compute_mle=False, time='channel', space=prior,
                         n_hp_time=3, n_hp_space={'ald': 4, 'asd': 1}[prior])
        self.XtX, self.XtY, self.YtY = self.XtX / sample_factor, self.XtY / sample_factor, self.YtY / sample_factor
        self.n_samples = self.n_samples / sample_factor
        self.negative_log_evidence = jax.jit(self.negative_log_evidence)  # rfest evaluates it un-jitted every step

    def cov1d_time(self, params, ncoeff):
        chol = jnp.array([[params[0], 0.], [params[1], params[2]]])
        return _dense_prior(chol @ chol.T)

    def prior_chol(self, params):
        """Cholesky factor of the prior covariance, the Kronecker product of the per-dimension factors."""
        n_t, n_s = self.n_hp_time, self.n_hp_space
        c_c = self.cov1d_time(params[2:2 + n_t], self.dims[0])[0]
        c_y = self.cov1d_space(params[2 + n_t:2 + n_t + n_s], self.dims[1])[0]
        c_x = self.cov1d_space(params[2 + n_t + n_s:], self.dims[2])[0]
        chol_c, chol_y, chol_x = (jnp.linalg.cholesky(c) for c in (c_c, c_y, c_x))
        return jnp.sqrt(jnp.abs(params[1])) * jnp.kron(chol_c, jnp.kron(chol_y, chol_x))

    def _posterior(self, params):
        """With C_prior = L L': m_post = L M^-1 L' X'y / s^2, M = I + L' X'X L / s^2, log|C_prior C_post^-1| = log|M|."""
        sigma = params[0]
        chol_prior = self.prior_chol(params)
        m = jnp.eye(self.n_features) + chol_prior.T @ self.XtX @ chol_prior / sigma ** 2
        chol_m = jnp.linalg.cholesky(m)
        xty = self.XtY / sigma ** 2
        m_post = chol_prior @ jax.scipy.linalg.cho_solve((chol_m, True), chol_prior.T @ xty)
        return m_post, 2 * jnp.sum(jnp.log(jnp.diag(chol_m))), xty

    def negative_log_evidence(self, params):
        """As rfest (Park & Pillow 2011, eq. 10), but using Cholesky factors only (JAX's CPU LU/inv/pinv are slow)."""
        sigma = params[0]
        m_post, logdet, xty = self._posterior(params)
        t0 = jnp.log(2 * jnp.pi * sigma ** 2) * self.n_samples
        return 0.5 * (t0 + logdet - m_post @ xty + self.YtY / sigma ** 2)

    def posterior_mean(self, params):
        return self._posterior(params)[0]


def project_stim_time(stim, trfs):
    """Per-channel temporal projection sum_j trf[j, c] stim[t + shifts[j], ..., c], shape (n_frames, n_ch, ny, nx)."""
    n_frames = stim.shape[0]
    out = np.zeros(stim.shape, dtype=np.float32)
    for j, shift in enumerate(shifts):
        if shift <= 0:
            out[-shift:] += trfs[j] * stim[:n_frames + shift]
        else:
            out[:n_frames - shift] += trfs[j] * stim[shift:]
    return np.moveaxis(out, -1, 1)


def fit_eb_srf(stim, response, strf_init, idxs, prior, num_iters=500, verbose=False):
    """Two-stage fit: tRFs from strf_init, then ALD/ASD of the chromatic sRF. Returns the full STRF."""
    trfs = np.stack([rf.split_strf(strf_init[..., c])[1] for c in range(stim.shape[-1])], axis=1)  # (n_t, n_ch)
    z = mc.bin_movie(np.moveaxis(project_stim_time(stim, trfs)[idxs], 1, -1), EB_BIN)  # (n, ny, nx, n_ch)
    n_ch, ny, nx = z.shape[-1], z.shape[1], z.shape[2]
    z = np.moveaxis(z, -1, 1).reshape(idxs.size, -1).astype(np.float64)
    z -= z.mean(axis=0)
    y = response[idxs].astype(np.float64)
    y = (y - y.mean()) / y.std()

    # Initialization from a ridge solution, with rescaled features such that the RF weights are O(1)
    ztz = z.T @ z
    w0 = np.linalg.solve(ztz + np.trace(ztz) / ztz.shape[0] * np.eye(ztz.shape[0]), z.T @ y)
    scale = np.sqrt(np.mean(w0 ** 2))
    z *= scale
    w0 /= scale

    srf0 = np.abs(w0.reshape(n_ch, ny, nx)).sum(axis=0)
    iy, ix = np.unravel_index(np.argmax(srf0), srf0.shape)
    if prior == 'ald':
        p_space = [3., iy, 0.3, 0., 3., ix, 0.3, 0.]  # tau, nu, tau_f, nu_f for y, then x
    else:
        p_space = [2., 2.]  # smoothness length scales in pixels
    p0 = [0.9, 1., 1., 0.5, 0.8] + p_space

    model = ChromaticEB(z, y, [n_ch, ny, nx], prior=prior, sample_factor=EB_SAMPLE_FACTOR)
    model.fit(p0, num_iters=num_iters, step_size=0.1, tolerance=10, verbose=verbose)
    srf = np.asarray(model.posterior_mean(model.optimized_params)).reshape(n_ch, ny, nx) * scale  # unscaled features
    srf = np.kron(srf, np.ones((1, EB_BIN, EB_BIN))) / EB_BIN ** 2  # same prediction on the unbinned stimulus
    return np.einsum('jc,cyx->jyxc', trfs, srf), np.asarray(model.optimized_params)


def fit_eb(stim, responses, strfs_ridge, prior):
    strfs = {fold: np.zeros_like(strfs_ridge[fold]) for fold in folds}
    params = []
    for i in range(responses.shape[1]):
        for fold in folds:
            strfs[fold][i], p = fit_eb_srf(stim, responses[:, i], strfs_ridge[fold][i], fold_idxs[fold], prior)
        params.append(p)
    val_cc = np.mean([colwise_cc(rf.predict_linear(stim, strfs[fold], shifts, fold_idxs[val_fold[fold]]),
                                 responses[fold_idxs[val_fold[fold]]]) for fold in ('train1', 'train2')], axis=0)
    return strfs, dict(val_cc=val_cc, eb_params=params)


# %% [markdown]
# # Run on the example cells

# %%
METHODS = ['sta'] + [f'sta_clip{q}' for q in CLIP_QS] + ['wsta', 'ridge', 'lasso', 'enet', 'ald', 'asd']

all_strfs, rows, timings = {}, [], []
for field, df_ex_field in df_ex.groupby('field', sort=False):
    df_field = df_rois[df_rois.field == field].set_index('roi_id').loc[df_ex_field.roi_id].reset_index()
    responses = get_responses(df_field)
    stim = rf.standardize_stim(mc.load_mc_movie(df_field.mc.iloc[0], MC_ARRAY_DIR, bin_xy=BIN_XY))
    bases = get_bases(stim)
    feats = rf.compute_basis_features(stim, shifts, *bases)
    grams = {fold: rf.centered_gram(feats, responses, fold_idxs[fold]) for fold in folds}

    field_strfs = {}
    for method in tqdm(METHODS, desc=field):
        t0 = time.time()
        if method.startswith('sta'):
            clip_q = int(method.removeprefix('sta_clip')) if 'clip' in method else None
            strfs, info = fit_sta(stim, responses, clip_q)
        elif method == 'wsta':
            strfs, info = fit_wsta(stim, responses)
        elif method in ('ridge', 'lasso', 'enet'):
            strfs, info = fit_basis_glm(feats, grams, bases, responses, method)
        elif method in ('ald', 'asd'):
            strfs, info = fit_eb(stim, responses, field_strfs['ridge'], method)
        timings.append(dict(field=field, method=method, time_s=time.time() - t0, n_rois=responses.shape[1]))
        field_strfs[method] = strfs

        scores = evaluate(stim, responses, strfs)
        for i, roi_id in enumerate(df_field.roi_id):
            all_strfs[(field, roi_id, method)] = strfs['train'][i]
            rows.append(dict(field=field, roi_id=roi_id, method=method, **{k: v[i] for k, v in scores.items()},
                             **{k: v[i] for k, v in info.items()}))
    del feats, grams

df_scores = pd.DataFrame(rows).merge(df_ex, on=['field', 'roi_id'])
pd.DataFrame(timings).groupby('method', sort=False).time_s.sum().round(1)


# %% [markdown]
# # Postprocess


# %%
def describe_strf(strf):
    """Per channel sRF/tRF (SVD), Gaussian fit and tRF metrics; dominant channel by peak amplitude."""
    out = {}
    for i, ch in enumerate(mc.MC_CHANNELS):
        srf, trf, _ = rf.split_strf(strf[..., i])
        srf_fit, gauss = rf.fit_srf_gauss(srf)
        out.update({f'{ch}_srf': srf, f'{ch}_trf': trf, f'{ch}_srf_fit': srf_fit,
                    f'{ch}_amp': np.max(np.abs(strf[past, ..., i])), f'{ch}_gauss_r2': gauss['r2'],
                    f'{ch}_dia_um': rf.gauss_diameter(gauss['sx'], gauss['sy'], PIXEL_SIZE_UM),
                    f'{ch}_polarity': rf.compute_trf_metrics(trf, rf_time)['polarity']})
    dom = mc.MC_CHANNELS[int(out['uv_amp'] > out['green_amp'])]
    out.update(dom_ch=dom, **{f'dom_{k}': out[f'{dom}_{k}'] for k in ('dia_um', 'gauss_r2', 'polarity')},
               uv_green_index=(out['uv_amp'] - out['green_amp']) / (out['uv_amp'] + out['green_amp']))
    return out


df_desc = pd.DataFrame([dict(field=f, roi_id=r, method=m, **describe_strf(s)) for (f, r, m), s in all_strfs.items()])
df_scores = df_scores.merge(df_desc, on=['field', 'roi_id', 'method'])
df_scores['method'] = pd.Categorical(df_scores.method, METHODS, ordered=True)
df_scores.to_pickle(os.path.join(RESULTS_DIR, 'mc_rf_methods_examples.pkl'))

summary_cols = ['test_cc', 'split_cc', 'val_cc', 'dom_dia_um', 'dom_gauss_r2']
df_scores.groupby('method', observed=True)[summary_cols].median().round(3)

# %%
pd.set_option('display.width', 250)
df_scores.pivot_table(index=['field', 'roi_id', 'supergroup'], columns='method', values='test_cc', observed=True).round(3)

# %% [markdown]
# # Figures

# %%
fig, axs = plt.subplots(1, 4, figsize=(13, 3))
x = np.arange(len(METHODS))
for ax, col in zip(axs, ['test_cc', 'split_cc', 'dom_dia_um', 'dom_gauss_r2']):
    df_p = df_scores.pivot_table(index=['field', 'roi_id'], columns='method', values=col, observed=True)[METHODS]
    ax.plot(x, df_p.values.T, c='gray', lw=0.5, alpha=0.6)
    ax.plot(x, df_p.median().values, c='k', lw=2, marker='o', ms=3)
    ax.set_xticks(x, METHODS, rotation=60, ha='right')
    ax.set(title=col)
axs[2].set_ylim(0, None)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'methods_summary.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'methods_summary.png'), dpi=120)
plt.show()


# %%
def plot_strf(row, axs, title=None):
    vabsmax = max(np.max(np.abs(row[f'{ch}_srf'])) for ch in mc.MC_CHANNELS)
    for ax, ch in zip(axs[:2], mc.MC_CHANNELS):
        ax.imshow(row[f'{ch}_srf'], cmap='RdBu_r', vmin=-vabsmax, vmax=vabsmax)
        if np.all(np.isfinite(row[f'{ch}_srf_fit'])):
            ax.contour(row[f'{ch}_srf_fit'], levels=[np.nanmax(row[f'{ch}_srf_fit']) * np.exp(-2)],
                       colors=CH_COLORS[ch], linewidths=0.6)
        ax.set(xticks=[], yticks=[])
    ax = axs[2]
    ax.axhline(0, c='lightgray', lw=0.5)
    ax.axvline(0, c='lightgray', lw=0.5)
    for ch in mc.MC_CHANNELS:
        ax.plot(rf_time, row[f'{ch}_trf'] * row[f'{ch}_amp'], c=CH_COLORS[ch], lw=0.8)
    ax.set(yticks=[], xticks=[-1, 0])
    if title:
        axs[0].set_title(title, fontsize=6, loc='left')


# Overview: one row per cell, one column triplet per method
n_m = len(METHODS)
fig, axs = plt.subplots(len(df_ex), 3 * n_m, figsize=(1.1 * 3 * n_m, 1.0 * len(df_ex)), squeeze=False,
                        gridspec_kw=dict(width_ratios=[1, 1, 1.4] * n_m, wspace=0.08, hspace=0.35))
for i, ex in df_ex.iterrows():
    for j, method in enumerate(METHODS):
        row = df_scores[(df_scores.field == ex.field) & (df_scores.roi_id == ex.roi_id) &
                        (df_scores.method == method)].iloc[0]
        plot_strf(row, axs[i, 3 * j:3 * j + 3], title=f'test {row.test_cc:.2f}, split {row.split_cc:.2f}')
        if i == 0:
            axs[i, 3 * j + 1].text(0.5, 1.6, method, transform=axs[i, 3 * j + 1].transAxes, ha='center',
                                   fontsize=9, fontweight='bold')
    axs[i, 0].set_ylabel(f'{ex.field} {ex.roi_id}\n{ex.supergroup}', fontsize=7)
plt.savefig(os.path.join(FIG_DIR, 'methods_examples.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'methods_examples.png'), dpi=110, bbox_inches='tight')
plt.show()

# %%
# Detail: one page per cell, full STRF slices of the dominant channel per method
lag_idxs = np.searchsorted(rf_time, [-0.5, -0.4, -0.3, -0.2, -0.13, -0.07, 0.])
with PdfPages(os.path.join(FIG_DIR, 'methods_examples_strf_slices.pdf')) as pdf:
    for _, ex in df_ex.iterrows():
        fig, axs = plt.subplots(n_m, 2 * len(lag_idxs), figsize=(12, 1.0 * n_m), squeeze=False)
        for i, method in enumerate(METHODS):
            strf = all_strfs[(ex.field, ex.roi_id, method)]
            vabsmax = np.max(np.abs(strf[past]))
            for c, ch in enumerate(mc.MC_CHANNELS):
                for k, li in enumerate(lag_idxs):
                    ax = axs[i, c * len(lag_idxs) + k]
                    ax.imshow(strf[li, ..., c], cmap='RdBu_r', vmin=-vabsmax, vmax=vabsmax)
                    ax.set(xticks=[], yticks=[])
                    if i == 0:
                        ax.set_title(f'{ch} {rf_time[li]:.2f}s', fontsize=6, color=CH_COLORS[ch])
            axs[i, 0].set_ylabel(method, fontsize=7)
        fig.suptitle(f'{ex.field} roi{ex.roi_id} {ex.supergroup}')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
