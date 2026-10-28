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
# # wSTA and spline GLM (lasso) RFs of mouse-cam ROIs
#
# Same data, response and train/test split as `mouse_cam_rfs.py` / `mc_rf_methods.py`. Two methods per ROI:
# - `wsta`: whitened STA, ``C_t^-1 STA C_s^-1`` with a separable approximation of the stimulus covariance and
#   ridge-regularized inverses (see `mc_rf_methods.py`).
# - `glm`: linear-Gaussian GLM with a lasso penalty on RFEst cubic-regression splines over (t, y, x), one input
#   channel per color (green, UV) with a shared penalty, fitted with Adam on GPU for all ROIs and penalty strengths of a
#   field at once (`receptive_fields_glm.fit_spline_glm_batch`). The responses are z-scored per ROI so that the penalty
#   grid is comparable across ROIs. The L1 penalty is shrinkage only (Adam), it does not produce exact zeros.
#
# Hyperparameters are chosen per ROI by 2-fold cross-validation (train1 <-> train2, correlation of the prediction), as
# in `mc_rf_methods.py`; `split_cc` (STRFs from train1 vs. train2, past lags) and `test_cc` (linear prediction vs. mean
# test response) are computed identically, so the numbers are comparable with that notebook.
#
# `SCOPE` selects the ROIs:
# - `best_only`: the 12 example cells of `mc_rf_methods.py` (best STAs, opposite-polarity and medium-quality cells).
# - `good_only`: ROIs with a reliable STA, `rf_ok` in `results/mc_rfs.pkl` (from `mouse_cam_rfs.py`).
# - `all`: all ROIs.
#
# Every finished (field, method) is cached in `results/glm_cache/<tag>/` (the tag contains the scope and the GLM
# settings), so an interrupted run resumes at the next field/method. Delete the folder after changing other settings.

# %%
import os

# Limit the CPU cores; must be set before numpy / jax are imported (BLAS, OpenMP and XLA-CPU thread pools).
N_CPUS = 40
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[var] = str(N_CPUS)
if hasattr(os, 'sched_setaffinity'):  # Linux: hard limit, also covers any thread pool missed above (e.g. XLA-CPU)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:N_CPUS])

import pickle
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from eyewire2_functional_analysis import data_loader, paths, receptive_fields as rf, style
from eyewire2_functional_analysis.receptive_fields_glm import fit_spline_glm_batch
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

style.set_rc_params()
jax.config.update('jax_debug_nans', False)
print('JAX devices:', jax.devices())

# %%
SCOPE = 'best_only'  # 'best_only' | 'good_only' | 'all'
assert SCOPE in ('best_only', 'good_only', 'all'), SCOPE

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FIG_DIR = os.path.join(HERE, 'figures', 'glm', SCOPE)
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

# Same example cells as in mc_rf_methods.py
EXAMPLES = [
    ('GCL0', 5), ('GCL0', 68), ('GCL0', 10), ('GCL1', 46), ('GCL2', 54), ('GCL3', 26),
    ('GCL4', 63), ('GCL0', 9), ('GCL0', 21), ('GCL0', 25), ('GCL1', 47), ('GCL1', 33),
]
RF_OK_PATH = os.path.join(RESULTS_DIR, 'mc_rfs.pkl')  # with column rf_ok, from mouse_cam_rfs.py

WSTA_REGS = np.logspace(-2, 3, 6)  # relative to the mean eigenvalue, for both C_t and C_s
GLM_DF = (10, 14, 14)  # spline degrees of freedom along (t, y, x); the RF is (46, 28, 28) per channel
GLM_BETAS = np.logspace(-4, 0, 9)  # lasso penalty strengths (for z-scored responses), chosen per ROI by CV
GLM_N_ITERS = 1000  # Adam iterations, no early stopping
GLM_ROI_CHUNK = 64  # ROIs on the GPU at once
METHODS = ['wsta', 'glm']

CACHE_DIR = os.path.join(RESULTS_DIR, 'glm_cache', f'{SCOPE}_bin{BIN_XY}_df{"-".join(map(str, GLM_DF))}'
                                                   f'_b{GLM_BETAS[0]:g}-{GLM_BETAS[-1]:g}x{GLM_BETAS.size}_it{GLM_N_ITERS}')
os.makedirs(CACHE_DIR, exist_ok=True)

# %% [markdown]
# # Data

# %%
df_rois = data_loader.load_df_rois(DATA_2P).reset_index(drop=True)

if SCOPE == 'best_only':
    df_sel = pd.DataFrame(EXAMPLES, columns=['field', 'roi_id'])
elif SCOPE == 'good_only':
    assert os.path.exists(RF_OK_PATH), f'run mouse_cam_rfs.py first to create {RF_OK_PATH}'
    df_rf = pd.read_pickle(RF_OK_PATH)
    df_sel = df_rf.loc[df_rf.rf_ok, ['field', 'roi_id']]
else:
    df_sel = df_rois[['field', 'roi_id']]
df_sel = df_sel.merge(df_rois[['field', 'roi_id', 'supergroup', 'group_id']], on=['field', 'roi_id'],
                      validate='one_to_one').reset_index(drop=True)
print(f'{SCOPE}: {len(df_sel)} ROIs\n', df_sel.groupby('field').size().to_dict())

blocks = mc.get_frame_blocks()
fold_idxs = {'train1': np.where(blocks == 'train1')[0], 'train2': np.where(blocks == 'train2')[0],
             'train': np.where(np.isin(blocks, TRAIN_BLOCKS))[0]}
folds = ('train1', 'train2', 'train')
val_fold = {'train1': 'train2', 'train2': 'train1'}
test_idxs = np.stack([np.where(blocks == b)[0] for b in TEST_BLOCKS])
shifts, rf_time = rf.get_rf_shifts(DT_FRAME, DUR_PAST_S, DUR_FUTURE_S)
past = rf_time <= 0
n_eval_skip = np.sum(past)
N_PAST, N_FUTURE = int(np.sum(shifts <= 0)), int(np.sum(shifts > 0))  # lag 0 counts as past


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


def cached(name, fn):
    """Result of fn(), loaded from CACHE_DIR/name.pkl if present, else computed and saved (atomically)."""
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            pass  # unreadable, recompute
    result = fn()
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(result, f)
    os.replace(path + '.tmp', path)
    return result


def select_by_cv(val_ccs):
    """Index of the best hyperparameter per ROI from val_ccs (n_params, n_rois) and the corresponding cc."""
    best = np.nanargmax(val_ccs, axis=0)
    return best, val_ccs[best, np.arange(val_ccs.shape[1])]


# %% [markdown]
# # Methods


# %%
def fit_sta(stim, responses):
    masks = np.zeros((stim.shape[0], len(folds)), dtype=bool)
    for i, fold in enumerate(folds):
        masks[fold_idxs[fold], i] = True
    return dict(zip(folds, rf.compute_sta(stim, responses, shifts, masks)))


def fit_wsta(stim, responses):
    """Whitened STA, (reg_t, reg_s) chosen by 2-fold CV."""
    stas = fit_sta(stim, responses)
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


def fit_glm(stim, responses):
    """Lasso spline GLM for all ROIs; beta chosen by 2-fold CV (common beta for the train1, train2 and train fits)."""
    z = (responses - responses[fold_idxs['train']].mean(axis=0)) / (responses[fold_idxs['train']].std(axis=0) + 1e-9)
    kw = dict(n_past=N_PAST, n_future=N_FUTURE, df=GLM_DF, betas=GLM_BETAS, alpha=1., n_iters=GLM_N_ITERS,
              roi_chunk=GLM_ROI_CHUNK, return_all_betas=True)
    fits, val_ccs = {}, []
    for fold in folds:
        # train1 / train2 are validated on the other half; the final fit on all train frames has no dev frames
        dev_idx = fold_idxs[val_fold[fold]] if fold in val_fold else None
        _, fits[fold] = fit_spline_glm_batch(stim, z, train_idx=fold_idxs[fold], dev_idx=dev_idx, **kw)
        if dev_idx is not None:
            val_ccs.append(fits[fold]['dev_ccs'])
    best, val_cc = select_by_cv(np.mean(val_ccs, axis=0))
    roi_idx = np.arange(responses.shape[1])
    strfs = {fold: fits[fold]['strfs_all'][best, roi_idx] for fold in folds}
    return strfs, dict(val_cc=val_cc, beta=GLM_BETAS[best])


# %% [markdown]
# # Run

# %%
all_strfs, rows, timings = {}, [], []
for field, df_sel_field in df_sel.groupby('field', sort=False):
    df_field = df_rois[df_rois.field == field].set_index('roi_id').loc[df_sel_field.roi_id].reset_index()
    responses = get_responses(df_field)
    stim = rf.standardize_stim(mc.load_mc_movie(df_field.mc.iloc[0], MC_ARRAY_DIR, bin_xy=BIN_XY))

    for method in METHODS:
        t0 = time.time()
        fit = {'wsta': fit_wsta, 'glm': fit_glm}[method]
        strfs, info = cached(f'{field}_{method}', lambda: fit(stim, responses))
        timings.append(dict(field=field, method=method, time_s=time.time() - t0, n_rois=responses.shape[1]))

        scores = evaluate(stim, responses, strfs)
        for i, roi_id in enumerate(df_field.roi_id):
            all_strfs[(field, roi_id, method)] = strfs['train'][i]
            rows.append(dict(field=field, roi_id=roi_id, method=method, **{k: v[i] for k, v in scores.items()},
                             **{k: v[i] for k, v in info.items()}))
    del stim

df_scores = pd.DataFrame(rows).merge(df_sel, on=['field', 'roi_id'])
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
df_scores.to_pickle(os.path.join(RESULTS_DIR, f'mc_rf_glm_{SCOPE}.pkl'))
with open(os.path.join(RESULTS_DIR, f'mc_rf_glm_{SCOPE}_strfs.pkl'), 'wb') as f:
    pickle.dump(dict(strfs=all_strfs, rf_time=rf_time, shifts=shifts), f)  # train STRFs {(field, roi_id, method): ...}

summary_cols = ['test_cc', 'split_cc', 'val_cc', 'dom_dia_um', 'dom_gauss_r2']
df_scores.groupby('method', observed=True)[summary_cols].median().round(3)

# %%
# The lasso penalty should not sit at the edge of the grid
df_glm = df_scores[df_scores.method == 'glm']
edge = df_glm.beta.isin([GLM_BETAS[0], GLM_BETAS[-1]])
print(f'GLM beta at grid edge: {edge.mean():.0%} (lowest: {np.mean(df_glm.beta == GLM_BETAS[0]):.0%}, '
      f'highest: {np.mean(df_glm.beta == GLM_BETAS[-1]):.0%})')
df_glm.beta.value_counts().sort_index()

# %% [markdown]
# # Figures

# %%
fig, axs = plt.subplots(1, 4, figsize=(11, 3))
x = np.arange(len(METHODS))
for ax, col in zip(axs, ['test_cc', 'split_cc', 'dom_dia_um', 'dom_gauss_r2']):
    df_p = df_scores.pivot_table(index=['field', 'roi_id'], columns='method', values=col, observed=True)[METHODS]
    ax.plot(x, df_p.values.T, c='gray', lw=0.5, alpha=min(1., 6. / len(df_p) ** 0.5))
    ax.plot(x, df_p.median().values, c='k', lw=2, marker='o', ms=3)
    ax.set_xticks(x, METHODS)
    ax.set(title=col, xlim=(-0.3, len(METHODS) - 0.7))
axs[2].set_ylim(0, None)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'wsta_vs_glm_summary.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'wsta_vs_glm_summary.png'), dpi=120)
plt.show()

# %%
df_w = df_scores.pivot_table(index=['field', 'roi_id'], columns='method', values=['test_cc', 'dom_dia_um'], observed=True)
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
for ax, col in zip(axs, ['test_cc', 'dom_dia_um']):
    ax.scatter(df_w[col]['wsta'], df_w[col]['glm'], s=8, c='k', alpha=0.6)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, c='r', lw=0.5)
    ax.set(xlabel=f'wsta {col}', ylabel=f'glm {col}', aspect='equal')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'wsta_vs_glm_scatter.pdf'))
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


# Overview of the cells with the best GLM test_cc: one row per cell, one column triplet (green sRF, UV sRF, tRF) per method
N_PLOT = 24
df_top = df_glm.sort_values('test_cc', ascending=False).head(N_PLOT)
n_m = len(METHODS)
fig, axs = plt.subplots(len(df_top), 3 * n_m, figsize=(1.1 * 3 * n_m, 1.0 * len(df_top)), squeeze=False,
                        gridspec_kw=dict(width_ratios=[1, 1, 1.4] * n_m, wspace=0.08, hspace=0.35))
for i, (_, top) in enumerate(df_top.iterrows()):
    for j, method in enumerate(METHODS):
        row = df_scores[(df_scores.field == top.field) & (df_scores.roi_id == top.roi_id) & (df_scores.method == method)].iloc[0]
        plot_strf(row, axs[i, 3 * j:3 * j + 3], title=f'{top.field} {top.roi_id} {method}  cc={row.test_cc:.2f}')
plt.savefig(os.path.join(FIG_DIR, 'wsta_vs_glm_strfs.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'wsta_vs_glm_strfs.png'), dpi=120, bbox_inches='tight')
plt.show()
