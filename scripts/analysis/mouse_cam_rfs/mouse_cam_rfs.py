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
# # Colored STAs from mouse-cam movie responses
#
# First-pass receptive field (RF) estimation from the mouse-cam (MC) natural movie responses:
# - Movie: green and UV channel, binned 2x2 to 28x28 pixels (25 µm/px, 700 µm in total), 30 Hz.
# - Response: positive part of the derivative of `mc_pp_trace`, resampled at the movie frame times.
# - STA: response-weighted average of the preceding movie frames, minus the average movie of the same frames.
#   Estimated from the train clips only; the three test-clip repeats are kept for evaluation.
#
# Quality measures:
# - `split_cc`: correlation of the STAs from train1 and train2 (past lags only).
# - `test_cc`: correlation of the linear STA prediction with the mean test response.
# - Control: same, but using a wrong MC sequence (same clips in a different order).
#
# Note: natural movies are strongly correlated in space, time and across channels (green vs. UV),
# so STAs are blurred versions of the actual filters. Regularized estimates (e.g. ridge, ASD with rfest) are a next step.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from eyewire2_functional_analysis import data_loader, paths, receptive_fields as rf, style
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

style.set_rc_params()

# %%
HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FIG_DIR = os.path.join(HERE, 'figures')
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data from the eyewire2-data Hugging Face dataset; falls back to the repo's default data folder
DATA_2P = os.path.join(paths.PROJECT_ROOT, '..', 'huggingface', 'eyewire2-data', 'data-2p')
if not os.path.exists(DATA_2P):
    DATA_2P = data_loader.DATA_2P
MC_ARRAY_DIR = os.path.join(DATA_2P, 'stimuli', 'mouse_cam_movies', 'mc_arrays')  # created with mc_to_numpy.py

BIN_XY = 2
PIXEL_SIZE_UM = mc.MC_PIXEL_SIZE_UM * BIN_XY
DT_FRAME = 1. / mc.MC_FRAME_RATE_HZ
DUR_PAST_S, DUR_FUTURE_S = 1.2, 0.3
RESPONSE_KIND = 'gradient'
TRAIN_BLOCKS = ('train1', 'train2')
TEST_BLOCKS = ('test1', 'test2', 'test3')

CH_COLORS = {'green': 'forestgreen', 'uv': 'darkviolet'}

# %% [markdown]
# # Load data

# %%
df_rois = data_loader.load_df_rois(DATA_2P).reset_index(drop=True)
df_rois.groupby('field').agg(n_rois=('roi_id', 'size'), mc=('mc', 'first'), n_qfilt=('qfilt', 'sum'))

# %% [markdown]
# ## Stimulus overview

# %%
mov = mc.load_mc_movie('mc16', MC_ARRAY_DIR, bin_xy=BIN_XY)

frames = [0, 150 * 10, 150 * 30, 150 * 80]
fig, axs = plt.subplots(2, len(frames) + 1, figsize=(8, 3.5))
for i, ch in enumerate(mc.MC_CHANNELS):
    for j, frame in enumerate(frames):
        axs[i, j].imshow(mov[frame, :, :, i], cmap='gray', vmin=0, vmax=1)
        axs[i, j].set_title(f'{ch}, frame {frame}')
    axs[i, -1].imshow(mov[..., i].mean(axis=0), cmap='gray', vmin=0, vmax=1)
    axs[i, -1].set_title(f'{ch}, mean')
for ax in axs.flat:
    ax.set(xticks=[], yticks=[])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stimulus_overview.pdf'))
plt.show()

print('green-UV pixel correlation:', np.corrcoef(mov[::10, ..., 0].ravel(), mov[::10, ..., 1].ravel())[0, 1].round(2))
del mov

# %% [markdown]
# # Compute STAs

# %%
blocks = mc.get_frame_blocks()
frame_masks = np.stack([blocks == 'train1', blocks == 'train2', np.isin(blocks, TRAIN_BLOCKS)], axis=1)
test_idxs = np.stack([np.where(blocks == b)[0] for b in TEST_BLOCKS])
shifts, rf_time = rf.get_rf_shifts(DT_FRAME, DUR_PAST_S, DUR_FUTURE_S)
past = rf_time <= 0
n_eval_skip = np.sum(past)  # skip the first test frames, where the stimulus history differs between repeats


def get_field_responses(df_field):
    """Responses of all ROIs of a field at the movie frame times, shape (n_frames, n_rois)."""
    triggertimes = np.stack(df_field.mc_triggertimes.values)
    assert np.allclose(triggertimes, triggertimes[0])
    frame_times = mc.get_frame_times(triggertimes[0])

    responses, test_qidxs = [], []
    for _, row in df_field.iterrows():
        tracetime = row.mc_pp_trace_t0 + np.arange(row.mc_pp_trace.size) * row.mc_pp_trace_dt
        responses.append(rf.compute_response(row.mc_pp_trace, tracetime, frame_times, kind=RESPONSE_KIND))
        trace_at_frames = np.interp(frame_times, tracetime, row.mc_pp_trace)
        test_qidxs.append(rf.compute_repeat_qidx(trace_at_frames[test_idxs]))
    return np.stack(responses, axis=1), np.array(test_qidxs)


def compute_field_stas(responses, mc_name):
    """Train STAs and quality measures for all ROIs of a field using the movie `mc_name`."""
    stim = rf.standardize_stim(mc.load_mc_movie(mc_name, MC_ARRAY_DIR, bin_xy=BIN_XY))
    sta_train1, sta_train2, sta_train = rf.compute_sta(stim, responses, shifts, frame_masks)

    split_cc = np.array([np.corrcoef(a[past].ravel(), b[past].ravel())[0, 1] for a, b in zip(sta_train1, sta_train2)])

    pred = np.mean([rf.predict_linear(stim, sta_train, shifts, idxs) for idxs in test_idxs], axis=0)
    resp_test = responses[test_idxs].mean(axis=0)
    test_cc = np.array([np.corrcoef(pred[n_eval_skip:, i], resp_test[n_eval_skip:, i])[0, 1]
                        for i in range(responses.shape[1])])
    return sta_train, split_cc, test_cc


def get_control_mc(mc_name):
    """A different MC sequence to use as control (next sequence column)."""
    return f"mc{(int(mc_name.removeprefix('mc')) + 1) % 20:02d}"


# %%
results = []
for field, df_field in tqdm(df_rois.groupby('field')):
    mc_name = df_field.mc.iloc[0]
    responses, test_qidx = get_field_responses(df_field)
    sta, split_cc, test_cc = compute_field_stas(responses, mc_name)
    _, split_cc_ctrl, test_cc_ctrl = compute_field_stas(responses, get_control_mc(mc_name))

    results.append(pd.DataFrame(dict(
        field=field, roi_id=df_field.roi_id.values, sta=list(sta), test_qidx=test_qidx,
        split_cc=split_cc, test_cc=test_cc, split_cc_ctrl=split_cc_ctrl, test_cc_ctrl=test_cc_ctrl)))

df_rf = pd.concat(results, ignore_index=True)

# %% [markdown]
# # Postprocess STAs
#
# Per channel: SVD split into sRF and tRF, 2D Gaussian fit of the sRF, tRF polarity / peak lag / transience.


# %%
def postprocess_sta(sta):
    out = dict()
    for i, ch in enumerate(mc.MC_CHANNELS):
        strf = sta[..., i]
        srf, trf, ev = rf.split_strf(strf)
        srf_fit, gauss = rf.fit_srf_gauss(srf)
        ny, nx = srf.shape
        out.update({
            f'{ch}_srf': srf, f'{ch}_trf': trf, f'{ch}_srf_fit': srf_fit, f'{ch}_ev': ev,
            f'{ch}_amp': np.max(np.abs(strf[past])),
            f'{ch}_gauss_r2': gauss['r2'],
            f'{ch}_x_um': (gauss['x0'] - (nx - 1) / 2) * PIXEL_SIZE_UM,
            f'{ch}_y_um': (gauss['y0'] - (ny - 1) / 2) * PIXEL_SIZE_UM,
            f'{ch}_dia_um': rf.gauss_diameter(gauss['sx'], gauss['sy'], PIXEL_SIZE_UM),
            **{f'{ch}_{k}': v for k, v in rf.compute_trf_metrics(trf, rf_time).items()},
        })
    out['dom_ch'] = mc.MC_CHANNELS[int(out['uv_amp'] > out['green_amp'])]
    out['uv_green_index'] = (out['uv_amp'] - out['green_amp']) / (out['uv_amp'] + out['green_amp'])
    return out


df_rf = pd.concat([df_rf, pd.DataFrame([postprocess_sta(sta) for sta in tqdm(df_rf.sta)])], axis=1)
for col in ['polarity', 'peak_lag_s', 'transience', 'x_um', 'y_um', 'dia_um', 'gauss_r2']:
    df_rf[f'dom_{col}'] = [row[f'{row.dom_ch}_{col}'] for _, row in df_rf.iterrows()]

# %% [markdown]
# ## RF quality
#
# An STA counts as reliable if its split-half correlation exceeds the 95th percentile of the control distribution
# (wrong sequence) and the Gaussian fit explains a reasonable fraction of the sRF.

# %%
SPLIT_CC_THRESH = np.quantile(df_rf.split_cc_ctrl, 0.95)
df_rf['rf_ok'] = (df_rf.split_cc > SPLIT_CC_THRESH) & (df_rf.dom_gauss_r2 > 0.3)

df = df_rois.merge(df_rf, on=['field', 'roi_id'], validate='one_to_one')
df['roi_dx_um'] = df.temporal_nasal_pos_um - df.field_temporal_nasal_pos_um
df['roi_dy_um'] = df.ventral_dorsal_pos_um - df.field_ventral_dorsal_pos_um

print(f'split_cc threshold: {SPLIT_CC_THRESH:.3f}')
print(f"rf_ok: {df.rf_ok.sum()} / {len(df)} (qfilt ROIs: {df[df.qfilt].rf_ok.sum()} / {df.qfilt.sum()})")
df.groupby('field')[['rf_ok', 'qfilt']].mean().round(2)

# %%
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

for ax, col in zip(axs[:2], ['split_cc', 'test_cc']):
    bins = np.linspace(-0.5, 1, 31) if col == 'split_cc' else np.linspace(-0.3, 0.6, 31)
    ax.hist(df[f'{col}_ctrl'], bins=bins, color='gray', alpha=0.6, label='control (wrong sequence)')
    ax.hist(df[col], bins=bins, histtype='step', color='k', lw=1.5, label='true sequence')
    ax.set(xlabel=col, ylabel='# ROIs')
axs[0].axvline(SPLIT_CC_THRESH, c='r', ls='--', lw=1, label='threshold')
axs[0].legend(frameon=False, loc='upper left')

ax = axs[2]
ax.scatter(df.test_qidx, df.split_cc, c=np.where(df.rf_ok, 'k', 'lightgray'), s=8)
ax.set(xlabel='test repeat QI (mc_pp_trace)', ylabel='split_cc')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_quality.pdf'))
plt.show()

# %% [markdown]
# ## Example RFs


# %%
def plot_rf(row, axs):
    """sRF green, sRF UV (same color scale), and both tRFs."""
    vabsmax = max(np.max(np.abs(row[f'{ch}_srf'])) for ch in mc.MC_CHANNELS)
    for ax, ch in zip(axs[:2], mc.MC_CHANNELS):
        ax.imshow(row[f'{ch}_srf'], cmap='RdBu_r', vmin=-vabsmax, vmax=vabsmax)
        ax.contour(row[f'{ch}_srf_fit'], levels=[np.nanmax(row[f'{ch}_srf_fit']) * np.exp(-2)],
                   colors=CH_COLORS[ch], linewidths=0.8)
        ax.set(xticks=[], yticks=[])
        ax.set_title(ch, color=CH_COLORS[ch], fontsize=7)

    ax = axs[2]
    ax.axhline(0, c='lightgray', lw=0.5)
    ax.axvline(0, c='lightgray', lw=0.5)
    for ch in mc.MC_CHANNELS:
        ax.plot(rf_time, row[f'{ch}_trf'] * row[f'{ch}_amp'], c=CH_COLORS[ch], label=ch)
    ax.set_title(f"{row.field} roi{row.roi_id} {row.supergroup}\nsplit cc={row.split_cc:.2f}", fontsize=7)
    ax.set(yticks=[])


df_examples = df[df.rf_ok].sort_values('split_cc', ascending=False).head(24)
n_cols = 4
fig, axs = plt.subplots(int(np.ceil(len(df_examples) / n_cols)), 3 * n_cols, figsize=(14, 1.3 * len(df_examples) / n_cols),
                        squeeze=False, gridspec_kw=dict(width_ratios=[1, 1, 1.6] * n_cols))
for i, (_, row) in enumerate(df_examples.iterrows()):
    plot_rf(row, axs[i // n_cols, 3 * (i % n_cols):3 * (i % n_cols) + 3])
for ax in axs.flat[3 * len(df_examples):]:
    ax.axis('off')
axs[0, 2].legend(frameon=False, fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_examples.pdf'))
plt.show()

# %% [markdown]
# ## tRFs per functional supergroup
#
# Sanity check: the tRF polarity should match the chirp-based supergroup (ON vs. OFF).

# %%
supergroups = [sg for sg in ['OFF', 'ON-OFF', 'Fast ON', 'Slow ON', 'Unc. ON', 'Unc. SbC', 'dAC'] if sg in df.supergroup.unique()]

fig, axs = plt.subplots(2, len(supergroups), figsize=(12, 4), sharex=True, sharey='row')
for j, sg in enumerate(supergroups):
    df_sg = df[(df.supergroup == sg) & df.rf_ok]
    for i, ch in enumerate(mc.MC_CHANNELS):
        ax = axs[i, j]
        ax.axhline(0, c='lightgray', lw=0.5)
        ax.axvline(0, c='lightgray', lw=0.5)
        if len(df_sg) > 0:
            trfs = np.stack(df_sg[f'{ch}_trf'].values)
            ax.plot(rf_time, trfs.T, c=CH_COLORS[ch], lw=0.3, alpha=0.4)
            ax.plot(rf_time, trfs.mean(axis=0), c='k', lw=1.5)
        ax.set_title(f'{sg} (n={len(df_sg)})' if i == 0 else None)
    axs[1, j].set_xlabel('time rel. to response (s)')
for i, ch in enumerate(mc.MC_CHANNELS):
    axs[i, 0].set_ylabel(f'{ch} tRF')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'trfs_per_supergroup.pdf'))
plt.show()

pd.crosstab(df[df.rf_ok].supergroup, df[df.rf_ok].dom_polarity)

# %% [markdown]
# ## RF positions vs. ROI positions
#
# The orientation of the movie relative to the retina is not documented in detail
# (the movies were rotated in the setup, and montage/flip conventions differ between tools).
# Find the rotation/flip that best maps the RF centers onto the ROI positions (both relative to their field mean).

# %%
TRANSFORMS = {
    'identity': np.array([[1, 0], [0, 1]]), 'rot90': np.array([[0, -1], [1, 0]]),
    'rot180': np.array([[-1, 0], [0, -1]]), 'rot270': np.array([[0, 1], [-1, 0]]),
    'flip_x': np.array([[-1, 0], [0, 1]]), 'flip_y': np.array([[1, 0], [0, -1]]),
    'transpose': np.array([[0, 1], [1, 0]]), 'anti_transpose': np.array([[0, -1], [-1, 0]]),
}

df_pos = df[df.rf_ok].copy()
rf_xy = df_pos[['dom_x_um', 'dom_y_um']].values - df_pos.groupby('field')[['dom_x_um', 'dom_y_um']].transform('mean').values
roi_xy = df_pos[['roi_dx_um', 'roi_dy_um']].values - df_pos.groupby('field')[['roi_dx_um', 'roi_dy_um']].transform('mean').values

transform_scores = pd.DataFrame([dict(
    transform_name=name,
    rmse_um=np.sqrt(np.mean(np.sum((rf_xy @ T.T - roi_xy) ** 2, axis=1))),
    cc_x=np.corrcoef((rf_xy @ T.T)[:, 0], roi_xy[:, 0])[0, 1],
    cc_y=np.corrcoef((rf_xy @ T.T)[:, 1], roi_xy[:, 1])[0, 1],
) for name, T in TRANSFORMS.items()]).sort_values('rmse_um')
best_transform = transform_scores.transform_name.iloc[0]
print('RMSE without alignment (RF positions shuffled):',
      np.sqrt(np.mean(np.sum((np.random.default_rng(0).permutation(rf_xy) - roi_xy) ** 2, axis=1))).round(1))
transform_scores.round(3)

# %%
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))

ax = axs[0]
for field, df_f in df_pos.groupby('field'):
    ax.scatter(df_f.dom_x_um, df_f.dom_y_um, s=8, label=field)
half_um = mc.MC_PIXEL_SIZE_UM * 56 / 2
ax.set(xlim=(-half_um, half_um), ylim=(half_um, -half_um), aspect='equal',
       xlabel='x (µm)', ylabel='y (µm)', title='RF centers (stimulus coordinates)')
ax.legend(frameon=False, fontsize=6)

rf_xy_t = rf_xy @ TRANSFORMS[best_transform].T
for ax, k, label in zip(axs[1:], [0, 1], ['temporal-nasal', 'ventral-dorsal']):
    ax.scatter(roi_xy[:, k], rf_xy_t[:, k], s=8, c='k')
    lim = np.max(np.abs(np.concatenate([roi_xy[:, k], rf_xy_t[:, k]])))
    ax.plot([-lim, lim], [-lim, lim], c='gray', ls='--', lw=0.8)
    ax.set(aspect='equal', xlabel=f'ROI {label} (µm, rel. field mean)', ylabel=f'RF ({best_transform}, µm)',
           title=f'cc={np.corrcoef(roi_xy[:, k], rf_xy_t[:, k])[0, 1]:.2f}')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_positions.pdf'))
plt.show()

# %% [markdown]
# ## RF size and chromatic preference

# %%
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
df_ok = df[df.rf_ok]

ax = axs[0]
ax.hist(df_ok.dom_dia_um, bins=np.linspace(0, 500, 26), color='gray')
ax.set(xlabel='RF diameter (µm, 2 sd)', ylabel='# ROIs')

ax = axs[1]
for field, df_f in df_ok.groupby('field'):
    ax.scatter(df_f.field_ventral_dorsal_pos_um, df_f.uv_green_index, s=8, label=field)
ax.axhline(0, c='lightgray', lw=0.5)
ax.set(xlabel='field ventral-dorsal pos. (µm)', ylabel='UV-green index')
ax.legend(frameon=False, fontsize=6)

ax = axs[2]
ax.hist(df_ok.dom_peak_lag_s, bins=np.arange(0, DUR_PAST_S + DT_FRAME, 2 * DT_FRAME), color='gray')
ax.set(xlabel='tRF peak lag (s)', ylabel='# ROIs')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_properties.pdf'))
plt.show()

# %% [markdown]
# # Save results

# %%
df_rf.to_pickle(os.path.join(RESULTS_DIR, 'mc_rfs.pkl'))
np.save(os.path.join(RESULTS_DIR, 'mc_rf_time.npy'), rf_time)
