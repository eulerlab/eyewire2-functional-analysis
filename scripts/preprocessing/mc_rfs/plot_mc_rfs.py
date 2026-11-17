# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: ew2-func
#     language: python
#     name: ew2-func
# ---

# %% [markdown]
# # QC of the mouse-cam RFs (`fit_mc_rfs.py`)
#
# Reads `df_eyewire2_roi_level_mc_rf.parquet` and plots
# - `mc_rf_summary`: test correlation, test MSE and fraction of non-zero weights vs. the penalty (thin lines = ROIs,
#   thick = median, split by `qfilt`), and the distribution of the selected penalty and its test correlation;
# - `mc_rf_examples`: sRF green / UV with the 2-sigma ellipses of the DoG center (solid) and surround (dotted), and the
#   tRFs of the selected penalty, for example ROIs.
# Figures go to `figures/` next to this script.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from eyewire2_functional_analysis import data_loader, receptive_fields as rf, style
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

style.set_rc_params()

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
MC_RF_PATH = os.path.join(data_loader.DATA_ROOT, 'mc_rfs', 'df_eyewire2_roi_level_mc_rf.parquet')
CH_COLORS = {'green': 'forestgreen', 'uv': 'darkviolet'}
MIN_SURROUND_VOL = 0.1  # draw the surround ellipse only if its volume relative to the center (w k^2) is at least this
EXAMPLES = [('GCL0', 5), ('GCL0', 68), ('GCL0', 10), ('GCL1', 46), ('GCL2', 54), ('GCL3', 26), ('GCL4', 63),
            ('GCL0', 9)]
N_BEST = 8  # additionally plot the ROIs with the best test correlation

# %%
df_rois = data_loader.load_df_rois().reset_index(drop=True)
df = data_loader.load_parquet_df(MC_RF_PATH).merge(
    df_rois[['experimenter', 'date', 'exp_num', 'field', 'roi_id', 'qfilt', 'supergroup']],
    on=['experimenter', 'date', 'exp_num', 'field', 'roi_id'], validate='one_to_one')
alphas = df.mc_rf_alphas.iloc[0]
print(f'{len(df)} ROIs, {df.qfilt.sum()} pass qfilt, {(df.mc_rf_best_alpha_idx < 0).sum()} without a valid RF')
print(f'Not converged: {np.mean(np.stack(df.mc_rf_converged) == 0):.1%} of the fits')
df.groupby('qfilt').mc_rf_best_cc.describe().round(3)

# %%
fig, axs = plt.subplots(1, 5, figsize=(17, 3))
for ax, col, ylabel in zip(axs[:3], ['mc_rf_cc', 'mc_rf_mse', 'mc_rf_frac_nonzero'],
                           ['test correlation', 'test MSE', 'fraction non-zero weights']):
    for qfilt, c in ((True, 'C3'), (False, 'gray')):
        if not (df.qfilt == qfilt).any():
            continue
        vals = np.stack(df.loc[df.qfilt == qfilt, col])
        ax.plot(alphas, vals.T, c=c, lw=0.3, alpha=0.15)
        ax.plot(alphas, np.nanmedian(vals, axis=0), c=c, lw=2, marker='o', ms=3, label=f'qfilt={qfilt}')
    ax.set(xscale='log', xlabel='alpha / alpha_max', ylabel=ylabel)
    if col == 'mc_rf_mse':
        ax.set_yscale('log')
axs[0].legend(fontsize=7)
for qfilt, c in ((True, 'C3'), (False, 'gray')):
    d = df[df.qfilt == qfilt]
    axs[3].hist(d.mc_rf_best_alpha_idx, bins=np.arange(-1.5, len(alphas)), color=c, alpha=0.6)
    axs[4].hist(d.mc_rf_best_cc.dropna(), bins=np.linspace(-0.2, 1, 25), color=c, alpha=0.6)
axs[3].set(xlabel='selected alpha idx (-1: none)', ylabel='# ROIs')
axs[4].set(xlabel='test correlation (selected alpha)', ylabel='# ROIs')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_summary.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_summary.png'), dpi=120)
plt.show()


# %%
def dog_params(row, ch):
    return {k.removeprefix(f'mc_rf_{ch}_dog_'): v for k, v in row.items() if k.startswith(f'mc_rf_{ch}_dog_')}


def plot_rf(row, axs):
    srfs = {ch: row[f'mc_rf_{ch}_srf'] for ch in mc.MC_CHANNELS}
    vabsmax = np.nanmax([np.nanmax(np.abs(s)) for s in srfs.values()]) if np.any(np.isfinite(srfs['green'])) else 1.
    for ax, ch in zip(axs[:2], mc.MC_CHANNELS):
        ax.imshow(srfs[ch], cmap='RdBu_r', vmin=-vabsmax, vmax=vabsmax)
        rf.draw_srf_fit(ax, dog_params(row, ch), srfs[ch].shape, CH_COLORS[ch], min_surround_vol=MIN_SURROUND_VOL)
        ax.set(xticks=[], yticks=[])
    ax = axs[2]
    rf_time = row.mc_rf_trf_t0 + np.arange(row.mc_rf_green_trf.size) * row.mc_rf_trf_dt
    ax.axhline(0, c='lightgray', lw=0.5)
    ax.axvline(0, c='lightgray', lw=0.5)
    for ch in mc.MC_CHANNELS:  # scaled by the sRF amplitude so that the channels are comparable
        ax.plot(rf_time, row[f'mc_rf_{ch}_trf'] * np.nanmax(np.abs(srfs[ch])), c=CH_COLORS[ch], lw=0.8)
    ax.set(yticks=[], xticks=[-1, 0])


ex_idx = [df.index[(df.field == f) & (df.roi_id == r)][0] for f, r in EXAMPLES if ((df.field == f) & (df.roi_id == r)).any()]
best_idx = [i for i in df.sort_values('mc_rf_best_cc', ascending=False).index if i not in ex_idx][:N_BEST]
rows = df.loc[ex_idx + best_idx]
n_cols = 4
n_rows = int(np.ceil(len(rows) / n_cols))
fig = plt.figure(figsize=(3.2 * n_cols, 1.4 * n_rows))
outer = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.15)
for k, (_, row) in enumerate(rows.iterrows()):
    inner = outer[k // n_cols, k % n_cols].subgridspec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.08)
    axs = [fig.add_subplot(inner[0, j]) for j in range(3)]
    plot_rf(row, axs)
    axs[0].set_title(f'{row.field} {row.roi_id} {row.supergroup}\ncc {row.mc_rf_best_cc:.2f}, '
                     f'alpha {row.mc_rf_best_alpha:.2g}', fontsize=6, loc='left')
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_examples.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_examples.png'), dpi=120, bbox_inches='tight')
plt.show()
