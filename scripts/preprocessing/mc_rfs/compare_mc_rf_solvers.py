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
# # Mouse-cam RFs: FISTA vs. Adam
#
# Compares `df_eyewire2_roi_level_mc_rf.parquet` (FISTA) with `df_eyewire2_roi_level_mc_rf_adam.parquet`
# (`fit_mc_rfs.py --solver adam`), each at its own selected penalty:
# - `mc_rf_solvers_cc`: test correlation of the selected penalty, FISTA vs. Adam, per ROI;
# - `mc_rf_solvers_examples`: per ROI sRF green / UV and tRFs, FISTA (left) and Adam (right).
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
MC_RF_DIR = os.path.join(data_loader.DATA_ROOT, 'mc_rfs')
SOLVERS = {'fista': 'df_eyewire2_roi_level_mc_rf.parquet', 'adam': 'df_eyewire2_roi_level_mc_rf_adam.parquet'}
KEY_COLS = ['experimenter', 'date', 'exp_num', 'field', 'roi_id']
CH_COLORS = {'green': 'forestgreen', 'uv': 'darkviolet'}
MIN_SURROUND_VOL = 0.1
EXAMPLES = [('GCL0', 5), ('GCL0', 68), ('GCL0', 10), ('GCL1', 46), ('GCL2', 54), ('GCL3', 26), ('GCL4', 63),
            ('GCL0', 9)]
N_BEST = 8  # additionally plot the ROIs with the best test correlation (max over both solvers)

# %%
df_rois = data_loader.load_df_rois().reset_index(drop=True)
dfs = {s: data_loader.load_parquet_df(os.path.join(MC_RF_DIR, f)) for s, f in SOLVERS.items()}
df = dfs['fista'].merge(dfs['adam'], on=KEY_COLS, suffixes=('@fista', '@adam'), validate='one_to_one').merge(
    df_rois[KEY_COLS + ['qfilt', 'supergroup']], on=KEY_COLS, validate='one_to_one')
print(f'{len(df)} ROIs in both ({", ".join(f"{s}: {len(d)}" for s, d in dfs.items())}), {df.qfilt.sum()} pass qfilt')


def get(row, col, solver):
    return row[f'{col}@{solver}']


# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
ax = axs[0]
for qfilt, c in ((False, 'gray'), (True, 'C3')):
    d = df[df.qfilt == qfilt]
    ax.scatter(d['mc_rf_best_cc@fista'], d['mc_rf_best_cc@adam'], s=4, c=c, alpha=0.6, label=f'qfilt={qfilt}')
lim = [np.nanmin(df[['mc_rf_best_cc@fista', 'mc_rf_best_cc@adam']].values) - 0.02,
       np.nanmax(df[['mc_rf_best_cc@fista', 'mc_rf_best_cc@adam']].values) + 0.02]
ax.plot(lim, lim, c='k', lw=0.5)
ax.set(xlim=lim, ylim=lim, xlabel='test cc FISTA', ylabel='test cc Adam', aspect='equal')
ax.legend(fontsize=7)
ax = axs[1]
diff = (df['mc_rf_best_cc@adam'] - df['mc_rf_best_cc@fista'])[df.qfilt]
ax.hist(diff.dropna(), bins=30, color='C3', alpha=0.6)
ax.axvline(0, c='k', lw=0.5)
ax.set(xlabel='test cc Adam - FISTA (qfilt)', ylabel='# ROIs', title=f'median {diff.median():+.3f}')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_solvers_cc.png'), dpi=120)
plt.show()


# %%
def plot_rf(row, solver, axs):
    srfs = {ch: get(row, f'mc_rf_{ch}_srf', solver) for ch in mc.MC_CHANNELS}
    vabsmax = np.nanmax([np.nanmax(np.abs(s)) for s in srfs.values()]) if np.any(np.isfinite(srfs['green'])) else 1.
    for ax, ch in zip(axs[:2], mc.MC_CHANNELS):
        ax.imshow(srfs[ch], cmap='RdBu_r', vmin=-vabsmax, vmax=vabsmax)
        prefix = f'mc_rf_{ch}_dog_'
        params = {k.removeprefix(prefix).removesuffix(f'@{solver}'): v for k, v in row.items()
                  if k.startswith(prefix) and k.endswith(f'@{solver}')}
        rf.draw_srf_fit(ax, params, srfs[ch].shape, CH_COLORS[ch], min_surround_vol=MIN_SURROUND_VOL)
        ax.set(xticks=[], yticks=[])
    ax = axs[2]
    trf0 = get(row, 'mc_rf_green_trf', solver)
    rf_time = get(row, 'mc_rf_trf_t0', solver) + np.arange(trf0.size) * get(row, 'mc_rf_trf_dt', solver)
    ax.axhline(0, c='lightgray', lw=0.5)
    ax.axvline(0, c='lightgray', lw=0.5)
    for ch in mc.MC_CHANNELS:
        ax.plot(rf_time, get(row, f'mc_rf_{ch}_trf', solver) * np.nanmax(np.abs(srfs[ch])), c=CH_COLORS[ch], lw=0.8)
    ax.set(yticks=[], xticks=[-1, 0])
    axs[0].set_title(f'{solver}: cc {get(row, "mc_rf_best_cc", solver):.2f}, '
                     f'alpha {get(row, "mc_rf_best_alpha", solver):.2g}', fontsize=6, loc='left')


ex_idx = [df.index[(df.field == f) & (df.roi_id == r)][0] for f, r in EXAMPLES if ((df.field == f) & (df.roi_id == r)).any()]
best_cc = df[['mc_rf_best_cc@fista', 'mc_rf_best_cc@adam']].max(axis=1)
best_idx = [i for i in best_cc.sort_values(ascending=False).index if i not in ex_idx][:N_BEST]
rows = df.loc[ex_idx + best_idx]
n_cols = 2  # ROIs per figure row, each FISTA | Adam
n_rows = int(np.ceil(len(rows) / n_cols))
fig = plt.figure(figsize=(6.4 * n_cols, 1.5 * n_rows))
outer = fig.add_gridspec(n_rows, n_cols, hspace=0.7, wspace=0.12)
for k, (_, row) in enumerate(rows.iterrows()):
    inner = outer[k // n_cols, k % n_cols].subgridspec(1, 7, width_ratios=[1, 1, 1.4, 0.25, 1, 1, 1.4], wspace=0.08)
    for j0, solver in ((0, 'fista'), (4, 'adam')):
        plot_rf(row, solver, [fig.add_subplot(inner[0, j0 + j]) for j in range(3)])
    fig.axes[-6].text(0, 1.45, f'{row.field} {row.roi_id} {row.supergroup}', transform=fig.axes[-6].transAxes,
                      fontsize=7, fontweight='bold')
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_solvers_examples.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_solvers_examples.png'), dpi=120, bbox_inches='tight')
plt.show()
