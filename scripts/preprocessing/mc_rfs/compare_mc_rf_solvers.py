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
# # Mouse-cam RFs: FISTA vs. Adam vs. group lasso vs. elastic net
#
# Compares `df_eyewire2_roi_level_mc_rf.parquet` (lasso, FISTA) with the other solvers of `fit_mc_rfs.py`
# (`--solver adam`: lasso, Adam; `--solver group`: group lasso over the channels, FISTA; `--solver enet`: elastic net,
# FISTA, all `--l1-ratio` runs found), each at its own selected penalty. Solvers whose
# parquet does not exist yet are skipped.
# - `mc_rf_solvers_cc`: test correlation of the selected penalty, each solver vs. FISTA, per ROI;
# - `mc_rf_solvers_examples`: per ROI sRF green / UV and tRFs, one block per solver (left to right).
# Figures go to `figures/` next to this script.

# %%
import glob
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
REFERENCE = 'fista'
SOLVERS = {'fista': 'df_eyewire2_roi_level_mc_rf.parquet', 'adam': 'df_eyewire2_roi_level_mc_rf_adam.parquet',
           'group': 'df_eyewire2_roi_level_mc_rf_group.parquet'}
# all elastic-net runs (one parquet per --l1-ratio), largest l1_ratio first
_enet = [os.path.basename(f) for f in glob.glob(os.path.join(MC_RF_DIR, 'df_eyewire2_roi_level_mc_rf_enet*.parquet'))]
_enet = {f.removeprefix('df_eyewire2_roi_level_mc_rf_').removesuffix('.parquet'): f for f in _enet}
SOLVERS.update(sorted(_enet.items(), key=lambda kv: -float(kv[0].removeprefix('enet'))))
KEY_COLS = ['experimenter', 'date', 'exp_num', 'field', 'roi_id']
CH_COLORS = {'green': 'forestgreen', 'uv': 'darkviolet'}
MIN_SURROUND_VOL = 0.1
EXAMPLES = [('GCL0', 5), ('GCL0', 68), ('GCL0', 10), ('GCL1', 46), ('GCL2', 54), ('GCL3', 26), ('GCL4', 63),
            ('GCL0', 9)]
N_BEST = 8  # additionally plot the ROIs with the best test correlation (max over all solvers)

# %%
missing = [s for s, f in SOLVERS.items() if not os.path.exists(os.path.join(MC_RF_DIR, f))]
if missing:
    print(f'Skipping {missing} (no parquet)')
solvers = [s for s in SOLVERS if s not in missing]
assert REFERENCE in solvers and len(solvers) > 1, solvers

df_rois = data_loader.load_df_rois().reset_index(drop=True)
df = df_rois[KEY_COLS + ['qfilt', 'supergroup']]
n_rows = {}
for s in solvers:
    d = data_loader.load_parquet_df(os.path.join(MC_RF_DIR, SOLVERS[s]))
    n_rows[s] = len(d)
    d = d.rename(columns={c: f'{c}@{s}' for c in d.columns if c not in KEY_COLS})
    df = df.merge(d, on=KEY_COLS, validate='one_to_one')
print(f'{len(df)} ROIs in all ({", ".join(f"{s}: {n}" for s, n in n_rows.items())}), {df.qfilt.sum()} pass qfilt')


def get(row, col, solver):
    return row[f'{col}@{solver}']


# %%
others = [s for s in solvers if s != REFERENCE]
fig, axs = plt.subplots(len(others), 2, figsize=(7, 3.2 * len(others)), squeeze=False)
cc_cols = [f'mc_rf_best_cc@{s}' for s in solvers]
lim = [np.nanmin(df[cc_cols].values) - 0.02, np.nanmax(df[cc_cols].values) + 0.02]
for (ax, ax_hist), s in zip(axs, others):
    for qfilt, c in ((False, 'gray'), (True, 'C3')):
        d = df[df.qfilt == qfilt]
        ax.scatter(d[f'mc_rf_best_cc@{REFERENCE}'], d[f'mc_rf_best_cc@{s}'], s=4, c=c, alpha=0.6, label=f'qfilt={qfilt}')
    ax.plot(lim, lim, c='k', lw=0.5)
    ax.set(xlim=lim, ylim=lim, xlabel=f'test cc {REFERENCE}', ylabel=f'test cc {s}', aspect='equal')
    ax.legend(fontsize=7)
    diff = (df[f'mc_rf_best_cc@{s}'] - df[f'mc_rf_best_cc@{REFERENCE}'])[df.qfilt]
    ax_hist.hist(diff.dropna(), bins=30, color='C3', alpha=0.6)
    ax_hist.axvline(0, c='k', lw=0.5)
    ax_hist.set(xlabel=f'test cc {s} - {REFERENCE} (qfilt)', ylabel='# ROIs', title=f'median {diff.median():+.3f}')
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
best_cc = df[cc_cols].max(axis=1)
best_idx = [i for i in best_cc.sort_values(ascending=False).index if i not in ex_idx][:N_BEST]
rows = df.loc[ex_idx + best_idx]
n_solvers = len(solvers)
n_cols = 2 if n_solvers <= 2 else 1  # ROIs per figure row, each with one block (sRF green, sRF UV, tRFs) per solver
n_fig_rows = int(np.ceil(len(rows) / n_cols))
fig = plt.figure(figsize=(3.2 * n_solvers * n_cols, 1.5 * n_fig_rows))
outer = fig.add_gridspec(n_fig_rows, n_cols, hspace=0.7, wspace=0.12)
width_ratios = ([1, 1, 1.4, 0.25] * n_solvers)[:-1]
for k, (_, row) in enumerate(rows.iterrows()):
    inner = outer[k // n_cols, k % n_cols].subgridspec(1, len(width_ratios), width_ratios=width_ratios, wspace=0.08)
    for i, solver in enumerate(solvers):
        plot_rf(row, solver, [fig.add_subplot(inner[0, 4 * i + j]) for j in range(3)])
    fig.axes[-3 * n_solvers].text(0, 1.45, f'{row.field} {row.roi_id} {row.supergroup}',
                                  transform=fig.axes[-3 * n_solvers].transAxes, fontsize=7, fontweight='bold')
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_solvers_examples.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'mc_rf_solvers_examples.png'), dpi=120, bbox_inches='tight')
plt.show()
