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
# # Mouse-cam receptive fields on morphology: several cells
#
# Same panels as `plot_rf_on_morph_single.py`, two cells per row: the best cell (highest test correlation of the RF
# model) of the best cell types (up to two per type), restricted to cells with well-fitted sRFs. Per row: the green and UV sRF (zoomed
# to the RF size; skeleton and DoG ellipses are cropped to it), the whole stimulus window, and the tRFs.
#
# **Note:** the orientation of the mouse-cam movie in the retinal frame (`receptive_fields.movie_to_retina`) is an
# assumption derived from the stimulus setup and has not been verified with a cell of known RF position yet.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from eyewire2_functional_analysis import data_loader, plot_rf, registration, space_mapping, style
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

style.set_rc_params()

# %%
HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
KEY_COLS = ['experimenter', 'date', 'exp_num', 'field', 'roi_id']

N_CELLS, N_PER_ROW = 24, 2  # number of cells and cells per row
MAX_PER_TYPE = 2  # at most this many cells per cell type (the best ones)
R2_MIN = 0.7  # minimum variance of the sRF explained by the DoG fit, in both channels
MAX_RF_RADIUS_UM = 150.  # excludes poorly constrained (huge) DoG fits and caps the zoom

# %% [markdown]
# ## Load data
#
# Load the ROIs with their matched EM cells, join the RFs, and keep cells with a skeleton, good response quality
# (`qfilt`) and well-fitted sRFs in both channels whose centers lie inside the stimulus and are not larger than
# `MAX_RF_RADIUS_UM`. Then take the best cells per type (at most `MAX_PER_TYPE`).

# %%
df_rois = data_loader.load_df_rois()
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = df.merge(data_loader.load_df_mc_rfs(), on=KEY_COLS, validate='one_to_one')
df = data_loader.add_skels(df=df, inplace=True)

n_px = df.mc_rf_green_srf.iloc[0].shape[0]
r2 = np.minimum(df.mc_rf_green_dog_r2, df.mc_rf_uv_dog_r2)
center_inside = np.all([df[f'mc_rf_{ch}_dog_{xy}0'].between(2, n_px - 3) for ch in mc.MC_CHANNELS for xy in 'xy'],
                       axis=0)
df = df[df.skel.notnull() & df.qfilt & (df.mc_rf_best_alpha_idx >= 0) & (r2 >= R2_MIN) & center_inside]
df = df[[plot_rf.rf_radius(row, margin=0.) <= MAX_RF_RADIUS_UM for _, row in df.iterrows()]]
df = df.sort_values('mc_rf_best_cc', ascending=False)
df = df[df['Cell Type'].notnull() & (df['Cell Type'] != 'Unknown')]
df = df.groupby('Cell Type', sort=False).head(MAX_PER_TYPE)
df_plot = df.head(N_CELLS)
print(f"{len(df)} cells of {df['Cell Type'].nunique()} types pass, plotting {len(df_plot)}")
df_plot[['field', 'roi_id', 'Cell Type', 'mc_rf_best_cc', 'mc_rf_green_dog_r2', 'mc_rf_uv_dog_r2']]

# %% [markdown]
# ## Plot
#
# Columns as in the single-cell plot: green and UV sRF (red = positive, blue = negative, same color scale for both
# channels of a cell) with the skeleton and the 2-sigma DoG center (solid) and surround (dotted); the stimulus window
# (gray frame); and the tRFs scaled by the sRF peak. sRF and tRF are flipped such that the fitted center is positive,
# i.e. the tRF carries the polarity (positive just before the response = ON center).

# %%
reg = registration.load_registration(REG_FILE)

n_rows = int(np.ceil(len(df_plot) / N_PER_ROW))
w_cell = (1, 1, 1, 1.1)
ratios = []
for k in range(N_PER_ROW):
    ratios += ([0.35] if k else []) + list(w_cell)  # spacer between the cells of a row
fig = plt.figure(figsize=(8.2 * N_PER_ROW, 2.75 * n_rows))
gs = fig.add_gridspec(n_rows, len(ratios), width_ratios=ratios, hspace=0.55, wspace=0.1, left=0.01, right=0.99,
                      top=0.975, bottom=0.025)
for i, (_, row) in enumerate(df_plot.iterrows()):
    r_, k = divmod(i, N_PER_ROW)
    c0 = k * (len(w_cell) + 1)
    axs_row = [fig.add_subplot(gs[r_, c0 + j]) for j in range(len(w_cell))]
    skel = space_mapping.align_and_place_skel(row.skel, reg, field=row.field, target_xy=(0., 0.))
    rad = min(plot_rf.rf_radius(row, margin=10.), MAX_RF_RADIUS_UM + 10.)  # RF size
    lim = (-rad, rad, -rad, rad)
    vabsmax = max(np.nanmax(np.abs(row[f'mc_rf_{ch}_srf'])) for ch in mc.MC_CHANNELS)

    for ax, ch in zip(axs_row[:2], mc.MC_CHANNELS):
        plot_rf.plot_srf_on_morph(ax, row, skel, ch, vabsmax=vabsmax)
        ax.set(xlim=lim[:2], ylim=lim[2:])
        ax.set_title(f'{plot_rf.CH_LABELS[ch]} sRF (r² = {row[f"mc_rf_{ch}_dog_r2"]:.2f})',
                     color=plot_rf.CH_COLORS[ch], fontsize=8)
        plot_rf.add_scale_bar(ax, lim, size=100, fontsize=6)

    ax = axs_row[2]
    _, ext = plot_rf.srf_retina(row, 'green')
    plot_rf.plot_srf_on_morph(ax, row, skel, 'green', img_alpha=0., fit_chs=mc.MC_CHANNELS, skel_lw=0.4)
    pad = 0.03 * (ext[1] - ext[0])
    lim_stim = (ext[0] - pad, ext[1] + pad, ext[2] - pad, ext[3] + pad)
    ax.set(xlim=lim_stim[:2], ylim=lim_stim[2:], title='stimulus window')
    ax.title.set_fontsize(8)
    ax.plot(0, 0, 'k+', ms=6, mew=0.8)  # soma
    plot_rf.add_scale_bar(ax, lim_stim, size=200, fontsize=6)

    last_row = r_ == n_rows - 1 or i + N_PER_ROW >= len(df_plot)
    plot_rf.plot_trfs(axs_row[3], row, lw=1., labels=last_row)
    axs_row[3].set_title('tRF', fontsize=8)
    axs_row[3].tick_params(labelsize=6)

    label = f"{row['Cell Type']} ({row.field} ROI {row.roi_id}), test cc {row.mc_rf_best_cc:.2f}"
    axs_row[0].text(0., 1.2, label, transform=axs_row[0].transAxes, ha='left', va='bottom', fontsize=8,
                    fontweight='bold')

fig.legend(handles=[Line2D([], [], c=plot_rf.CH_COLORS[ch], label=plot_rf.CH_LABELS[ch]) for ch in mc.MC_CHANNELS]
                   + [Line2D([], [], c='0.3', ls='-', label='DoG center (2σ)'),
                      Line2D([], [], c='0.3', ls=':', label='DoG surround (2σ)')],
           loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
fig.savefig(os.path.join(FIG_DIR, 'plot_rf_on_morph_grid.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'plot_rf_on_morph_grid.png'), dpi=150, bbox_inches='tight')
plt.show()
