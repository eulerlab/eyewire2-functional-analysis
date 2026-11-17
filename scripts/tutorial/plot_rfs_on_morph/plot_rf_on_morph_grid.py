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
# # Mouse-cam receptive fields on morphology: grid of cells
#
# Same as `plot_rf_on_morph_single.py`, but for a 4 x 4 grid of cells: the best cell (highest test correlation of the RF
# model) of each of the 16 best cell types. Each panel shows the sRF of the cell's dominant channel (the channel with
# the larger sRF peak), the skeleton, the 2-sigma DoG ellipses of both channels (center solid, surround dotted) and the
# tRFs as an inset.
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

N_ROWS, N_COLS = 4, 4
ONE_PER_TYPE = True  # at most one cell per cell type
R2_MIN = 0.5  # minimum variance of the sRF explained by the DoG fit (in at least one channel)

# %% [markdown]
# ## Load data
#
# Load the ROIs with their matched EM cells, join the RFs, and keep cells with a skeleton, good response quality
# (`qfilt`) and a well-fitted sRF whose center lies inside the stimulus.

# %%
df_rois = data_loader.load_df_rois()
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = df.merge(data_loader.load_df_mc_rfs(), on=KEY_COLS, validate='one_to_one')
df = data_loader.add_skels(df=df, inplace=True)

n_px = df.mc_rf_green_srf.iloc[0].shape[0]
r2 = np.maximum(df.mc_rf_green_dog_r2, df.mc_rf_uv_dog_r2)
center_inside = np.all([df[f'mc_rf_{ch}_dog_{xy}'].between(2, n_px - 3) for ch in mc.MC_CHANNELS for xy in 'xy'],
                       axis=0)
df = df[df.skel.notnull() & df.qfilt & (df.mc_rf_best_alpha_idx >= 0) & (r2 >= R2_MIN) & center_inside]
df = df.sort_values('mc_rf_best_cc', ascending=False)
if ONE_PER_TYPE:
    df = df[df['Cell Type'].notnull()].drop_duplicates('Cell Type')
df_plot = df.head(N_ROWS * N_COLS)
print(f'{len(df)} cells pass, plotting {len(df_plot)}')
df_plot[['field', 'roi_id', 'Cell Type', 'mc_rf_best_cc']]

# %% [markdown]
# ## Plot
#
# Each panel is centered on the soma and zoomed to the dendrites or the DoG centers, whichever is larger (scale bar:
# 100 µm). sRF colors: red = positive, blue = negative, normalized per cell. sRF and tRF are flipped such that the
# fitted center is positive, i.e. the tRF carries the polarity (positive just before the response = ON center).

# %%
reg = registration.load_registration(REG_FILE)

fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=(2.6 * N_COLS, 2.75 * N_ROWS), squeeze=False)
for ax, (_, row) in zip(axs.flat, df_plot.iterrows()):
    skel = space_mapping.align_and_place_skel(row.skel, reg, field=row.field, target_xy=(0., 0.))
    ch = max(mc.MC_CHANNELS, key=lambda c: np.nanmax(np.abs(row[f'mc_rf_{c}_srf'])))  # dominant channel
    rad = max(plot_rf.cell_radius(skel), plot_rf.rf_radius(row))
    lim = (-rad, rad, -rad, rad)
    plot_rf.plot_srf_on_morph(ax, row, skel, ch, fit_chs=mc.MC_CHANNELS, skel_lw=0.5)
    ax.set(xlim=lim[:2], ylim=lim[2:])
    plot_rf.add_scale_bar(ax, lim, size=100, fontsize=6)
    ax.set_title(f"{row['Cell Type']}", fontsize=8, pad=3)
    ax.text(0.02, 0.98, f'{plot_rf.CH_LABELS[ch]} sRF\ncc {row.mc_rf_best_cc:.2f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=6, color=plot_rf.CH_COLORS[ch])

    ax_in = ax.inset_axes([0.62, 0.70, 0.36, 0.27])
    ax_in.patch.set(facecolor='w', alpha=0.75)
    plot_rf.plot_trfs(ax_in, row, lw=0.9, labels=False)
    ax_in.set_xticks([])
    ax_in.spines['bottom'].set_visible(False)
for ax in axs.flat[len(df_plot):]:
    ax.axis('off')

fig.legend(handles=[Line2D([], [], c=plot_rf.CH_COLORS[ch], label=plot_rf.CH_LABELS[ch]) for ch in mc.MC_CHANNELS]
                   + [Line2D([], [], c='0.3', ls='-', label='DoG center (2σ)'),
                      Line2D([], [], c='0.3', ls=':', label='DoG surround (2σ)')],
           loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.text(0.5, 1.0, 'Inset: tRFs from -1.2 to 0.3 s relative to the response', ha='center', fontsize=8, color='0.3')
plt.tight_layout(rect=(0, 0.03, 1, 0.99))
fig.savefig(os.path.join(FIG_DIR, 'plot_rf_on_morph_grid.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'plot_rf_on_morph_grid.png'), dpi=200, bbox_inches='tight')
plt.show()
