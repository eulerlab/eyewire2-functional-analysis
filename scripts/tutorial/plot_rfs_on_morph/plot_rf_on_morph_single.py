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
# # Mouse-cam receptive field on morphology: single cell
#
# Plot the pre-estimated mouse-cam receptive field (RF) of one cell on top of its EM skeleton. The RFs are loaded from
# `df_eyewire2_roi_level_mc_rf.parquet` (see `scripts/preprocessing/mc_rfs/fit_mc_rfs.py` and the column docs in
# `df_eyewire2_roi_level_mc_rf.md`): per ROI, the spatial RF (sRF) and temporal RF (tRF) of the green and UV channel of a
# lasso GLM, and a difference-of-Gaussians (DoG) fit of the sRFs.
#
# The skeleton is rotated into the retinal frame with the fitted 2p <-> EM registration (see
# `scripts/preprocessing/em-2p-mapping.py`) and placed with its soma at the ROI position. The sRF is placed with the
# stimulus centered on the recording field. Right = nasal, top = dorsal.
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

# Cell to plot as (field, roi_id); None = the cell with the best test correlation among the cells that pass the
# quality criteria below.
EXAMPLE = None
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
print(f'{len(df)} cells pass')
df[['field', 'roi_id', 'Cell Type', 'mc_rf_best_cc', 'mc_rf_green_dog_r2', 'mc_rf_uv_dog_r2']].head(10)

# %%
if EXAMPLE is None:
    row = df.iloc[0]
else:
    row = df[(df.field == EXAMPLE[0]) & (df.roi_id == EXAMPLE[1])].iloc[0]

reg = registration.load_registration(REG_FILE)
skel = space_mapping.align_and_place_skel(row.skel, reg, field=row.field, target_xy=(0., 0.))  # soma at the origin
print(f"{row.field} ROI {row.roi_id}: {row['Cell Type']}, test cc {row.mc_rf_best_cc:.2f}")

# %% [markdown]
# ## Plot
#
# - **green / UV**: sRF of each channel (red = positive, blue = negative; both channels on the same color scale) with
#   the skeleton and the 2-sigma ellipses of the DoG center (solid) and surround (dotted). sRF and tRF are flipped such
#   that the fitted center is positive, i.e. the tRF carries the polarity: a tRF that is positive just before the
#   response means an ON center.
# - **stimulus**: the whole stimulus window (gray frame) with the skeleton and both channels' DoG ellipses.
# - **tRF**: temporal RFs, scaled by the sRF peak so that the channels are comparable.

# %%
rad = max(plot_rf.cell_radius(skel), plot_rf.rf_radius(row))
lim = (-rad, rad, -rad, rad)
vabsmax = max(np.nanmax(np.abs(row[f'mc_rf_{ch}_srf'])) for ch in mc.MC_CHANNELS)

fig, axs = plt.subplots(1, 4, figsize=(12, 3.3), width_ratios=(1, 1, 1, 1.1))
for ax, ch in zip(axs[:2], mc.MC_CHANNELS):
    im = plot_rf.plot_srf_on_morph(ax, row, skel, ch, vabsmax=vabsmax)
    ax.set(xlim=lim[:2], ylim=lim[2:])
    ax.set_title(f'{plot_rf.CH_LABELS[ch]} sRF (DoG r² = {row[f"mc_rf_{ch}_dog_r2"]:.2f})', color=plot_rf.CH_COLORS[ch])
    plot_rf.add_scale_bar(ax, lim, size=100)
cbar = fig.colorbar(im, ax=axs[1], fraction=0.04, pad=0.02, ticks=[-vabsmax, 0, vabsmax])
cbar.ax.set_yticklabels(['−', '0', '+'])
cbar.outline.set_visible(False)

ax = axs[2]
_, ext = plot_rf.srf_retina(row, 'green')
plot_rf.plot_srf_on_morph(ax, row, skel, 'green', img_alpha=0., fit_chs=mc.MC_CHANNELS, skel_lw=0.4)
pad = 0.03 * (ext[1] - ext[0])
lim_stim = (ext[0] - pad, ext[1] + pad, ext[2] - pad, ext[3] + pad)
ax.set(xlim=lim_stim[:2], ylim=lim_stim[2:], title='stimulus window')
ax.plot(0, 0, 'k+', ms=6, mew=0.8)  # soma
plot_rf.add_scale_bar(ax, lim_stim, size=200)

plot_rf.plot_trfs(axs[3], row)
axs[3].set_title('tRF')
axs[3].legend(handles=[Line2D([], [], c=plot_rf.CH_COLORS[ch], label=plot_rf.CH_LABELS[ch]) for ch in mc.MC_CHANNELS]
                      + [Line2D([], [], c='0.3', ls='-', label='DoG center'),
                         Line2D([], [], c='0.3', ls=':', label='DoG surround')],
              frameon=False, fontsize=7, loc='upper left')

fig.suptitle(f"{row['Cell Type']} ({row.field} ROI {row.roi_id}), test correlation {row.mc_rf_best_cc:.2f}", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'plot_rf_on_morph_single.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'plot_rf_on_morph_single.png'), dpi=200, bbox_inches='tight')
plt.show()
