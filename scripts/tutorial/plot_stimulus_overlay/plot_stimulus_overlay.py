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
# # Stimulus overlay grid: field x stimulus type
#
# For each of the 5 recording fields (rows) and each of the 3 stimulus types
# (columns), draw that field's stimulus footprint -- at its true recorded
# position and physical scale -- on top of the retinal outline, cropped to
# the lower-left wing of the retina where all the recording fields sit (with
# (0, 0) at the crop's top-right corner).
#
# There is only one recording session in this dataset, so
# `data/experiment-overview_consolidated.csv` covers the whole thing.

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd

from eyewire2_functional_analysis import data_loader
from eyewire2_functional_analysis.stimulus import stim_outlines

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

FIELDS = ['GCL0', 'GCL1', 'GCL2', 'GCL3', 'GCL4']  # rows
STIM_TYPES = ['Chirp', 'DS', 'MouseCam_Right']  # columns
STIM_COL_TITLES = {'Chirp': 'Chirp', 'DS': 'Moving bar', 'MouseCam_Right': 'Mouse cam'}

# %% [markdown]
# ## Stimulus geometry
#
# Parameters mirror `scripts/analysis/light_exposure/EW2_stim_history.py`.

# %%
FOV_DIAM_UM = 1000  # approx. field of view through the objective (W Plan-Apochromat 20x/1.0, Zeiss)

BAR_DY_UM = 300.0
BAR_DX_UM = 1000.0
BAR_VEL_UM_S = 1000.0
BAR_DUR_S = 4.0
BAR_DIR_LIST = [0, 180, 45, 225, 90, 270, 135, 315]

BAR_L_EDGE_UM = BAR_DY_UM
BAR_TRAJ_LEN_UM = BAR_DUR_S * BAR_VEL_UM_S + BAR_DX_UM

CHIRP_DIAM_UM = 1000

MOUSECAM_FRAME_PX = 56
MOUSECAM_SCALE = 12.5
MOUSECAM_DX_UM = MOUSECAM_FRAME_PX * MOUSECAM_SCALE
MOUSECAM_DY_UM = MOUSECAM_FRAME_PX * MOUSECAM_SCALE


def stimulus_footprint(stim_type, x0, y0):
    """Shapely footprint of one stimulus presentation, positioned at (x0, y0), clipped to the FOV."""
    if stim_type == 'DS':
        return stim_outlines.movingBar(BAR_L_EDGE_UM, BAR_TRAJ_LEN_UM, BAR_DIR_LIST,
                                        x0=x0, y0=y0, FOV_diam=FOV_DIAM_UM)
    elif stim_type == 'Chirp':
        return stim_outlines.spot(diam=CHIRP_DIAM_UM, x0=x0, y0=y0, FOV_diam=FOV_DIAM_UM)
    elif stim_type == 'MouseCam_Right':
        return stim_outlines.box(MOUSECAM_DX_UM, MOUSECAM_DY_UM, x0=x0, y0=y0, FOV_diam=FOV_DIAM_UM)
    else:
        raise ValueError(f"Unknown stimulus type {stim_type!r}")


def plot_shapely_polygon(ax, geom, **kwargs):
    """Fill+outline a Shapely Polygon or MultiPolygon (e.g. after clipping to a FOV aperture)."""
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    for poly in polys:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, **kwargs)


# %% [markdown]
# ## Pick one presentation per field x stimulus type
#
# Some field/stimulus combos have more than one recording in the log (e.g.
# GCL3 has two separate Chirp recordings). Prefer the earliest presentation
# that has an actual recording file (`dataFileName`), falling back to the
# earliest overall if none do.
#
# Note: `pos_xyz` in the log is a *raw* stage position (QDSpy log/`.smh`
# headers, the microscope's own coordinate frame) -- not the
# `temporal_nasal_pos_um`/`ventral_dorsal_pos_um` retinal frame used
# everywhere else in the released dataset. So it's only used here to find
# *which* presentation happened; the actual plotting position for each field
# comes from that field's already-registered centre in `load_df_fields`.

# %%
CONSOL_PATH = os.path.join(data_loader.DATA_ROOT, 'experiment-overview_consolidated.csv')
df_stim_log = pd.read_csv(CONSOL_PATH, sep=';', on_bad_lines='warn')
df_stim_log = df_stim_log[df_stim_log['pos_xyz'].notna()].copy()
df_stim_log['fieldID'] = df_stim_log['fieldID'].astype(int)


def pick_presentation(field_idx, stim_type):
    candidates = df_stim_log[(df_stim_log['fieldID'] == field_idx) & (df_stim_log['stimFileName'] == stim_type)]
    assert len(candidates), f"No '{stim_type}' presentation logged for field {field_idx}"
    with_recording = candidates[candidates['dataFileName'].notna()]
    pool = with_recording if len(with_recording) else candidates
    return pool.loc[pool['t_abs_s'].idxmin()]


picks = {(field_name, stim_type): pick_presentation(int(field_name[3:]), stim_type)
         for field_name in FIELDS for stim_type in STIM_TYPES}

print(pd.DataFrame([
    {'field': field_name, 'stimulus': stim_type, 't_abs_s': p['t_abs_s'], 'dataFileName': p['dataFileName']}
    for (field_name, stim_type), p in picks.items()
]).to_string(index=False))

# %% [markdown]
# ## Plot

# %%
df_outline = data_loader.load_df_outline()
df_fields = data_loader.load_df_fields()
df_rois = data_loader.load_df_rois()

XLIM = (df_outline['temporal_nasal_pos_um'].min(), 300)
YLIM = (df_outline['ventral_dorsal_pos_um'].min(), 300)

fig, axs = plt.subplots(len(FIELDS), len(STIM_TYPES),
                         figsize=(3.2 * len(STIM_TYPES), 3.2 * len(FIELDS)),
                         sharex=True, sharey=True)

for r, field_name in enumerate(FIELDS):
    field_row = df_fields[df_fields['field'] == field_name].iloc[0]
    x0 = field_row['field_temporal_nasal_pos_um']
    y0 = field_row['field_ventral_dorsal_pos_um']

    rois_other = df_rois[df_rois['field'] != field_name]
    rois_field = df_rois[df_rois['field'] == field_name]

    for c, stim_type in enumerate(STIM_TYPES):
        ax = axs[r, c]
        ax.plot(df_outline['temporal_nasal_pos_um'], df_outline['ventral_dorsal_pos_um'],
                c='dimgray', ls=':', lw=1)

        footprint = stimulus_footprint(stim_type, x0, y0)
        plot_shapely_polygon(ax, footprint, facecolor='yellow', edgecolor='k', linewidth=1, alpha=0.4, zorder=5)

        ax.scatter(rois_other['temporal_nasal_pos_um'], rois_other['ventral_dorsal_pos_um'],
                   s=4, color='white', edgecolor='black', linewidth=0.2, alpha=0.8, zorder=6)
        ax.scatter(rois_field['temporal_nasal_pos_um'], rois_field['ventral_dorsal_pos_um'],
                   s=10, color='red', edgecolor='black', linewidth=0.2, zorder=7)

        ax.set(xlim=XLIM, ylim=YLIM)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

        if r == 0:
            ax.set_title(STIM_COL_TITLES[stim_type])
        if c == 0:
            ax.set_ylabel(f"{field_name}\nVentral <-> Dorsal [um]")
        if r == len(FIELDS) - 1:
            ax.set_xlabel('Temporal <-> Nasal [um]')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'stimulus_overlay_grid.pdf'))
plt.show()

# %%
