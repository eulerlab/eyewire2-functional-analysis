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
# # Readme
#
# Plot all functional and retina edge recordings.
# Connect the edge recordings to show the outline of the Retina.
# Note that the first edge recording is actually the optic disc.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from matplotlib.collections import LineCollection

# %%
from eyewire2_functional_analysis import data_loader, neuroglancer, registration
from eyewire2_functional_analysis.plot import plot_em_axis_indicator
from eyewire2_functional_analysis.space_mapping import align_and_place_skel

df_fields = data_loader.load_df_fields()
df_outline = data_loader.load_df_outline()


# %%
def plot_stack_average(ax, stack_avg, pixel_size_um, x_offset, y_offset, cmap='viridis', alpha=0.7, gamma=0.5):
    ps = pixel_size_um
    w, h = stack_avg.shape[:2]
    assert w == h
    extent = np.array([-w / 2 * ps, +w / 2 * ps, -w / 2 * ps, +w / 2 * ps])
    extent += (x_offset, x_offset, y_offset, y_offset)

    im = stack_avg.astype(float)
    vmin = np.percentile(im, q=5, axis=(0, 1))
    vmax = np.percentile(im, q=99, axis=(0, 1))
    im = (im - vmin) / (vmax - vmin)
    im = np.clip(im, 0, 1) ** gamma

    ax.imshow(im.T, extent=extent, cmap=cmap, interpolation='none', alpha=alpha)


# %%
HERE = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(HERE, 'figures', 'retinal_locations')
os.makedirs(fig_dir, exist_ok=True)

# %%
import matplotlib.patches as patches

fig, axs_all = plt.subplots(2, 2, figsize=(15, 15))

axs = axs_all[0, :]
for i, row in df_outline.iterrows():
    for j, ax in enumerate(axs):
        plot_stack_average(ax, row['stack_averages'][:, :, j],
                           row['pixel_size_um'],
                           row['temporal_nasal_pos_um'], row['ventral_dorsal_pos_um'],
                           cmap=['viridis', 'Reds'][j])

    for ax in axs:
        ax.set(xlim=(-3000, 3000), ylim=(-3000, 3000),
               xlabel='Temporal <-> Nasal [um]', ylabel='Ventral <-> Dorsal [um]')
        ax.plot(df_outline['temporal_nasal_pos_um'],
                df_outline['ventral_dorsal_pos_um'], c='dimgray', ls=':')
        ax.grid()
        ax.set_aspect('equal')

for axs in axs_all:
    for i, row in df_fields.iterrows():
        for j in [0, 1]:
            plot_stack_average(axs[j], row[f'ch{j}_average'],
                               row['pixel_size_um'],
                               row['field_temporal_nasal_pos_um'], row['field_ventral_dorsal_pos_um'],
                               cmap=['viridis', 'Reds'][j])
box_xlim = (-1100, -550)
box_ylim = (-700, -150)
for ax in axs_all[1, :]:
    ax.set(xlim=box_xlim, ylim=box_ylim,
           xlabel='Temporal <-> Nasal [um]', ylabel='Ventral <-> Dorsal [um]')
    ax.grid()
    ax.set_aspect('equal')

for ax in axs_all.flat:
    rect = patches.Rectangle(
        (box_xlim[0], box_ylim[0]), box_xlim[1] - box_xlim[0], box_ylim[1] - box_ylim[0],
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)

axs_all[0, 0].set_title('OGB-1')
axs_all[0, 1].set_title('SR-101')

axs_all[1, 0].set_title('OGB-1; functional recordings')
axs_all[1, 1].set_title('SR-101; functional recordings')

# %% [markdown]
# ## Overlay example cell morphologies
#
# Add 3 example cells, frozen by "Latest SegID" (rather than re-sampled each
# run) so the same 3 cells keep showing up regardless of upstream data
# changes/reordering: "OFF transient alpha" (field GCL3), "ON-OFF DS -
# ventral" (field GCL4 -- the same cell used in
# `scripts/tutorial/plot_DS_on_morph/plot_DS_on_morph.py`), and "nNOS-1"
# (field GCL4). Each skeleton is rotated into the 2p/retinal reference frame
# with the fitted em_to_2p registration and then translated to its true 2p
# ROI position -- `registration.align_skel` only rotates about the
# skeleton's own (still raw-EM) soma centre, it doesn't move it into 2p
# space.

# %%
EXAMPLE_CELLS = {
    "OFF transient alpha": "720575940559143627",
    "ON-OFF DS - ventral": "720575940567267766",
    "nNOS-1": "720575940556012650",
}
REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)

df_rois = data_loader.load_df_rois()
df_morph = data_loader.load_df_rois_morph(df_rois=df_rois)
df_morph = data_loader.add_skels(df=df_morph, inplace=True)
df_morph = df_morph[df_morph['skel'].notnull()].copy()

example_rows = {}
for cell_type, seg_id in EXAMPLE_CELLS.items():
    row = df_morph[df_morph['Latest SegID'] == seg_id].iloc[0]
    assert row['Cell Type'] == cell_type, f"Latest SegID {seg_id} is now '{row['Cell Type']}', not '{cell_type}'"
    example_rows[f"{row['field']}: {cell_type}"] = row

reg = registration.load_registration(REG_FILE)
example_colors = dict(zip(example_rows, ['C1', 'C2', 'C3']))

for label, row in example_rows.items():
    skel = align_and_place_skel(
        row.skel, reg, field=row['field'],
        target_xy=(row['temporal_nasal_pos_um'], row['ventral_dorsal_pos_um']),
    )
    color = example_colors[label]
    for ax in axs_all[:, 1]:
        seg = np.stack([skel.nodes[skel.edges[:, 0], :2], skel.nodes[skel.edges[:, 1], :2]], axis=1)
        ax.add_collection(LineCollection(seg, colors=color, linewidths=0.8, alpha=0.9, zorder=5))
        ax.scatter(*skel.soma.center[:2], color=color, s=25, zorder=6, label=label)

axs_all[0, 1].legend(loc='upper right', fontsize='small')

# %% [markdown]
# ## EM X/Y axis indicator
#
# Two 200 um lines sharing a start point (the zoomed-box centre) showing how
# EM's native X (red) and Y (green) axes are actually oriented once rotated
# into the 2p/retinal reference frame -- using the *global* em_to_2p rotation
# only (no per-field refinement). Shared with
# `scripts/tutorial/plot_DS_on_morph/plot_DS_on_morph.py` via
# `plot.plot_em_axis_indicator`.

# %%
AXIS_LEN_UM = 200
box_center = (np.mean(box_xlim), np.mean(box_ylim))

for ax in axs_all[1, :]:
    plot_em_axis_indicator(ax, reg, field=None, direction='em_to_2p', center=box_center, scale=AXIS_LEN_UM)

plt.savefig(os.path.join(fig_dir, 'retinal_field_locations.pdf'))
plt.show()

# %% [markdown]
# ## Neuroglancer link, for verification
#
# Link with one segmentation layer per example cell (so its full EM
# reconstruction, e.g. an axon, can be traced) plus, per 2p field, a point
# annotation layer marking every matched EM cell's real "Nuc Coords" (as in
# `scripts/preprocessing/em-2p-mapping.py`) -- so each example cell's true
# EM position can be checked against its own field's matched-cell point
# cloud, and against the position it was placed at above.

# %%
DATASTACK_NAME = "stroeh_mouse_retina"
LINK_FILE = "neuroglancer_link.txt"

df_map = pd.read_csv(os.path.join(data_loader.DATA_SS, data_loader.MAP_SHEET))

client = CAVEclient(datastack_name=DATASTACK_NAME)
example_cells = {label: (row['Latest NucID'], row['Latest SegID']) for label, row in example_rows.items()}
link = neuroglancer.spawn_example_cells_link(client, example_cells, df_map)

link_path = os.path.join(HERE, LINK_FILE)
with open(link_path, "w") as f:
    f.write(link + "\n")
print(f"wrote {link_path}")

# %%
