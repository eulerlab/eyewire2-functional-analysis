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
# # 2p <-> EM coordinate registration
#
# Fits the rotation angle and an isotropic scale that best map 2p ROI
# coordinates onto EM soma coordinates, using the known EM-cell <-> 2p-ROI
# correspondences in the EM-2p mapping table (instead of just guessing the
# angle). Flips are kept fixed since we already know they happen and why.
#
# The fit (global + per-field refinement, in both directions) is done by
# `eyewire2_functional_analysis.registration` and persisted to `REG_FILE` so
# it can be reused elsewhere (e.g. to rotate EM skeletons into the 2p/retinal
# reference frame for plotting) without re-fitting. Set `REFIT = True` below
# to force a re-fit even if `REG_FILE` already exists.
#
# Saves figures to `./figures/` next to this script.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient

from eyewire2_functional_analysis import data_loader, neuroglancer, registration
from eyewire2_functional_analysis.space_mapping import fit_z_plane, map_coords_per_row, predict_z_plane

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

data_folder = data_loader.DATA_2P
morph_folder = data_loader.DATA_SS

MAP_FILE = data_loader.MAP_SHEET
OUT_FILE = "2p_roi_estimated_em_coordinates.csv"
LINK_FILE = "neuroglancer_link.txt"

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
REFIT = False  # set True to re-fit and overwrite REG_FILE even if it already exists

DATASTACK_NAME = "stroeh_mouse_retina"

DEG_GUESS = -125.0
FLIP_X = True
FLIP_Y = False
MIN_MATCHED_PER_FIELD = 3


# %%
def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


# %% [markdown]
# ## Load data

# %%
df_rois, df_fields, df_outline = data_loader.load_all_dfs(data_folder)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})
sns.scatterplot(data=df_rois, x='temporal_nasal_pos_um', y='ventral_dorsal_pos_um', hue='field', alpha=0.5, ax=ax)
ax.set_title('2p ROI coordinates')
savefig("01_2p_rois.pdf")

# %%
df_map = pd.read_csv(os.path.join(morph_folder, MAP_FILE))
# Some 2p ROIs are marked with no "Nuc Coords" (read in as NaN by
# pandas) because they turned out not to be a real, single EM cell (e.g.
# an overlay of two cells) — keep the row, but its EM coordinate is
# unknown, so coerce rather than a hard int cast that would choke on NaN.
df_map[['em_x_um', 'em_y_um', 'em_z_um']] = (
    df_map['Nuc Coords']
    .str.split(',', expand=True)
    .apply(pd.to_numeric, errors='coerce')
    * np.array([16, 16, 40]) / 1000
)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})
sns.scatterplot(data=df_map, x='em_x_um', y='em_y_um', hue='2p-Field', alpha=0.5, ax=ax)
ax.set_title('EM soma coordinates')
savefig("02_em_somas.pdf")

# %% [markdown]
# ## Naive alignment
#
# Naive alignment with the guessed `DEG_GUESS`, no scaling, just for comparison.

# %%
coords_em = df_map[['em_x_um', 'em_y_um']].values
coords_2p = df_rois[['temporal_nasal_pos_um', 'ventral_dorsal_pos_um']].values
center_em = np.mean(coords_em, axis=0)
center_2p = np.mean(coords_2p, axis=0)

df_map[['em_rotx_um', 'em_roty_um']] = registration.rotate_and_flip_coordinates(
    coords_em, angle_deg=DEG_GUESS, scale=1.0, flip_x=FLIP_X, flip_y=FLIP_Y, center=center_em, new_center=center_2p)
df_rois[['roi_rotx_um', 'roi_roty_um']] = registration.rotate_and_flip_coordinates(
    coords_2p, angle_deg=DEG_GUESS, scale=1.0, flip_x=FLIP_X, flip_y=FLIP_Y, center=center_2p, new_center=center_em)

fig, axs = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={"aspect": "equal"})
sns.scatterplot(data=df_map, x='em_x_um', y='em_y_um', hue='2p-Field', alpha=0.5, ax=axs[0, 0])
axs[0, 0].set_title('Original EM Coordinates')
sns.scatterplot(data=df_map, x='em_rotx_um', y='em_roty_um', hue='2p-Field', alpha=0.5, ax=axs[0, 1])
axs[0, 1].set_title('Rotated EM Coordinates (guessed DEG)')
sns.scatterplot(data=df_rois, x='temporal_nasal_pos_um', y='ventral_dorsal_pos_um', hue='field', alpha=0.5, ax=axs[1, 1])
axs[1, 1].set_title('2P Coordinates')
sns.scatterplot(data=df_rois, x='roi_rotx_um', y='roi_roty_um', hue='field', alpha=0.5, ax=axs[1, 0])
axs[1, 0].set_title('Rotated 2p Coordinates (guessed DEG)')
plt.tight_layout()
savefig("03_naive_alignment.pdf")

# %% [markdown]
# ## Fit (or load) the 2p<->EM registration
#
# `df_map` gives, for each EM cell, the exact 2p field + ROI it corresponds
# to (`2p-Field`/`2p-ROI` <-> `df_rois`' `field`/`roi_id`). Instead of
# aligning the two point clouds blindly, use these known correspondences to
# fit the rotation angle and an isotropic scale by least squares (flips are
# kept fixed; translation is not a free parameter: for a fixed
# rotation/scale the optimal translation is the one that matches the
# centroids, so both point sets are centered on their own centroid and
# translated to the target centroid afterwards). A per-field rotation +
# scale + translation refinement is then fit on top, since the true 2p<->EM
# deformation isn't a single rigid map (it varies smoothly across the
# retina, e.g. from local tissue stretch). Both directions (`2p_to_em` and
# `em_to_2p`) are fit independently (not algebraically inverted from one
# another) -- see `registration.fit_registration`.
#
# By default this loads `REG_FILE` if it already exists rather than
# re-fitting; set `REFIT = True` above to force a re-fit.

# %%
reg = registration.load_or_fit_registration(
    REG_FILE, df_map=df_map, df_rois=df_rois, refit=REFIT,
    flip_x=FLIP_X, flip_y=FLIP_Y, min_matched_per_field=MIN_MATCHED_PER_FIELD, x0_angle_2p_to_em=DEG_GUESS,
)
print(f"{'refit' if REFIT or not os.path.exists(REG_FILE) else 'loaded'} registration ({REG_FILE})")

g_2p_to_em = reg['directions']['2p_to_em']['global']
print(f"2p->EM global: angle = {g_2p_to_em['angle_deg']:.2f} deg, scale = {g_2p_to_em['scale']:.3f}, "
      f"RMSE = {g_2p_to_em['rmse_um']:.2f} um (n={g_2p_to_em['n_matched']})")

# %% [markdown]
# ## Matched pairs, for diagnostics
#
# Re-derive the matched (2p, EM) pairs (same correspondences used for the
# fit) purely to plot/inspect it below; the fit itself already happened
# inside `fit_registration`.

# %%
df_matched = df_map.merge(
    df_rois[['field', 'roi_id', 'temporal_nasal_pos_um', 'ventral_dorsal_pos_um']],
    left_on=['2p-Field', '2p-ROI'],
    right_on=['field', 'roi_id'],
    how='inner',
)
n_not_a_cell = df_matched['em_x_um'].isna().sum()
if n_not_a_cell:
    print(f"Excluding {n_not_a_cell} ROI(s) with no EM coordinate (not a real cell) from the diagnostics")
    df_matched = df_matched[df_matched['em_x_um'].notna()].reset_index(drop=True)
print(f"Matched {len(df_matched)} / {len(df_map)} EM cells to a 2p ROI")

matched_em = df_matched[['em_x_um', 'em_y_um']].to_numpy()
matched_2p = df_matched[['temporal_nasal_pos_um', 'ventral_dorsal_pos_um']].to_numpy()
matched_fields = df_matched['field'].to_numpy()

# %%
initial_matched = registration.rotate_and_flip_coordinates(
    matched_2p, angle_deg=DEG_GUESS, scale=1.0, flip_x=FLIP_X, flip_y=FLIP_Y,
    center=np.mean(matched_2p, axis=0), new_center=np.mean(matched_em, axis=0),
)
global_matched = registration.rotate_and_flip_coordinates(
    matched_2p, angle_deg=g_2p_to_em['angle_deg'], scale=g_2p_to_em['scale'], flip_x=FLIP_X, flip_y=FLIP_Y,
    center=g_2p_to_em['center_source'], new_center=g_2p_to_em['center_target'],
)
final_matched = map_coords_per_row(matched_2p, matched_fields, reg, direction='2p_to_em')

rmse_initial = np.sqrt(np.mean(np.sum((initial_matched - matched_em) ** 2, axis=1)))
rmse_global = np.sqrt(np.mean(np.sum((global_matched - matched_em) ** 2, axis=1)))
rmse_final = np.sqrt(np.mean(np.sum((final_matched - matched_em) ** 2, axis=1)))
print(f"RMSE: naive guess = {rmse_initial:.2f} um, global fit = {rmse_global:.2f} um, "
      f"global+per-field fit = {rmse_final:.2f} um")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"aspect": "equal"})

axs[0].scatter(*initial_matched.T, c='C1', alpha=0.5, label='2p aligned (naive guess)')
axs[0].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
for p2_pt, em_pt in zip(initial_matched, matched_em):
    axs[0].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
axs[0].set_title(f"Naive guess (angle={DEG_GUESS:.1f})\nRMSE={rmse_initial:.1f} um")
axs[0].legend(loc='upper right', fontsize='small')

axs[1].scatter(*global_matched.T, c='C1', alpha=0.5, label='2p aligned (global fit)')
axs[1].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
for p2_pt, em_pt in zip(global_matched, matched_em):
    axs[1].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
axs[1].set_title(f"Global fit (angle={g_2p_to_em['angle_deg']:.1f}, scale={g_2p_to_em['scale']:.2f})\n"
                  f"RMSE={rmse_global:.1f} um")
axs[1].legend(loc='upper right', fontsize='small')

plt.tight_layout()
savefig("04_naive_vs_global_matched_pairs.pdf")

# %%
df_rois['roi_rotx_opt_um'], df_rois['roi_roty_opt_um'] = registration.rotate_and_flip_coordinates(
    coords_2p, angle_deg=g_2p_to_em['angle_deg'], scale=g_2p_to_em['scale'], flip_x=FLIP_X, flip_y=FLIP_Y,
    center=g_2p_to_em['center_source'], new_center=g_2p_to_em['center_target'],
).T

fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"aspect": "equal"})
sns.scatterplot(data=df_map, x='em_x_um', y='em_y_um', hue='2p-Field', alpha=0.5, ax=axs[0])
axs[0].set_title('EM coordinates (reference)')
sns.scatterplot(data=df_rois, x='roi_rotx_opt_um', y='roi_roty_opt_um', hue='field', alpha=0.5, ax=axs[1])
axs[1].set_title('2p coordinates, global fit')
plt.tight_layout()
savefig("05_global_alignment_overview.pdf")

# %% [markdown]
# ## Per-field refinement diagnostics

# %%
print("\nPer-field rigid refinement:")
field_diagnostics = []
for field in sorted(df_matched['field'].unique()):
    mask = matched_fields == field
    n = int(mask.sum())
    field_em = matched_em[mask]
    field_2p_global = global_matched[mask]
    rmse_before = np.sqrt(np.mean(np.sum((field_2p_global - field_em) ** 2, axis=1)))

    field_fit = reg['directions']['2p_to_em']['fields'].get(field)
    if field_fit is None:
        print(f"  field {field}: only {n} matched cell(s), no per-field fit (RMSE={rmse_before:.2f} um)")
        rmse_after = rmse_before
    else:
        rmse_after = field_fit['rmse_um']
        print(f"  field {field}: n={n}, angle={field_fit['angle_deg']:.2f} deg, scale={field_fit['scale']:.3f}"
              f"  ->  RMSE {rmse_before:.2f} -> {rmse_after:.2f} um")
    field_diagnostics.append({'field': field, 'n': n, 'rmse_before': rmse_before, 'rmse_after': rmse_after})

df_rois['roi_rotx_final_um'], df_rois['roi_roty_final_um'] = map_coords_per_row(
    coords_2p, df_rois['field'].to_numpy(), reg, direction='2p_to_em').T
df_matched['roi_rotx_final_um'], df_matched['roi_roty_final_um'] = final_matched.T

print(f"\nOverall RMSE: global={rmse_global:.2f} um, global+per-field={rmse_final:.2f} um")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"aspect": "equal"})
sns.scatterplot(data=df_map, x='em_x_um', y='em_y_um', hue='2p-Field', alpha=0.5, ax=axs[0])
axs[0].set_title('EM coordinates (reference)')
sns.scatterplot(data=df_rois, x='roi_rotx_final_um', y='roi_roty_final_um', hue='field', alpha=0.5, ax=axs[1])
axs[1].set_title('2p coordinates, global + per-field alignment')
plt.tight_layout()
savefig("06_per_field_alignment_overview.pdf")

# %%
df_diag = pd.DataFrame(field_diagnostics)
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(df_diag))
width = 0.35
ax.bar(x - width / 2, df_diag['rmse_before'], width, label='global only')
ax.bar(x + width / 2, df_diag['rmse_after'], width, label='global + per-field')
ax.set_xticks(x)
ax.set_xticklabels(df_diag['field'], rotation=90)
ax.set_ylabel('RMSE (um)')
ax.set_title('Per-field alignment RMSE')
ax.legend()
plt.tight_layout()
savefig("07_per_field_rmse.pdf")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"aspect": "equal"})

axs[0].scatter(*global_matched.T, c='C1', alpha=0.5, label='2p aligned (global)')
axs[0].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
for p2_pt, em_pt in zip(global_matched, matched_em):
    axs[0].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
axs[0].set_title(f"Global alignment\nRMSE={rmse_global:.1f} um")
axs[0].legend(loc='upper right', fontsize='small')

axs[1].scatter(*final_matched.T, c='C1', alpha=0.5, label='2p aligned (global + per-field)')
axs[1].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
for p2_pt, em_pt in zip(final_matched, matched_em):
    axs[1].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
axs[1].set_title(f"Global + per-field alignment\nRMSE={rmse_final:.1f} um")
axs[1].legend(loc='upper right', fontsize='small')

plt.tight_layout()
savefig("08_global_vs_per_field_matched_pairs.pdf")

# %% [markdown]
# ## Estimated EM-space coordinates for every 2p ROI
#
# Combine the aligned x/y (global + per-field) with a z estimate. Rather
# than a single per-field mean z (too coarse: z drifts smoothly across a
# field, same as x/y did), fit a 2D linear plane z ~= a*x + b*y + c per
# field on that field's matched (x, y, z) triples and use the plane to
# predict z for every ROI in the field. Write everything out as a single
# "x, y, z" EM *voxel* coordinate column, ready to paste into Neuroglancer
# to go find the still-unmatched cells.

# %%
df_out = df_rois[['field', 'roi_id', 'roi_rotx_final_um', 'roi_roty_final_um']].copy()
matched_keys = set(zip(df_matched['field'], df_matched['roi_id']))
not_a_cell_keys = set(zip(
    df_map.loc[df_map['em_x_um'].isna(), '2p-Field'],
    df_map.loc[df_map['em_x_um'].isna(), '2p-ROI'],
))
df_out['matched'] = [key in matched_keys for key in zip(df_out['field'], df_out['roi_id'])]
df_out['not_a_cell'] = [key in not_a_cell_keys for key in zip(df_out['field'], df_out['roi_id'])]

print("\nPer-field z-plane fit:")
df_out['em_z_um'] = np.nan
for field, df_field in df_matched.groupby('field'):
    xy_matched = df_field[['roi_rotx_final_um', 'roi_roty_final_um']].to_numpy()
    z_matched = df_field['em_z_um'].to_numpy()

    roi_mask = df_out['field'] == field
    xy_all = df_out.loc[roi_mask, ['roi_rotx_final_um', 'roi_roty_final_um']].to_numpy()

    if len(df_field) < MIN_MATCHED_PER_FIELD:
        print(f"  field {field}: only {len(df_field)} matched cell(s), using mean z instead of a plane")
        df_out.loc[roi_mask, 'em_z_um'] = z_matched.mean()
        continue

    coef = fit_z_plane(xy_matched, z_matched)
    z_rmse = np.sqrt(np.mean((predict_z_plane(xy_matched, coef) - z_matched) ** 2))
    print(f"  field {field}: n={len(df_field)}, z-plane fit RMSE = {z_rmse:.2f} um")
    df_out.loc[roi_mask, 'em_z_um'] = predict_z_plane(xy_all, coef)

voxel_xyz = (
    df_out[['roi_rotx_final_um', 'roi_roty_final_um', 'em_z_um']].to_numpy()
    * 1000 / np.array([16, 16, 40])
)
voxel_xyz = np.round(voxel_xyz).astype(int)
df_out['Estimated EM Coords'] = [f"{x}, {y}, {z}" for x, y, z in voxel_xyz]

df_out = df_out.rename(columns={'field': '2p-Field', 'roi_id': '2p-ROI'})
df_out = df_out.merge(
    df_map[['2p-Field', '2p-ROI', 'Nuc Coords']], on=['2p-Field', '2p-ROI'], how='left',
)
df_out = df_out[['2p-Field', '2p-ROI', 'matched', 'not_a_cell', 'Nuc Coords', 'Estimated EM Coords']]

df_out.to_csv(OUT_FILE, index=False)
print(f"wrote {OUT_FILE}")

# %% [markdown]
# ## Neuroglancer link
#
# Per-field segments (Latest NucID), estimated<->real coordinate lines
# for matched ROIs, and estimated-coordinate points for unmatched ROIs.

# %%
client = CAVEclient(datastack_name=DATASTACK_NAME)
link = neuroglancer.spawn_field_mapping_link(client, df_map, df_out)

link_path = os.path.join(HERE, LINK_FILE)
with open(link_path, "w") as f:
    f.write(link + "\n")
print(f"wrote {link_path}")

# %% [markdown]
# ## EM -> 2p (inverse direction)
#
# `fit_registration` fits `2p_to_em` and `em_to_2p` independently by least
# squares (see `registration._fit_direction`), rather than algebraically
# inverting one to get the other. So `reg['directions']['em_to_2p']` is
# already a directly-fit global + per-field mapping for this direction --
# reuse it as-is (no refit) to map every EM soma with a known 2p field into
# 2p space, for comparison against the true 2p ROI positions.

# %%
g_em_to_2p = reg['directions']['em_to_2p']['global']
print(f"EM->2p global: angle = {g_em_to_2p['angle_deg']:.2f} deg, scale = {g_em_to_2p['scale']:.3f}, "
      f"RMSE = {g_em_to_2p['rmse_um']:.2f} um (n={g_em_to_2p['n_matched']})")

matched_2p_from_em = map_coords_per_row(matched_em, matched_fields, reg, direction='em_to_2p')
rmse_em_to_2p = np.sqrt(np.mean(np.sum((matched_2p_from_em - matched_2p) ** 2, axis=1)))
print(f"EM->2p RMSE on matched cells (global + per-field): {rmse_em_to_2p:.2f} um")

has_em_coords = df_map['em_x_um'].notna()
df_map_valid = df_map[has_em_coords]
mapped_em_to_2p = map_coords_per_row(
    df_map_valid[['em_x_um', 'em_y_um']].to_numpy(),
    df_map_valid['2p-Field'].to_numpy(),
    reg, direction='em_to_2p',
)
df_map['em_rotx_2p_um'] = np.nan
df_map['em_roty_2p_um'] = np.nan
df_map.loc[has_em_coords, ['em_rotx_2p_um', 'em_roty_2p_um']] = mapped_em_to_2p

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"aspect": "equal"})
sns.scatterplot(data=df_rois, x='temporal_nasal_pos_um', y='ventral_dorsal_pos_um', hue='field', alpha=0.5, ax=axs[0])
axs[0].set_title('2p coordinates (reference)')
sns.scatterplot(data=df_map[has_em_coords], x='em_rotx_2p_um', y='em_roty_2p_um', hue='2p-Field', alpha=0.5, ax=axs[1])
axs[1].set_title(f'EM coordinates, mapped to 2p space\n(global + per-field, EM->2p fit)\nmatched-cell RMSE={rmse_em_to_2p:.1f} um')
plt.tight_layout()
savefig("09_em_to_2p_alignment_overview.pdf")
