"""
Fits the rotation angle and an isotropic scale that best map 2p ROI
coordinates onto EM soma coordinates, using the known EM-cell <-> 2p-ROI
correspondences in the EM-2p mapping table (instead of just guessing the
angle). Flips are kept fixed since we already know they happen and why.

Saves figures to ./figures/ next to this script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from scipy.optimize import minimize

from eyewire2_functional_analysis import data_loader, neuroglancer

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

data_root = os.path.join(HERE, "..", "..", "data")
data_folder = f"{data_root}/data-2p"
morph_folder = f"{data_root}/morphological-data"

MAP_FILE = "Eyewire II Proofread Cells Main List - EM-2p-mapping 2026-07-08e v2-final.csv"
OUT_FILE = "2p_roi_estimated_em_coordinates.csv"
LINK_FILE = "neuroglancer_link.txt"

DATASTACK_NAME = "stroeh_mouse_retina"

DEG = -125
SCALE = 1.0
FLIP_X = True
FLIP_Y = False


def rotate_and_flip_coordinates(coords, angle_deg=-125, scale=1.0, flip_x=True, flip_y=False, center=None, new_center=None):
    """
    Rotate, scale and flip EM coordinates to match 2P coordinates.

    Parameters:
    coords (np.ndarray): Array of EM coordinates.
    angle_deg (float): Angle in degrees to rotate the coordinates.
    scale (float): Isotropic scale factor applied to the centered coordinates.
    center (np.ndarray or None): Point to center `coords` on before rotating/scaling.
        Defaults to the mean of `coords`. Pass an explicit value to re-apply a fit
        obtained on a different (e.g. matched) subset of points.

    Returns:
    np.ndarray: Rotated, scaled and flipped coordinates.
    """
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])

    if center is None:
        center = np.mean(coords, axis=0)
    coords_centered = (coords - center) * scale

    rotated_coords = coords_centered @ rotation_matrix.T

    if flip_x:
        rotated_coords[:, 0] *= -1
    if flip_y:
        rotated_coords[:, 1] *= -1

    if new_center is not None:
        rotated_coords += new_center

    return rotated_coords


def align_em_to_2p(coords, center=None, new_center=None):
    return rotate_and_flip_coordinates(coords, angle_deg=DEG, scale=SCALE, flip_x=FLIP_X, flip_y=FLIP_Y, center=center, new_center=new_center)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


def main():
    global DEG, SCALE

    df_rois, df_fields, df_outline = data_loader.load_all_dfs(data_folder)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})
    sns.scatterplot(data=df_rois, x='temporal_nasal_pos_um', y='ventral_dorsal_pos_um', hue='field', alpha=0.5, ax=ax)
    ax.set_title('2p ROI coordinates')
    savefig("01_2p_rois.png")

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
    savefig("02_em_somas.png")

    coords_em = df_map[['em_x_um', 'em_y_um']].values
    coords_2p = df_rois[['temporal_nasal_pos_um', 'ventral_dorsal_pos_um']].values
    center_em = np.mean(coords_em, axis=0)
    center_2p = np.mean(coords_2p, axis=0)

    # Naive alignment with the guessed DEG, no scaling, just for comparison.
    df_map[['em_rotx_um', 'em_roty_um']] = align_em_to_2p(coords_em, new_center=center_2p)
    df_rois[['roi_rotx_um', 'roi_roty_um']] = align_em_to_2p(coords_2p, new_center=center_em)

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
    savefig("03_naive_alignment.png")

    # --- Optimize rotation and scaling to map 2p into EM space -------------
    #
    # df_map gives, for each EM cell, the exact 2p field + ROI it corresponds
    # to ('2p-Field'/'2p-ROI' <-> df_rois' 'field'/'roi_id'). Instead of
    # aligning the two point clouds blindly, use these known correspondences
    # to fit the rotation angle and an isotropic scale by least squares
    # (flips are kept fixed; translation is not a free parameter: for a fixed
    # rotation/scale the optimal translation is the one that matches the
    # centroids, so both point sets are centered on their own centroid and
    # translated to the target centroid afterwards). EM is kept as the
    # reference frame and the 2p ROI coordinates are mapped onto it, since
    # downstream analyses key off the EM/morphological coordinate space.
    df_matched = df_map.merge(
        df_rois[['field', 'roi_id', 'temporal_nasal_pos_um', 'ventral_dorsal_pos_um']],
        left_on=['2p-Field', '2p-ROI'],
        right_on=['field', 'roi_id'],
        how='inner',
    )
    n_not_a_cell = df_matched['em_x_um'].isna().sum()
    if n_not_a_cell:
        print(f"Excluding {n_not_a_cell} ROI(s) with no EM coordinate (not a real cell) from the fit")
        df_matched = df_matched[df_matched['em_x_um'].notna()].reset_index(drop=True)
    print(f"Matched {len(df_matched)} / {len(df_map)} EM cells to a 2p ROI")

    matched_em = df_matched[['em_x_um', 'em_y_um']].to_numpy()
    matched_2p = df_matched[['temporal_nasal_pos_um', 'ventral_dorsal_pos_um']].to_numpy()

    center_em_matched = np.mean(matched_em, axis=0)
    center_2p_matched = np.mean(matched_2p, axis=0)

    def registration_sq_error(params):
        angle_deg, scale = params
        transformed = rotate_and_flip_coordinates(
            matched_2p, angle_deg=angle_deg, scale=scale, flip_x=FLIP_X, flip_y=FLIP_Y,
            center=center_2p_matched, new_center=center_em_matched,
        )
        return np.sum((transformed - matched_em) ** 2)

    x0 = [DEG, 1.0]
    result = minimize(registration_sq_error, x0=x0, method='Nelder-Mead')
    deg_opt, scale_opt = result.x

    rmse_initial = np.sqrt(registration_sq_error(x0) / len(matched_2p))
    rmse_optimized = np.sqrt(result.fun / len(matched_2p))

    print(result.message)
    print(f"Initial guess: angle = {x0[0]:.2f} deg, scale = {x0[1]:.3f}  ->  RMSE = {rmse_initial:.2f} um")
    print(f"Optimized:     angle = {deg_opt:.2f} deg, scale = {scale_opt:.3f}  ->  RMSE = {rmse_optimized:.2f} um")

    DEG, SCALE = deg_opt, scale_opt

    df_rois[['roi_rotx_opt_um', 'roi_roty_opt_um']] = rotate_and_flip_coordinates(
        coords_2p, angle_deg=DEG, scale=SCALE, flip_x=FLIP_X, flip_y=FLIP_Y,
        center=center_2p_matched, new_center=center_em_matched,
    )
    df_matched[['roi_rotx_opt_um', 'roi_roty_opt_um']] = rotate_and_flip_coordinates(
        matched_2p, angle_deg=DEG, scale=SCALE, flip_x=FLIP_X, flip_y=FLIP_Y,
        center=center_2p_matched, new_center=center_em_matched,
    )

    initial_matched = rotate_and_flip_coordinates(
        matched_2p, angle_deg=x0[0], scale=x0[1], flip_x=FLIP_X, flip_y=FLIP_Y,
        center=center_2p_matched, new_center=center_em_matched,
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"aspect": "equal"})

    axs[0].scatter(*initial_matched.T, c='C1', alpha=0.5, label='2p aligned (initial)')
    axs[0].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
    for p2_pt, em_pt in zip(initial_matched, matched_em):
        axs[0].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
    axs[0].set_title(f"Initial guess (angle={x0[0]:.1f}, scale={x0[1]:.2f})\nRMSE={rmse_initial:.1f} um")
    axs[0].legend(loc='upper right', fontsize='small')

    opt_matched = df_matched[['roi_rotx_opt_um', 'roi_roty_opt_um']].to_numpy()
    axs[1].scatter(*opt_matched.T, c='C1', alpha=0.5, label='2p aligned (optimized)')
    axs[1].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
    for p2_pt, em_pt in zip(opt_matched, matched_em):
        axs[1].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
    axs[1].set_title(f"Optimized (angle={deg_opt:.1f}, scale={scale_opt:.2f})\nRMSE={rmse_optimized:.1f} um")
    axs[1].legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    savefig("04_initial_vs_optimized_matched_pairs.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"aspect": "equal"})
    sns.scatterplot(data=df_map, x='em_x_um', y='em_y_um', hue='2p-Field', alpha=0.5, ax=axs[0])
    axs[0].set_title('EM coordinates (reference)')
    sns.scatterplot(data=df_rois, x='roi_rotx_opt_um', y='roi_roty_opt_um', hue='field', alpha=0.5, ax=axs[1])
    axs[1].set_title('2p coordinates, optimized alignment')
    plt.tight_layout()
    savefig("05_optimized_alignment_overview.png")

    # --- Per-field refinement (shift + rotation + scale) --------------------
    #
    # The global similarity transform above captures the overall rotation and
    # scale between the two coordinate systems, but the residual RMSE stays
    # high because the true 2p<->EM deformation isn't a single rigid map (it
    # varies smoothly across the retina, e.g. from local tissue stretch, so a
    # single global scale doesn't capture per-field differences in stretch
    # either). As a pragmatic correction, refine the globally aligned 2p
    # coordinates with an additional per-field rotation + isotropic scale +
    # translation, fit on the matched pairs within that field only and
    # applied to every ROI in that field.
    MIN_MATCHED_PER_FIELD = 3

    def fit_rotation_scale(source, target):
        """Fit the rotation and isotropic scale that best map `source` onto `target`."""
        center_source = np.mean(source, axis=0)
        center_target = np.mean(target, axis=0)

        def sq_error(params):
            angle_deg, scale = params
            transformed = rotate_and_flip_coordinates(
                source, angle_deg=angle_deg, scale=scale, flip_x=False, flip_y=False,
                center=center_source, new_center=center_target,
            )
            return np.sum((transformed - target) ** 2)

        res = minimize(sq_error, x0=[0.0, 1.0], method='Nelder-Mead')
        angle_deg, scale = res.x
        return angle_deg, scale, center_source, center_target, res.fun

    df_rois['roi_rotx_final_um'] = df_rois['roi_rotx_opt_um']
    df_rois['roi_roty_final_um'] = df_rois['roi_roty_opt_um']
    df_matched['roi_rotx_final_um'] = df_matched['roi_rotx_opt_um']
    df_matched['roi_roty_final_um'] = df_matched['roi_roty_opt_um']

    print("\nPer-field rigid refinement:")
    field_diagnostics = []
    for field, df_field in df_matched.groupby('field'):
        n = len(df_field)
        field_em = df_field[['em_x_um', 'em_y_um']].to_numpy()
        field_2p = df_field[['roi_rotx_opt_um', 'roi_roty_opt_um']].to_numpy()
        rmse_before = np.sqrt(np.mean(np.sum((field_2p - field_em) ** 2, axis=1)))

        if n < MIN_MATCHED_PER_FIELD:
            print(f"  field {field}: only {n} matched cell(s), skipping local fit (RMSE={rmse_before:.2f} um)")
            field_diagnostics.append({'field': field, 'n': n, 'rmse_before': rmse_before, 'rmse_after': rmse_before})
            continue

        angle_deg, scale, center_source, center_target, sq_err = fit_rotation_scale(field_2p, field_em)
        rmse_after = np.sqrt(sq_err / n)
        print(f"  field {field}: n={n}, angle={angle_deg:.2f} deg, scale={scale:.3f}  ->  RMSE {rmse_before:.2f} -> {rmse_after:.2f} um")
        field_diagnostics.append({'field': field, 'n': n, 'rmse_before': rmse_before, 'rmse_after': rmse_after})

        roi_mask = df_rois['field'] == field
        df_rois.loc[roi_mask, ['roi_rotx_final_um', 'roi_roty_final_um']] = rotate_and_flip_coordinates(
            df_rois.loc[roi_mask, ['roi_rotx_opt_um', 'roi_roty_opt_um']].to_numpy(),
            angle_deg=angle_deg, scale=scale, flip_x=False, flip_y=False,
            center=center_source, new_center=center_target,
        )
        matched_mask = df_matched['field'] == field
        df_matched.loc[matched_mask, ['roi_rotx_final_um', 'roi_roty_final_um']] = rotate_and_flip_coordinates(
            field_2p, angle_deg=angle_deg, scale=scale, flip_x=False, flip_y=False,
            center=center_source, new_center=center_target,
        )

    rmse_global = np.sqrt(np.mean(np.sum(
        (df_matched[['roi_rotx_opt_um', 'roi_roty_opt_um']].to_numpy() - matched_em) ** 2, axis=1)))
    rmse_final = np.sqrt(np.mean(np.sum(
        (df_matched[['roi_rotx_final_um', 'roi_roty_final_um']].to_numpy() - matched_em) ** 2, axis=1)))
    print(f"\nOverall RMSE: global={rmse_global:.2f} um, global+per-field={rmse_final:.2f} um")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"aspect": "equal"})
    sns.scatterplot(data=df_map, x='em_x_um', y='em_y_um', hue='2p-Field', alpha=0.5, ax=axs[0])
    axs[0].set_title('EM coordinates (reference)')
    sns.scatterplot(data=df_rois, x='roi_rotx_final_um', y='roi_roty_final_um', hue='field', alpha=0.5, ax=axs[1])
    axs[1].set_title('2p coordinates, global + per-field alignment')
    plt.tight_layout()
    savefig("06_per_field_alignment_overview.png")

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
    savefig("07_per_field_rmse.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"aspect": "equal"})

    global_matched = df_matched[['roi_rotx_opt_um', 'roi_roty_opt_um']].to_numpy()
    axs[0].scatter(*global_matched.T, c='C1', alpha=0.5, label='2p aligned (global)')
    axs[0].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
    for p2_pt, em_pt in zip(global_matched, matched_em):
        axs[0].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
    axs[0].set_title(f"Global alignment\nRMSE={rmse_global:.1f} um")
    axs[0].legend(loc='upper right', fontsize='small')

    final_matched = df_matched[['roi_rotx_final_um', 'roi_roty_final_um']].to_numpy()
    axs[1].scatter(*final_matched.T, c='C1', alpha=0.5, label='2p aligned (global + per-field)')
    axs[1].scatter(*matched_em.T, c='C2', alpha=0.5, label='EM (matched)')
    for p2_pt, em_pt in zip(final_matched, matched_em):
        axs[1].plot([p2_pt[0], em_pt[0]], [p2_pt[1], em_pt[1]], c='gray', lw=0.5)
    axs[1].set_title(f"Global + per-field alignment\nRMSE={rmse_final:.1f} um")
    axs[1].legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    savefig("08_global_vs_per_field_matched_pairs.png")

    # --- Estimated EM-space coordinates for every 2p ROI --------------------
    #
    # Combine the aligned x/y (global + per-field) with a z estimate. Rather
    # than a single per-field mean z (too coarse: z drifts smoothly across a
    # field, same as x/y did), fit a 2D linear plane z ~= a*x + b*y + c per
    # field on that field's matched (x, y, z) triples and use the plane to
    # predict z for every ROI in the field. Write everything out as a single
    # "x, y, z" EM *voxel* coordinate column, ready to paste into Neuroglancer
    # to go find the still-unmatched cells.
    def fit_z_plane(xy, z):
        """Fit z ~= a*x + b*y + c by least squares; returns coefficients [a, b, c]."""
        design = np.column_stack([xy, np.ones(len(xy))])
        coef, *_ = np.linalg.lstsq(design, z, rcond=None)
        return coef

    def predict_z_plane(xy, coef):
        design = np.column_stack([xy, np.ones(len(xy))])
        return design @ coef

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

    # --- Neuroglancer link ---------------------------------------------------
    #
    # Per-field segments (Latest NucID), estimated<->real coordinate lines
    # for matched ROIs, and estimated-coordinate points for unmatched ROIs.
    client = CAVEclient(datastack_name=DATASTACK_NAME)
    link = neuroglancer.spawn_field_mapping_link(client, df_map, df_out)

    link_path = os.path.join(HERE, LINK_FILE)
    with open(link_path, "w") as f:
        f.write(link + "\n")
    print(f"wrote {link_path}")


if __name__ == "__main__":
    main()
