"""2p <-> EM coordinate/orientation registration.

Fits (and persists) the similarity transform (rotation + isotropic scale,
with a fixed mirror flip) that best maps 2p ROI coordinates onto EM soma
coordinates, using the known EM-cell <-> 2p-ROI correspondences in the EM-2p
mapping table -- see scripts/preprocessing/em-2p-mapping.py, which fits and
saves the parameters this module loads.

The fit is bidirectional ('2p_to_em' and 'em_to_2p' are fit independently by
least squares, not algebraically inverted from one another) and per-field: a
global similarity transform captures the overall rotation/scale, and an
additional per-field rotation+scale refinement corrects for local tissue
stretch that a single global transform can't capture. Fields with too few
matched cells to refine fall back to the global-only transform.
"""
import os
import warnings

import numpy as np
import yaml
from scipy.optimize import minimize


def rotate_and_flip_coordinates(coords, angle_deg=0.0, scale=1.0, flip_x=False, flip_y=False, center=None, new_center=None):
    """Rotate, scale and flip 2-D coordinates.

    Args:
        coords: Array of shape (n, 2).
        angle_deg: Rotation angle in degrees.
        scale: Isotropic scale factor applied to the centered coordinates.
        flip_x: If True, negate x after rotating.
        flip_y: If True, negate y after rotating.
        center: Point to center `coords` on before rotating/scaling. Defaults
            to the mean of `coords`.
        new_center: Point to re-center the result on. Defaults to no shift.

    Returns:
        np.ndarray: Transformed coordinates, shape (n, 2).
    """
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])

    coords = np.asarray(coords, dtype=np.float64)
    if center is None:
        center = np.mean(coords, axis=0)
    coords_centered = (coords - np.asarray(center)) * scale

    rotated_coords = coords_centered @ rotation_matrix.T

    if flip_x:
        rotated_coords[:, 0] *= -1
    if flip_y:
        rotated_coords[:, 1] *= -1

    if new_center is not None:
        rotated_coords = rotated_coords + np.asarray(new_center)

    return rotated_coords


def similarity_matrix(angle_deg, flip_x=False, flip_y=False):
    """Column-vector 2x2 matrix combining a rotation and a mirror flip.

    ``similarity_matrix(...) @ point`` rotates ``point`` by ``angle_deg`` and
    then mirrors it, matching the point convention used by
    :func:`rotate_and_flip_coordinates` (rotate, then flip; no scale/translation).
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    F = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0])
    return F @ R


def fit_rotation_scale(source, target, flip_x=False, flip_y=False, x0_angle=0.0):
    """Fit the rotation, scale (and optional fixed flip) that best maps `source` onto `target`.

    Args:
        source: Array of shape (n, 2).
        target: Array of shape (n, 2), paired with `source`.
        flip_x: Fixed (not fitted) mirror flip applied after rotation.
        flip_y: Fixed (not fitted) mirror flip applied after rotation.
        x0_angle: Initial guess for the rotation angle (degrees).

    Returns:
        dict: {'angle_deg', 'scale', 'center_source', 'center_target', 'rmse_um', 'n_matched'}.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    center_source = np.mean(source, axis=0)
    center_target = np.mean(target, axis=0)

    def sq_error(params):
        angle_deg, scale = params
        transformed = rotate_and_flip_coordinates(
            source, angle_deg=angle_deg, scale=scale, flip_x=flip_x, flip_y=flip_y,
            center=center_source, new_center=center_target,
        )
        return np.sum((transformed - target) ** 2)

    result = minimize(sq_error, x0=[x0_angle, 1.0], method='Nelder-Mead')
    angle_deg, scale = result.x
    rmse = float(np.sqrt(result.fun / len(source)))

    return {
        'angle_deg': float(angle_deg),
        'scale': float(scale),
        'center_source': [float(v) for v in center_source],
        'center_target': [float(v) for v in center_target],
        'rmse_um': rmse,
        'n_matched': int(len(source)),
    }


def _fit_direction(source, target, fields, flip_x, flip_y, min_matched_per_field, x0_angle):
    """Fit a global + per-field registration mapping `source` onto `target`."""
    global_fit = fit_rotation_scale(source, target, flip_x=flip_x, flip_y=flip_y, x0_angle=x0_angle)

    aligned_source = rotate_and_flip_coordinates(
        source, angle_deg=global_fit['angle_deg'], scale=global_fit['scale'], flip_x=flip_x, flip_y=flip_y,
        center=global_fit['center_source'], new_center=global_fit['center_target'],
    )

    field_fits = {}
    fields = np.asarray(fields)
    for field in sorted(set(fields.tolist())):
        mask = fields == field
        n = int(mask.sum())
        if n < min_matched_per_field:
            continue
        field_fits[str(field)] = fit_rotation_scale(aligned_source[mask], target[mask])

    return {'global': global_fit, 'fields': field_fits}


def fit_registration(df_map, df_rois, flip_x=True, flip_y=False, min_matched_per_field=3, x0_angle_2p_to_em=-125.0):
    """Fit the bidirectional 2p<->EM coordinate registration.

    Args:
        df_map: EM-2p mapping DataFrame (as loaded from the mapping CSV) with
            '2p-Field', '2p-ROI', 'em_x_um', 'em_y_um' columns.
        df_rois: ROI-level DataFrame with 'field', 'roi_id',
            'temporal_nasal_pos_um', 'ventral_dorsal_pos_um' columns.
        flip_x: Fixed mirror flip (shared by both directions) applied after rotation.
        flip_y: Fixed mirror flip (shared by both directions) applied after rotation.
        min_matched_per_field: Minimum matched cells for a field to get its own
            per-field refinement; fields below this fall back to the global fit.
        x0_angle_2p_to_em: Initial guess for the 2p->EM global rotation angle
            (degrees); the EM->2p fit is initialized at its negative.

    Returns:
        dict: Registration parameters, with 'meta' and 'directions'
        ('2p_to_em' / 'em_to_2p', each with 'global' and 'fields') keys.
        Suitable for :func:`save_registration` / :func:`map_coords` /
        :func:`get_field_rotation_matrix`.
    """
    df_matched = df_map.merge(
        df_rois[['field', 'roi_id', 'temporal_nasal_pos_um', 'ventral_dorsal_pos_um']],
        left_on=['2p-Field', '2p-ROI'], right_on=['field', 'roi_id'], how='inner',
    )
    df_matched = df_matched[df_matched['em_x_um'].notna()].reset_index(drop=True)

    em = df_matched[['em_x_um', 'em_y_um']].to_numpy()
    tp = df_matched[['temporal_nasal_pos_um', 'ventral_dorsal_pos_um']].to_numpy()
    fields = df_matched['field'].to_numpy()

    directions = {
        '2p_to_em': _fit_direction(tp, em, fields, flip_x, flip_y, min_matched_per_field, x0_angle_2p_to_em),
        'em_to_2p': _fit_direction(em, tp, fields, flip_x, flip_y, min_matched_per_field, -x0_angle_2p_to_em),
    }

    return {
        'meta': {
            'flip_x': bool(flip_x),
            'flip_y': bool(flip_y),
            'min_matched_per_field': int(min_matched_per_field),
            'n_matched_total': int(len(df_matched)),
        },
        'directions': directions,
    }


def save_registration(reg, path):
    """Save a registration dict (as returned by :func:`fit_registration`) to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(reg, f, sort_keys=False)


def load_registration(path):
    """Load a registration dict previously written by :func:`save_registration`."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_or_fit_registration(path, df_map=None, df_rois=None, refit=False, **fit_kwargs):
    """Load the registration from `path` if it exists and `refit` is False; otherwise fit and save it.

    Args:
        path: YAML file path to load from / save to.
        df_map: EM-2p mapping DataFrame; required if a fit is actually needed
            (missing/stale file, or `refit=True`) -- see :func:`fit_registration`.
        df_rois: ROI-level DataFrame; required under the same conditions as `df_map`.
        refit: If True, always (re-)fit and overwrite `path`, even if it already exists.
        **fit_kwargs: Forwarded to :func:`fit_registration`.

    Returns:
        dict: The registration parameters.
    """
    if not refit and os.path.exists(path):
        return load_registration(path)

    if df_map is None or df_rois is None:
        raise ValueError(
            f"No registration found at {path} (or refit=True), but df_map/df_rois were not provided to fit one."
        )
    reg = fit_registration(df_map, df_rois, **fit_kwargs)
    save_registration(reg, path)
    return reg


def _get_field_fit(reg, direction, field):
    d = reg['directions'][direction]
    field_fit = d['fields'].get(field) if field is not None else None
    if field_fit is None and field is not None:
        warnings.warn(
            f"No per-field '{direction}' registration for field={field!r} "
            f"(fewer than {reg['meta']['min_matched_per_field']} matched cells); "
            "falling back to the global fit."
        )
    return d['global'], field_fit


def map_coords(coords, reg, direction, field=None):
    """Map 2-D coordinates from one space to the other using a fitted registration.

    Applies the global similarity transform, then the per-field refinement for
    `field` if one was fit (falls back to the global-only result, with a
    warning, if `field` has no per-field refinement or is ``None``).

    Args:
        coords: Array of shape (n, 2), in the `direction`'s source space
            ('2p_to_em' -> 2p (temporal_nasal_pos_um, ventral_dorsal_pos_um);
            'em_to_2p' -> EM (em_x_um, em_y_um)).
        reg: Registration dict, as returned by :func:`fit_registration`/:func:`load_registration`.
        direction: '2p_to_em' or 'em_to_2p'.
        field: 2p field name (e.g. 'GCL0'), used to look up the per-field refinement.

    Returns:
        np.ndarray: Transformed coordinates, shape (n, 2).
    """
    global_fit, field_fit = _get_field_fit(reg, direction, field)
    flip_x, flip_y = reg['meta']['flip_x'], reg['meta']['flip_y']

    out = rotate_and_flip_coordinates(
        coords, angle_deg=global_fit['angle_deg'], scale=global_fit['scale'], flip_x=flip_x, flip_y=flip_y,
        center=global_fit['center_source'], new_center=global_fit['center_target'],
    )
    if field_fit is None:
        return out

    return rotate_and_flip_coordinates(
        out, angle_deg=field_fit['angle_deg'], scale=field_fit['scale'], flip_x=False, flip_y=False,
        center=field_fit['center_source'], new_center=field_fit['center_target'],
    )


def get_field_rotation_matrix(reg, direction, field=None):
    """Return the combined rotation+flip (no scale/translation) 2x2 matrix for `field`.

    Intended for re-orienting an already-correctly-scaled object (e.g. an EM
    skeleton) rather than mapping point coordinates between the two spaces --
    see :func:`map_coords` for that. Column-vector convention: ``matrix @ point``.

    Args:
        reg: Registration dict.
        direction: '2p_to_em' or 'em_to_2p'.
        field: 2p field name used to look up the per-field refinement (falls
            back to the global-only rotation, with a warning, if missing).

    Returns:
        np.ndarray: 2x2 matrix.
    """
    global_fit, field_fit = _get_field_fit(reg, direction, field)
    flip_x, flip_y = reg['meta']['flip_x'], reg['meta']['flip_y']

    M = similarity_matrix(global_fit['angle_deg'], flip_x, flip_y)
    if field_fit is None:
        return M

    M_field = similarity_matrix(field_fit['angle_deg'], False, False)
    return M_field @ M


def align_skel(skel, reg, field, direction='em_to_2p'):
    """Rotate/flip an EM skeleton's XY coordinates into the 2p (or EM) reference frame.

    Uses only the rotation+flip part of the fitted registration (no scale, no
    translation): the skeleton is already in true physical µm units, so only
    its orientation needs correcting, not its size or position. See
    :func:`get_field_rotation_matrix` for the fallback behaviour when `field`
    has no per-field refinement.

    Args:
        skel: skeliner.core.Skeleton to rotate.
        reg: Registration dict.
        field: 2p field name (e.g. ``row['field']``) used to look up the
            per-field rotation refinement.
        direction: '2p_to_em' or 'em_to_2p' (default: 'em_to_2p', i.e. rotate
            an EM skeleton into the 2p/retinal reference frame for display).

    Returns:
        skeliner.core.Skeleton: The rotated skeleton (a copy; `skel` is untouched).
    """
    from eyewire2_functional_analysis.skeleton import transform_skel_xy

    matrix = get_field_rotation_matrix(reg, direction=direction, field=field)
    return transform_skel_xy(skel, matrix)
