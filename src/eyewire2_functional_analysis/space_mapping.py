"""Helpers for moving coordinates/skeletons between the EM and 2p reference frames.

These build on the low-level fitted-registration primitives in
`eyewire2_functional_analysis.registration` (`map_coords`, `align_skel`), but
operate on whole batches (grouped by field) or on a skeleton's absolute
position rather than just its orientation -- shared by
`scripts/preprocessing/em-2p-mapping.py` and
`scripts/tutorial/plot_retinal_outline/plot_retinal_outline.py`.
"""
import numpy as np

from eyewire2_functional_analysis import registration


def map_coords_per_row(coords, fields, reg, direction):
    """Apply `registration.map_coords` to `coords`, grouped by `fields` (so each
    row gets its own field's per-field refinement, falling back to the global
    fit -- with a warning -- for fields that have none)."""
    coords = np.asarray(coords, dtype=np.float64)
    fields = np.asarray(fields)
    out = np.empty_like(coords)
    for field in np.unique(fields):
        mask = fields == field
        out[mask] = registration.map_coords(coords[mask], reg, direction=direction, field=field)
    return out


def fit_z_plane(xy, z):
    """Fit z ~= a*x + b*y + c by least squares; returns coefficients [a, b, c]."""
    design = np.column_stack([xy, np.ones(len(xy))])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    return coef


def predict_z_plane(xy, coef):
    """Evaluate the z ~= a*x + b*y + c plane fit returned by `fit_z_plane` at `xy`."""
    design = np.column_stack([xy, np.ones(len(xy))])
    return design @ coef


def align_and_place_skel(skel, reg, field, target_xy, direction='em_to_2p'):
    """Rotate `skel` into the target reference frame about its own soma centre
    (see `registration.align_skel`), then translate it so that centre lands
    exactly on `target_xy` -- `align_skel` only rotates/flips, it doesn't move
    the skeleton's absolute position into the target frame.

    Args:
        skel: skeliner.core.Skeleton to place.
        reg: Fitted 2p<->EM registration dict.
        field: 2p field name used to look up the per-field rotation.
        target_xy: ``(x, y)`` position, in the target frame, to place the
            skeleton's soma at.
        direction: '2p_to_em' or 'em_to_2p' (default), forwarded to
            `registration.align_skel`.

    Returns:
        skeliner.core.Skeleton: The rotated + translated skeleton (a copy;
        `skel` is untouched).
    """
    skel = registration.align_skel(skel, reg, field=field, direction=direction)
    dx = target_xy[0] - skel.soma.center[0]
    dy = target_xy[1] - skel.soma.center[1]
    skel.nodes[:, 0] += dx
    skel.nodes[:, 1] += dy
    skel.soma.center[0] += dx
    skel.soma.center[1] += dy
    return skel
