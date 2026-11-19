"""Per-field alignment corrections between the 2p frame, the visual stimulus and the EM reconstructions.

Estimated in ``scripts/analysis/rf_offsets/rf_offsets.py`` and stored in ``data/registration/rf_alignment.yaml``:

- **Stimulus shift** (``stim_shift_um``): where the stimulus was relative to where it is assumed to be (centered on the
  recording field), per field, in um in the retinal frame (x = nasal, y = dorsal). Measured with the moving bar: the
  mean offset from the 2p ROI centers to the RF centers estimated from the bar response timing, over all ROIs of the
  field with a reliable bar RF. RF positions are corrected by subtracting it (``correct_rf_xy``).
- **EM affine** (``em_affine``): the 2x2 linear part of the per-field affine map from the EM nucleus positions to the 2p
  ROI positions (all matched cells of the field), relative to the registration rotation. It captures the anisotropic
  stretch / shear between EM and 2p that the rotation-only skeleton alignment (``registration.align_skel``) misses, and
  is applied to the already rotated skeleton about its soma (``space_mapping.align_and_place_skel``).

Both corrections are applied by default by ``plot_rf`` and ``space_mapping.align_and_place_skel`` (``align=True`` /
``em_affine=True``). If the file or a field is missing, the corrections are the identity (with a warning).
"""
import functools
import os
import warnings

import numpy as np
import yaml

from eyewire2_functional_analysis import paths

ALIGNMENT_FILE = os.path.join(paths.PROJECT_ROOT, 'data', 'registration', 'rf_alignment.yaml')


@functools.lru_cache(maxsize=4)
def load_alignment(path: str = ALIGNMENT_FILE) -> dict:
    """The alignment file as a dict (``meta`` and ``fields``), or an empty one (with a warning) if it is missing."""
    if not os.path.isfile(path):
        warnings.warn(f'No RF alignment file {path}; no alignment corrections are applied. '
                      'Run scripts/analysis/rf_offsets/rf_offsets.py to create it.')
        return {'meta': {}, 'fields': {}}
    with open(path) as f:
        return yaml.safe_load(f)


def save_alignment(alignment: dict, path: str = ALIGNMENT_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(alignment, f, sort_keys=False)
    load_alignment.cache_clear()


def _field(field: str, path: str) -> dict | None:
    fields = load_alignment(path)['fields']
    if field not in fields:
        if fields:
            warnings.warn(f'No RF alignment for field {field!r}; no correction applied.')
        return None
    return fields[field]


def stim_shift(field: str, path: str = ALIGNMENT_FILE) -> np.ndarray:
    """Stimulus shift ``(dx, dy)`` in um (retinal frame) of ``field``; zeros if unknown."""
    f = _field(field, path)
    return np.zeros(2) if f is None else np.asarray(f['stim_shift_um'], float)


def em_affine(field: str, path: str = ALIGNMENT_FILE) -> np.ndarray:
    """2x2 EM -> 2p stretch / shear of ``field`` (column-vector convention, applied after the registration rotation);
    the identity if unknown."""
    f = _field(field, path)
    return np.eye(2) if f is None else np.asarray(f['em_affine'], float)


def correct_rf_xy(x, y, field: str, path: str = ALIGNMENT_FILE):
    """RF position(s) ``x, y`` (um, retinal frame) corrected for the stimulus shift of ``field``."""
    dx, dy = stim_shift(field, path)
    return np.asarray(x) - dx, np.asarray(y) - dy
