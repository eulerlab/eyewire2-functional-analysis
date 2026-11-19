"""Plot the pre-estimated mouse-cam receptive fields (``df_eyewire2_roi_level_mc_rf.parquet``, see
``scripts/preprocessing/mc_rfs``) on top of the EM skeletons.

All plots are in the retinal frame of ``plot_morph`` (x = nasal, y = dorsal, in um) relative to the soma, which is placed
at the ROI position. The mouse-cam movie is centered on the field center (``field_temporal_nasal_pos_um``,
``field_ventral_dorsal_pos_um``). The orientation of the movie in the retinal frame is ``receptive_fields.movie_to_retina``;
it fits better than every other flip / rotation and is confirmed by the moving-bar RF centers
(``scripts/analysis/rf_offsets/REPORT.md``). ``row`` is a ROI row that has the ``mc_rf_*`` columns and the ROI / field
positions.

With ``align=True`` (default), RF positions are corrected for the per-field stimulus shift (``rf_alignment.stim_shift``,
measured with the moving bar); place the skeleton with ``space_mapping.align_and_place_skel`` (EM affine on by default)
to also correct the EM -> 2p stretch. ``align=False``: the stimulus as assumed (centered on the field).
"""
import matplotlib.pyplot as plt
import numpy as np

from eyewire2_functional_analysis import plot_morph, receptive_fields as rf, rf_alignment
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

CH_COLORS = {'green': '#2ca02c', 'uv': '#8e44ad'}
CH_LABELS = {'green': 'green', 'uv': 'UV'}
SRF_CMAP = 'RdBu_r'


def dog_params(row, ch: str) -> dict:
    """DoG fit parameters of channel ``ch`` (``mc_rf_{ch}_dog_*`` columns, see ``receptive_fields.fit_srfs``)."""
    prefix = f'mc_rf_{ch}_dog_'
    return {k.removeprefix(prefix): row[k] for k in row.index if k.startswith(prefix)}


def srf_retina(row, ch: str, align: bool = True) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """sRF of channel ``ch`` in the retinal frame and its extent ``(left, right, bottom, top)`` in um relative to the
    soma, to be shown with ``imshow(img, extent=extent, origin='lower')``. ``align``: correct for the stimulus shift of
    the field (see the module docstring)."""
    img = rf.movie_to_retina(row[f'mc_rf_{ch}_srf'])
    px = row.mc_rf_pixel_size_um
    cx = row.field_temporal_nasal_pos_um - row.temporal_nasal_pos_um  # stimulus center relative to the soma
    cy = row.field_ventral_dorsal_pos_um - row.ventral_dorsal_pos_um
    if align:
        cx, cy = rf_alignment.correct_rf_xy(cx, cy, row.field)
    ny, nx = img.shape
    return img, (cx - nx * px / 2, cx + nx * px / 2, cy - ny * px / 2, cy + ny * px / 2)


def movie_px_to_retina_um(row, x, y, shape, align: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Map movie pixel coordinates (``x`` = column, ``y`` = row of the sRF array of ``shape``) to the retinal frame (um
    relative to the soma), consistent with ``srf_retina`` (pixel centers)."""
    ny, nx = shape
    img_r = (nx - 1) - np.asarray(x)  # rows of the retinal image (see `movie_to_retina`: transpose, both axes reversed)
    img_c = (ny - 1) - np.asarray(y)
    _, (left, _, bottom, _) = srf_retina(row, 'green', align=align)
    px = row.mc_rf_pixel_size_um
    return left + (img_c + 0.5) * px, bottom + (img_r + 0.5) * px


def dog_ellipse(row, ch: str, which: str = 'center', n_sigma: float = 2., n: int = 200, align: bool = True):
    """``n_sigma`` ellipse (default 2: the ``exp(-2)`` contour of the unit Gaussian) of the DoG center or surround of
    channel ``ch`` in the retinal frame (um relative to the soma). Returns ``(xs, ys)``, or None if not fitted."""
    p = dog_params(row, ch)
    sx, sy = (p['sx'], p['sy']) if which == 'center' else (p['sx_s'], p['sy_s'])
    if not np.all(np.isfinite([p['x0'], p['y0'], sx, sy, p['theta']])):
        return None
    t = np.linspace(0, 2 * np.pi, n)
    u, v = n_sigma * sx * np.cos(t), n_sigma * sy * np.sin(t)
    c, s = np.cos(p['theta']), np.sin(p['theta'])
    # Principal axes of `receptive_fields._gauss2d` in (column, row) coordinates: sx along (cos, -sin), sy along (sin, cos)
    x, y = p['x0'] + c * u + s * v, p['y0'] - s * u + c * v
    return movie_px_to_retina_um(row, x, y, row[f'mc_rf_{ch}_srf'].shape, align=align)


def surround_volume(row, ch: str) -> float:
    """Volume of the DoG surround relative to the center, ``w k^2``."""
    p = dog_params(row, ch)
    return p['surround_rel_amp'] * p['surround_ratio'] ** 2


def draw_dog(ax, row, ch: str, color=None, lw: float = 1.2, min_surround_vol: float = 0.1, center_ls='-',
             halo: bool = True, zorder: int = 4, align: bool = True) -> None:
    """Draw the 2-sigma ellipses of the DoG center (``center_ls``) and, if its relative volume is at least
    ``min_surround_vol``, the surround (dotted) of channel ``ch``. ``halo`` adds a thin white outline for contrast."""
    color = color or CH_COLORS[ch]
    for which, ls in (('center', center_ls), ('surround', ':')):
        if which == 'surround' and not surround_volume(row, ch) >= min_surround_vol:
            continue
        xy = dog_ellipse(row, ch, which, align=align)
        if xy is None:
            continue
        if halo:
            ax.plot(*xy, c='w', lw=lw + 1.4, ls='-', alpha=0.8, zorder=zorder, solid_capstyle='round')
        ax.plot(*xy, c=color, lw=lw, ls=ls, zorder=zorder + 0.1)


def cell_radius(skel, margin: float = 20.) -> float:
    """Distance of the furthest non-axon node from the soma, plus ``margin``."""
    pts = skel.nodes[:, :2] - skel.soma.center[:2]
    if skel.ntype is not None and (skel.ntype != 2).any():
        pts = pts[skel.ntype != 2]
    return float(np.abs(pts).max()) + margin


def rf_radius(row, chs=mc.MC_CHANNELS, margin: float = 20., align: bool = True) -> float:
    """Half-width of a soma-centered square that contains the 2-sigma DoG center ellipses of ``chs``, plus ``margin``."""
    rads = [np.abs(np.concatenate(xy)).max() for ch in chs if (xy := dog_ellipse(row, ch, align=align)) is not None]
    return float(max(rads)) + margin if rads else 0.


def plot_srf_on_morph(ax, row, skel, ch: str, vabsmax: float | None = None, show_fit=True, fit_chs=None,
                      skel_color='k', skel_lw: float = 0.6, img_alpha: float = 0.9, interpolation='bicubic',
                      min_surround_vol: float = 0.1, align: bool = True):
    """sRF of channel ``ch`` with the skeleton (soma at the origin, see ``space_mapping.align_and_place_skel``) and the
    DoG ellipses of ``fit_chs`` (default: ``ch``) on top. Returns the image (e.g. for a colorbar)."""
    img, extent = srf_retina(row, ch, align=align)
    if vabsmax is None:
        vabsmax = np.nanmax(np.abs(img)) if np.any(np.isfinite(img)) else 1.
    im = ax.imshow(img, cmap=SRF_CMAP, vmin=-vabsmax, vmax=vabsmax, origin='lower', extent=extent,
                   interpolation=interpolation, alpha=img_alpha, zorder=0)
    ax.add_patch(plt.Rectangle(extent[::2], extent[1] - extent[0], extent[3] - extent[2], fill=False, ec='0.6',
                               lw=0.5, zorder=1))  # stimulus border
    plot_morph.plot_skel_lines(ax, skel, color=skel_color, lw=skel_lw, alpha=0.9)
    if show_fit:
        for c in fit_chs or [ch]:
            draw_dog(ax, row, c, min_surround_vol=min_surround_vol, align=align)
    ax.set(aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_trfs(ax, row, chs=mc.MC_CHANNELS, scale_by_srf: bool = True, lw: float = 1.2, labels: bool = True) -> None:
    """tRFs of ``chs`` (they carry the polarity of the DoG center), scaled by the sRF peak so that the channels are
    comparable."""
    t = row.mc_rf_trf_t0 + np.arange(row[f'mc_rf_{chs[0]}_trf'].size) * row.mc_rf_trf_dt
    ax.axhline(0, c='0.8', lw=0.6, zorder=0)
    ax.axvline(0, c='0.8', lw=0.6, ls='--', zorder=0)
    for ch in chs:
        scale = np.nanmax(np.abs(row[f'mc_rf_{ch}_srf'])) if scale_by_srf else 1.
        ax.plot(t, row[f'mc_rf_{ch}_trf'] * scale, c=CH_COLORS[ch], lw=lw, label=CH_LABELS[ch])
    ax.set(xlim=(t[0], t[-1]), yticks=[])
    if labels:
        ax.set_xlabel('time to response (s)')
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)


def add_scale_bar(ax, lim, size: float = 100., fontsize: float = 7, color='k') -> None:
    """Horizontal ``size`` um scale bar in the lower left of an axis with limits ``lim`` = ``(xmin, xmax, ymin, ymax)``."""
    w, h = lim[1] - lim[0], lim[3] - lim[2]
    x0, y0 = lim[0] + 0.06 * w, lim[2] + 0.06 * h
    ax.plot([x0, x0 + size], [y0, y0], c=color, lw=2, solid_capstyle='butt', zorder=6)
    ax.text(x0 + size / 2, y0 + 0.035 * h, f'{size:.0f} µm', ha='center', va='bottom', fontsize=fontsize, color=color,
            zorder=6)
