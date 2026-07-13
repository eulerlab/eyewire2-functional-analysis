import numpy as np
import skeliner as sk
from matplotlib import pyplot as plt
from matplotlib import patches as patches

from eyewire2_functional_analysis import registration
from eyewire2_functional_analysis.plot_traces import plot_bar_dir_grid
from eyewire2_functional_analysis.plot_utils import get_extent, plot_scale_bar
from eyewire2_functional_analysis.skeleton import rotate_skel


def plot_ds_on_morph(row, reg=None, rotation_deg=None, annotate_orientation=None, show_em_axes=None,
                      figsize=(10, 5), rad=200):
    """Plot a cell's morphology next to its moving-bar DS/OS response summary.

    Args:
        row: DataFrame row with a ``'skel'`` entry plus the moving-bar columns
            required by :func:`plot_bar_dir_grid`.
        reg: Optional fitted 2p<->EM registration dict, forwarded to
            :func:`plot_morph` to rotate/flip the skeleton into the
            2p/retinal reference frame. Takes precedence over `rotation_deg`.
        rotation_deg: Optional manual counterclockwise rotation (degrees)
            applied instead, when no fitted `reg` is available.
        annotate_orientation: Forwarded to :func:`plot_morph`; defaults to
            ``reg is not None``.
        show_em_axes: Forwarded to :func:`plot_morph` (EM X/Y axis indicator,
            see :func:`plot_em_axis_indicator`); defaults to ``reg is not None``.
        figsize: Figure size passed to ``plt.figure``.
        rad: Half-width in µm of the morphology axis limits around the soma.

    Returns:
        matplotlib.figure.Figure: The combined figure.
    """
    fig = plt.figure(figsize=figsize, facecolor='w')
    gs = fig.add_gridspec(1, 2, width_ratios=(1, 1.2))

    ax_morph = fig.add_subplot(gs[0, 0])
    plot_morph(ax=ax_morph, row=row, rad=rad, reg=reg, rotation_deg=rotation_deg,
               annotate_orientation=annotate_orientation, show_em_axes=show_em_axes)

    plot_bar_dir_grid(fig, gs[0, 1], row)

    plt.tight_layout()
    return fig


def annotate_retinal_axes(ax, fontsize=9, pad_pt=22, color='dimgray'):
    """Label the 4 edges of a `plot_morph` axes with the retinal directions.

    Only correct for a `plot_morph` axes whose skeleton was rotated into the
    2p/retinal reference frame via a fitted em_to_2p registration (i.e.
    ``plot_morph(..., reg=...)``). `plot_morph` plots with ``plane='xy'`` (not
    inverted), the same convention as
    `scripts/tutorial/plot_retinal_outline/plot_retinal_outline.py`; combined
    with 'ventral_dorsal_pos_um' increasing dorsally and
    'temporal_nasal_pos_um' increasing nasally (see
    ``data/stimuli/README.md``), the screen edges work out to
    right=nasal, left=temporal, top=dorsal, bottom=ventral.

    Args:
        ax: A `plot_morph` Axes.
        fontsize: Label font size.
        pad_pt: Distance from the axes edge to each label, in points (a fixed
            point offset clears the tick labels regardless of axes size,
            unlike an axes-fraction pad).
        color: Label colour.
    """
    kwargs = dict(xycoords='axes fraction', textcoords='offset points',
                  fontsize=fontsize, color=color, clip_on=False, annotation_clip=False)
    ax.annotate('Nasal', xy=(1, 0.5), xytext=(pad_pt, 0), ha='left', va='center', **kwargs)
    ax.annotate('Temporal', xy=(0, 0.5), xytext=(-pad_pt, 0), ha='right', va='center', **kwargs)
    ax.annotate('Dorsal', xy=(0.5, 1), xytext=(0, pad_pt), ha='center', va='bottom', **kwargs)
    ax.annotate('Ventral', xy=(0.5, 0), xytext=(0, -pad_pt), ha='center', va='top', **kwargs)


def plot_em_axis_indicator(ax, reg, field=None, direction='em_to_2p', center=None, scale=None,
                            autofit_frac=0.2, colors=('red', 'green'), labels=('EM X', 'EM Y'),
                            lw=2, fontsize=9, zorder=10):
    """Draw two orthogonal lines from a shared start point showing how EM's
    native X/Y axes are oriented once rotated into the target reference frame.

    Shared by `scripts/tutorial/plot_retinal_outline/plot_retinal_outline.py`
    and :func:`plot_morph`/:func:`plot_ds_on_morph`, so both scripts render
    this indicator identically.

    Args:
        ax: Matplotlib Axes to draw on.
        reg: Fitted 2p<->EM registration dict (see
            ``eyewire2_functional_analysis.registration``).
        field: 2p field name used to look up a per-field rotation refinement
            (falls back to the global-only rotation, with a warning, if that
            field has none). Defaults to ``None`` (global-only, no warning).
        direction: '2p_to_em' or 'em_to_2p' (default), forwarded to
            :func:`registration.get_field_rotation_matrix`.
        center: ``(x, y)`` shared start point for both lines. Defaults to the
            midpoint of `ax`'s current x/y limits.
        scale: Length of each line, in data units. Defaults (``None``) to
            `autofit_frac` times the smaller of `ax`'s current x/y span.
        autofit_frac: Fraction of the smaller axis span used to auto-size
            `scale` when it isn't given explicitly.
        colors: ``(x_color, y_color)``.
        labels: ``(x_label, y_label)`` text drawn at each line's end.
        lw: Line width.
        fontsize: Label font size.
        zorder: Drawing order (high, so the indicator sits above other artists).

    Returns:
        tuple: ``(x_end, y_end)`` -- the two line endpoints, in data coordinates.
    """
    rotation = registration.get_field_rotation_matrix(reg, direction=direction, field=field)
    x_dir = rotation @ np.array([1.0, 0.0])
    y_dir = rotation @ np.array([0.0, 1.0])

    if center is None:
        center = np.array([np.mean(ax.get_xlim()), np.mean(ax.get_ylim())])
    else:
        center = np.asarray(center, dtype=np.float64)

    if scale is None:
        xspan = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
        yspan = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        scale = autofit_frac * min(xspan, yspan)

    x_end = center + scale * x_dir
    y_end = center + scale * y_dir

    x_color, y_color = colors
    x_label, y_label = labels
    ax.plot([center[0], x_end[0]], [center[1], x_end[1]], color=x_color, lw=lw, zorder=zorder)
    ax.plot([center[0], y_end[0]], [center[1], y_end[1]], color=y_color, lw=lw, zorder=zorder)
    ax.text(*x_end, x_label, color=x_color, fontsize=fontsize, fontweight='bold', zorder=zorder)
    ax.text(*y_end, y_label, color=y_color, fontsize=fontsize, fontweight='bold', zorder=zorder)

    return x_end, y_end


def plot_morph(ax, row, rad: float | None = 150, reg=None, rotation_deg=None, annotate_orientation=None,
               show_em_axes=None, min_rad: float | None = None, margin=10, scale_bar_um: float | None = None):
    """Plot an XY morphology projection of a skeleton centred on its soma.

    Args:
        ax: Matplotlib Axes to plot on.
        row: DataFrame row with a ``skel`` attribute (a ``skeliner.Skeleton``)
            and, if `reg` is given, a ``field`` column.
        rad: Half-width of the axis limits in µm around the soma centre. If
            ``None``, auto-fit instead to the skeleton's own extent (its
            furthest non-axon node from the soma, plus `margin`), clamped to
            at least `min_rad` if given.
        reg: Optional fitted 2p<->EM registration dict (see
            ``eyewire2_functional_analysis.registration``). If given, the
            skeleton is rotated/flipped into the 2p/retinal reference frame
            using the per-field fit for ``row['field']`` (falling back to the
            global fit, with a warning, if that field has none). Takes
            precedence over `rotation_deg`.
        rotation_deg: Optional manual counterclockwise rotation (degrees)
            applied instead, when no fitted `reg` is available. Ignored if
            `reg` is given.
        annotate_orientation: If ``True``, label the axes edges with
            Nasal/Temporal/Dorsal/Ventral via :func:`annotate_retinal_axes`
            (only correct when `reg` orients the skeleton). Defaults to
            ``reg is not None``.
        show_em_axes: If ``True``, draw the EM X/Y axis indicator (see
            :func:`plot_em_axis_indicator`), autofit to `rad`, centred on the
            soma. Only meaningful when `reg` orients the skeleton. Defaults
            to ``reg is not None``.
        min_rad: When `rad` is ``None``, the minimum half-width to use even if
            the skeleton itself is smaller.
        margin: When `rad` is ``None``, extra half-width (µm) added around the
            skeleton's own extent.
        scale_bar_um: If given, draw a scale bar of this length (µm) near the
            bottom-left corner of the view, via :func:`plot_scale_bar`.

    Returns:
        tuple: ``(sx, sy, sz)`` – soma centre coordinates in µm.
    """
    skel = row.skel
    if reg is not None:
        from eyewire2_functional_analysis.registration import align_skel
        skel = align_skel(skel, reg, field=row['field'], direction='em_to_2p')
    elif rotation_deg:
        skel = rotate_skel(skel, rotation_deg=rotation_deg)

    sk.plot.projection(skel, ax=ax, plane='xy')  # , color_by="ntype", skel_cmap='Grays')
    sx, sy, sz = skel.soma.center

    if rad is None:
        pts = skel.nodes[:, :2]
        if skel.ntype is not None:
            non_axon = skel.ntype != 2
            if non_axon.any():
                pts = pts[non_axon]
        rad = np.max(np.abs(pts - [sx, sy])) + margin
        if min_rad is not None:
            rad = max(rad, min_rad)

    ax.set_xlim(sx - rad, sx + rad)
    ax.set_ylim(sy - rad, sy + rad)

    if annotate_orientation is None:
        annotate_orientation = reg is not None
    if annotate_orientation:
        annotate_retinal_axes(ax)

    if show_em_axes is None:
        show_em_axes = reg is not None
    if show_em_axes:
        plot_em_axis_indicator(ax, reg, field=row['field'], direction='em_to_2p')

    if scale_bar_um is not None:
        plot_scale_bar(ax=ax, x0=sx - rad + scale_bar_um / 2 + 5, y0=sy - rad + 5,
                        size=scale_bar_um, unit='µm', text=True, fontsize=8)

    return sx, sy, sz


def plot_mosaic(df, extent=(350, 1000, 0, 650)):
    """Plot a coverage-density mosaic map for all cells in ``df``.

    Builds a coverage-density map from the convex hull of each cell's skeleton
    nodes and overlays one randomly sampled morphology projection.

    Args:
        df: DataFrame with a ``skel`` column containing ``skeliner.Skeleton`` objects.
        extent: Tuple ``(xmin, xmax, ymin, ymax)`` in µm defining the map bounds.

    Returns:
        tuple: ``(fig, ax)`` – the Matplotlib Figure and Axes.
    """
    import cell_mosaics

    assert df.shape[0] > 0, "No data to plot"
    mapper = cell_mosaics.CoverageDensityMapper(field_bounds=extent, resolution=500)
    for i, (seg_id, row) in enumerate(df.iterrows()):
        mapper.add_convex_hull(row.skel.nodes[row.skel.nodes[:, 2] > -10, :2])
    fig, ax, im = mapper.plot_coverage(colormap='bone_r', plot_cell_outlines=True)
    for i, (seg_id, row) in enumerate(df.sample(1).iterrows()):
        plot_morph(ax=ax, row=row)
    ax.set(xlim=extent[:2], ylim=extent[2:])
    return fig, ax


# DSI / OSI from djimaging
def plot_retina_orientation(ax, tdist=50, x0=0, y0=0, size=1000, fontsize=14):
    """Draw a retinal orientation cross (N/T/V/D labels) on the given axes.

    Args:
        ax: Matplotlib Axes to draw on.
        tdist: Distance from the cross tips to the text labels in data units.
        x0: X coordinate of the cross centre.
        y0: Y coordinate of the cross centre.
        size: Full length of each arm of the cross in data units.
        fontsize: Font size for the direction labels.
    """
    ax.plot([x0 - size / 2, x0 + size / 2], [y0, y0], c='k', solid_capstyle='butt', clip_on=False)
    ax.plot([x0, x0], [y0 - size / 2, y0 + size / 2], c='k', solid_capstyle='butt', clip_on=False)
    ax.text(x0 - size / 2 - tdist, y0, 'N', c='k', va='center', ha='right', fontsize=fontsize)
    ax.text(x0 + size / 2 + tdist, y0, 'T', c='k', va='center', ha='left', fontsize=fontsize)
    ax.text(x0, y0 - size / 2 - tdist, 'V', c='k', va='top', ha='center', fontsize=fontsize)
    ax.text(x0, y0 + size / 2 + tdist, 'D', c='k', va='bottom', ha='center', fontsize=fontsize)


def plot_stack_average(ax, stack_avg, pixel_size_um, x_offset, y_offset, cmap='viridis', alpha=0.7, gamma=0.5):
    """Display a gamma-corrected, percentile-normalised image of a stack average.

    Args:
        ax: Matplotlib Axes to display the image on.
        stack_avg: 2-D or 3-D array to display (first two axes are spatial).
        pixel_size_um: Physical size of one pixel in µm, used to set the extent.
        x_offset: X coordinate of the image centre in µm.
        y_offset: Y coordinate of the image centre in µm.
        cmap: Colormap name passed to ``imshow``.
        alpha: Transparency of the image.
        gamma: Gamma exponent applied after percentile normalisation.

    Returns:
        numpy.ndarray: The extent array ``[xmin, xmax, ymin, ymax]`` in µm.
    """
    extent = get_extent(stack_avg, pixel_size_um, x_offset, y_offset)

    im = stack_avg.astype(float)
    vmin = np.percentile(im, q=5, axis=(0, 1))
    vmax = np.percentile(im, q=99, axis=(0, 1))
    im = (im - vmin) / (vmax - vmin)
    im = np.clip(im, 0, 1) ** gamma

    ax.imshow(im.T, extent=extent, cmap=cmap, interpolation='none', alpha=alpha)

    return extent


def make_square_bounding_box(xs, ys):
    """Compute a square bounding box that contains all points ``(xs, ys)``.

    Expands the shorter dimension symmetrically so that the resulting box has
    equal width and height.

    Args:
        xs: Iterable of x coordinates.
        ys: Iterable of y coordinates.

    Returns:
        tuple: ``(xmin, xmax, ymin, ymax)`` of the square bounding box.
    """
    # Step 1: Find initial min and max
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Step 2: Determine width and height
    width = xmax - xmin
    height = ymax - ymin

    # Step 3: Expand the smaller side to match the larger
    if width > height:
        # Increase height
        diff = width - height
        ymin -= diff / 2
        ymax += diff / 2
    elif height > width:
        # Increase width
        diff = height - width
        xmin -= diff / 2
        xmax += diff / 2

    return xmin, xmax, ymin, ymax


def plot_roi_mask(ax, rois, extent):
    """Display an ROI mask image with distinct colours per ROI ID.

    Negates and up-samples the mask (3× in each spatial dimension) before
    displaying it with a ``'jet'`` colourmap.

    Args:
        ax: Matplotlib Axes to display the image on.
        rois: 2-D integer array where each unique positive value identifies an ROI;
            zero or negative values are treated as background (shown as NaN).
        extent: Extent tuple ``[xmin, xmax, ymin, ymax]`` passed to ``imshow``.
    """
    _rois = -rois.copy()
    _rois = _rois.astype(float)
    _rois[_rois <= 0] = np.nan
    _rois = np.repeat(np.repeat(_rois, 3, axis=0), 3, axis=0)
    ax.imshow(_rois.T, cmap='jet', extent=extent)


def plot_sac_lines(ax, xlim, text=True, con='#FFC09F', coff='#17CFB9', ls='-', lw=1):
    """Draw horizontal ON and OFF stratification lines for an IPL depth plot.

    Args:
        ax: Matplotlib Axes to draw on.
        xlim: ``(xmin, xmax)`` in data units that defines the line span.
        text: If ``True``, annotate the lines with ``'ON'`` and ``'OFF'`` labels.
        con: Colour for the ON line.
        coff: Colour for the OFF line.
        ls: Line style string.
        lw: Line width.
    """
    ax.plot(xlim, [0, 0], c=con, ls=ls, lw=lw)
    ax.plot(xlim, [12, 12], c=coff, ls=ls, lw=lw)
    if text:
        ax.text(xlim[1], 0, '  ON', va='top', ha='right', color=con, fontsize=8)
        ax.text(xlim[1], 12, '  OFF', va='bottom', ha='right', color=coff, fontsize=8)


def plot_ipl_profile(ax, row, c='#DA3B3C', text=False):
    """Plot the IPL depth-density profile of a cell skeleton.

    Prunes soma nodes from the skeleton, computes a z-density profile using
    ``pywarper``, and renders it as a horizontal density curve against IPL depth.

    Args:
        ax: Matplotlib Axes to plot on.
        row: DataFrame row containing a ``'skel'`` key with a ``skeliner.Skeleton``.
        c: Colour of the density curve.
        text: If ``True``, annotate the ON/OFF stratification lines with labels.
    """
    from pywarper.warpers import get_z_profile
    import skeliner as sk
    from copy import deepcopy
    
    skel = deepcopy(row['skel'])
    skel.node2verts = None
    sk.post.prune(
        skel=skel,
        kind="nodes",
        nodes=np.where(skel.ntype == 2)[0]
    )
    zlim = (-30, 30)
    
    z_dict = get_z_profile(
        skel=skel,
        extent=zlim,
    )
    ipl = z_dict['x']
    dens = z_dict['distribution']

    vmax = dens.max()
    xlim = -0.1*vmax, vmax*1.1

    ax.set_aspect('auto', 'box')
    ax.set_ylim(zlim)

    plot_sac_lines(ax, xlim, text=text)

    ax.set_xlim(xlim)
    ax.plot(dens, ipl, c=c, lw=1)
    ax.set(xticks=[], yticks=[], xlabel=None, ylabel=None)
    ax.axis('off')