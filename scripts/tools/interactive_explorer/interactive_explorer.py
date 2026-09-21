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
# # Interactive EM <-> function explorer (v0)
#
# Three tabs, sharing one loaded dataset:
# - **Cell explorer** -- pick a 2p ROI, either from the field/ROI dropdowns or by clicking
#   a dot in the XY scatter, and see its EM skeleton (also overlaid on the XY scatter
#   itself) next to its direction-selectivity/morphology, chirp, and mouse-cam responses.
# - **Stimulus timeline** -- scrub absolute session time and see the ROI traces recorded at
#   that instant (heatmap), the field/stimulus footprint that was on screen at that instant,
#   and that field's cells, individually selectable.
# - **Cell type explorer** -- pick a cell class, morphological cell type, and recording
#   field (or all fields) and see every matching cell's EM skeleton, placed at its true
#   position in the shared retinal frame, next to its chirp/moving-bar/natural-movie/
#   direction-tuning responses overlaid across cells (plus the across-cell mean +/- SD).
#   Click a cell's skeleton/marker to highlight it across all its panels.
#
# This is a **quick first cut**, deliberately not fancy yet.
#
# Built with [Panel](https://panel.holoviz.org/) rather than `ipywidgets` -- two ways to run
# this:
# - Opened in Jupyter (`uv run jupyter lab`, right-click -> Open With -> Notebook) for
#   cell-by-cell exploration.
# - As a standalone local app, no Jupyter needed:
#   `uv run panel serve scripts/tools/interactive_explorer/interactive_explorer.py --show`

# %%
import asyncio
import os
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from scipy.spatial import ConvexHull, QhullError

from eyewire2_functional_analysis import baden16_utils, data_loader, plot_morph, plot_traces, registration, style
from eyewire2_functional_analysis.ds import MB_DIRS, MB_DIRS_SYMBOLS_D_UP
from eyewire2_functional_analysis.space_mapping import align_and_place_skel
from eyewire2_functional_analysis.stimulus import stim_outlines

style.set_rc_params()
pn.extension('plotly', 'modal')

# %% [markdown]
# ## Load data
#
# Shared by all three tabs. Restrict to ROIs that matched a proofread EM cell
# (`load_df_rois_morph`). Not all of those have an EM skeleton file on disk (`add_skels`
# leaves `skel` as `None` when the SWC file is missing) -- those are still selectable, just
# without a morphology panel/skeleton overlay (see `render_cell`/`make_morph_figure` below).

# %%
df_rois = data_loader.load_df_rois()
df_fields = data_loader.load_df_fields()
df_outline = data_loader.load_df_outline()
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = data_loader.add_skels(df, inplace=True)
df = df.reset_index(drop=True)

n_with_skel = df['skel'].notnull().sum()
print(f"{n_with_skel} / {len(df)} EM-matched ROI(s) have a skeleton file; the rest are still selectable")

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
reg = registration.load_registration(REG_FILE)

df['label'] = df.apply(lambda r: f"{r['field']} - ROI {r['roi_id']}\n{r['Cell Type']}", axis=1)

FIELDS = sorted(df['field'].unique())

# Session log used both by the "Stimulus timeline" tab (to build its session blocks) and by
# the "Cell type explorer" tab (to colour its response overlays by recording time).
CONSOL_PATH = os.path.join(data_loader.DATA_ROOT, 'experiment-overview_consolidated.csv')
df_stim_log = pd.read_csv(CONSOL_PATH, sep=';', on_bad_lines='warn')

# Fixed XY range for the cell-picker scatter, from the ROI cloud alone (with
# some padding, plus a fixed extra margin so there's room to see e.g. a
# stimulus footprint extending past the ROI cloud) -- kept constant regardless
# of which cell's skeleton is overlaid, so a large skeleton gets cropped
# rather than the view zooming out.
ZOOM_OUT_MARGIN_UM = 200
_x_pad = 0.1 * (df['temporal_nasal_pos_um'].max() - df['temporal_nasal_pos_um'].min()) + ZOOM_OUT_MARGIN_UM
_y_pad = 0.1 * (df['ventral_dorsal_pos_um'].max() - df['ventral_dorsal_pos_um'].min()) + ZOOM_OUT_MARGIN_UM
X_RANGE = (df['temporal_nasal_pos_um'].min() - _x_pad, df['temporal_nasal_pos_um'].max() + _x_pad)
Y_RANGE = (df['ventral_dorsal_pos_um'].min() - _y_pad, df['ventral_dorsal_pos_um'].max() + _y_pad)

# Morphology-panel scale bar / minimum zoom, so a 100 um scale bar always
# fits comfortably even for a small cell.
MORPH_SCALE_BAR_UM = 100
MORPH_MARGIN_UM = 10
MORPH_MIN_RAD_UM = MORPH_SCALE_BAR_UM / 2 + MORPH_MARGIN_UM

# %% [markdown]
# ## Stimulus footprint overlay
#
# Geometry/parameters mirror `scripts/tutorial/plot_stimulus_overlay/plot_stimulus_overlay.py`.
# The footprint is drawn at the *field's* registered centre (`load_df_fields`),
# not any individual ROI's position, so it only depends on which field the
# selected ROI belongs to.

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

STIM_OPTIONS = {'Chirp': 'Chirp', 'Moving bar': 'DS', 'Mouse cam': 'MouseCam_Right'}


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


# 0/180, 45/225, 90/270 and 135/315 sweep the same rectangle (just in
# opposite directions), so only 4 distinct bar orientations need to be drawn.
BAR_DIRECTIONS_UNIQUE = [0, 45, 90, 135]


def field_centre(field):
    """(x0, y0) of `field`'s registered centre, in the retinal frame."""
    field_row = df_fields[df_fields['field'] == field].iloc[0]
    return field_row['field_temporal_nasal_pos_um'], field_row['field_ventral_dorsal_pos_um']


def polys_to_xy(geom):
    """Flatten a Shapely Polygon/MultiPolygon's exterior(s) into plotly x/y lists, None-separated."""
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    xs, ys = [], []
    for poly in polys:
        px, py = poly.exterior.xy
        xs += list(px) + [None]
        ys += list(py) + [None]
    return xs, ys


def stimulus_trace(stim_type, field):
    """Plotly filled trace for `stim_type`'s footprint at `field`'s registered centre."""
    x0, y0 = field_centre(field)
    footprint = stimulus_footprint(stim_type, x0, y0)
    xs, ys = polys_to_xy(footprint)
    return go.Scatter(x=xs, y=ys, mode='lines', fill='toself',
                       fillcolor='rgba(255, 215, 0, 0.4)', line=dict(color='black', width=1),
                       hoverinfo='skip', showlegend=False)


def bar_direction_trace(field):
    """Plotly dashed trace of the actual moving-bar shape at each of the 4 distinct sweep directions."""
    x0, y0 = field_centre(field)
    xs, ys = [], []
    for angle in BAR_DIRECTIONS_UNIQUE:
        rect = stim_outlines.box(BAR_DY_UM, BAR_DX_UM, angle=angle, x0=x0, y0=y0, FOV_diam=FOV_DIAM_UM)
        rect_xs, rect_ys = polys_to_xy(rect)
        xs += rect_xs
        ys += rect_ys
    return go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='black', width=1.5, dash='dash'),
                       hoverinfo='skip', showlegend=False)


def empty_line_trace():
    return go.Scatter(x=[], y=[], mode='lines', hoverinfo='skip', showlegend=False)


# %% [markdown]
# ## Tab 1: Cell explorer -- cell picker (scatter + field/ROI/stimulus dropdowns) and detail panel

# %%
# ROI scatter colour-coding by ground-truth 'Cell Class' (from the master proofread-cell
# spreadsheet, see data_loader.load_df_rois_morph) -- RGC vs. AC (i.e. dAC, the only two classes
# populating the ROI-matched data). Anything else (Fragment, missing) falls back to gray.
CELL_CLASS_COLORS = {'RGC': '#d62728', 'AC': '#1f77b4'}
CELL_CLASS_DEFAULT_COLOR = '#999999'
SELECTED_COLOR = 'black'


def skel_edge_trace(row):
    """Plotly line trace for `row`'s EM skeleton, rotated+placed into the retinal frame."""
    skel = align_and_place_skel(
        row.skel, reg, field=row['field'],
        target_xy=(row['temporal_nasal_pos_um'], row['ventral_dorsal_pos_um']),
    )
    seg = np.stack([skel.nodes[skel.edges[:, 0], :2], skel.nodes[skel.edges[:, 1], :2]], axis=1)
    xs, ys = [], []
    for (x0, y0), (x1, y1) in zip(seg[:, 0], seg[:, 1]):
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='black', width=1),
                       hoverinfo='skip', showlegend=False)


def make_scatter_figure(selected_idx=None, stim_type='Chirp', field=None):
    """Build the XY cell-picker scatter, highlighting `selected_idx`, overlaying its skeleton, and
    drawing `stim_type`'s footprint at `field`'s centre (defaulting to `selected_idx`'s own field,
    or `FIELDS[0]` if neither is given -- used when the timeline tab has a field active but no ROI
    selected).

    Always emits exactly 4 traces (ROI dots, skeleton, stimulus footprint,
    bar-direction outlines -- empty traces when there's nothing to overlay)
    so the figure's trace count never changes between renders. Panel/Plotly's
    `Plotly.react`-based update otherwise gets confused about which trace is
    which when the trace count changes from one selection to the next, which
    was causing stray/delayed click events to resolve against the wrong point.
    """
    colors = [CELL_CLASS_COLORS.get(c, CELL_CLASS_DEFAULT_COLOR) for c in df['Cell Class']]
    sizes = [8] * len(df)
    if selected_idx is not None:
        colors[selected_idx] = SELECTED_COLOR
        sizes[selected_idx] = 14

    roi_trace = go.Scatter(
        x=df['temporal_nasal_pos_um'],
        y=df['ventral_dorsal_pos_um'],
        mode='markers',
        marker=dict(size=sizes, color=colors, line=dict(width=0.5, color='white')),
        text=df['label'],
        hoverinfo='text',
        showlegend=False,
    )

    row = df.iloc[selected_idx] if selected_idx is not None else None
    if row is not None and row['skel'] is not None:
        skel_trace = skel_edge_trace(row)
    else:
        skel_trace = empty_line_trace()

    if field is None:
        field = row['field'] if row is not None else FIELDS[0]
    stim_footprint_trace = stimulus_trace(stim_type, field)
    bar_dir_trace = bar_direction_trace(field) if stim_type == 'DS' else empty_line_trace()

    fig = go.Figure(data=[roi_trace, skel_trace, stim_footprint_trace, bar_dir_trace])
    fig.update_layout(
        width=480, height=480,
        margin=dict(l=50, r=10, t=10, b=40),
        showlegend=False,
        xaxis_title='Temporal ↔ Nasal [um]',
        yaxis_title='Ventral ↔ Dorsal [um]',
        xaxis=dict(range=X_RANGE),
        yaxis=dict(range=Y_RANGE, scaleanchor='x', scaleratio=1),
    )
    return fig


# Moving-bar and mouse-cam response snippets always have 3 repeats (confirmed on the released
# data: `bar_snippets` is always 8 dirs x 3 reps, and `plot_traces.plot_mc_test_snippets` always
# picks 3 test-clip presentations); `chirp_snippets` varies per row (currently always 5, but the
# repetition-colorbar logic below stays generic rather than hardcoding that).
N_MB_REPS = 3
N_MC_REPS = 3


def render_morph_panel(fig, gs_slot, row, correct_by_cellclass=False):
    """Render `row`'s XY morphology (with its IPL stratification profile alongside, from a
    precomputed CSV -- see `scripts/preprocessing/compute_ipl_profiles.py` and
    `data_loader.load_ipl_profile` -- rather than computing it live via `pywarper`, which is
    slow and depends on an optional native solver for good performance) plus its predicted-type
    probability bars below (`plot_morph.plot_type_prediction_bars`).

    `correct_by_cellclass` (see `cellclass_correction_dropdown` below) is forwarded to the
    type-prediction bars: the classifier confuses RGCs and dACs fairly often, and this lets the
    known ground-truth `Cell Class` override that at all 3 prediction levels.

    Shared by `render_cell` (main tab) and `render_morph_only` (timeline tab) so a cell's
    morphology panel looks identical in both places.
    """
    rows_gs = gs_slot.subgridspec(2, 1, height_ratios=(3, 1), hspace=0.35)
    morph_gs = rows_gs[0, 0].subgridspec(1, 2, width_ratios=(3, 1), wspace=0.05)
    ax_morph = fig.add_subplot(morph_gs[0, 0])
    ax_ipl = fig.add_subplot(morph_gs[0, 1])
    ax_types = fig.add_subplot(rows_gs[1, 0])

    if row['skel'] is not None:
        plot_morph.plot_morph(ax_morph, row, reg=reg, rad=None, min_rad=MORPH_MIN_RAD_UM, margin=MORPH_MARGIN_UM,
                        scale_bar_um=MORPH_SCALE_BAR_UM, annotate_orientation=True)
        ipl_profile = data_loader.load_ipl_profile(row['Latest SegID'])
        if ipl_profile is not None:
            plot_morph.plot_ipl_profile_from_arrays(ax_ipl, ipl_profile['z'].to_numpy(), ipl_profile['dens'].to_numpy())
        else:
            # Not yet computed (or failed) -- see scripts/preprocessing/compute_ipl_profiles.py.
            ax_ipl.text(0.5, 0.5, 'IPL profile\nnot cached', ha='center', va='center', fontsize=7, wrap=True)
            ax_ipl.axis('off')
    else:
        ax_morph.text(0.5, 0.5, f"no skeleton found for {row['Latest SegID']}",
                      ha='center', va='center', wrap=True)
        ax_morph.axis('off')
        ax_ipl.axis('off')
    ax_morph.set_title(row['label'])

    plot_morph.plot_type_prediction_bars(ax_types, row, correct_by_cellclass=correct_by_cellclass)

    return ax_morph, ax_ipl, ax_types


def align_ipl_panel_box(ax_morph, ax_ipl):
    """Match `ax_ipl`'s vertical box extent to `ax_morph`'s.

    `ax_morph` is drawn with `aspect='equal'` (see `plot_morph.plot_morph`), which -- via
    matplotlib's default `adjustable='box'` -- shrinks its *rendered* box to enforce that aspect
    ratio, so it usually ends up shorter than its allotted gridspec slot; `ax_ipl` has no such
    constraint and fills its slot fully, leaving the two side-by-side panels visually
    top/bottom-misaligned and differently sized. Must be called after the figure's layout is
    finalized (e.g. right after `fig.tight_layout()`), since the aspect-driven box shrinking is
    only resolved at draw time.
    """
    morph_pos = ax_morph.get_position()
    ipl_pos = ax_ipl.get_position()
    ax_ipl.set_position([ipl_pos.x0, morph_pos.y0, ipl_pos.width, morph_pos.height])


def add_rep_colorbar(fig, cax, n_reps, label='Repetition'):
    """Discrete colorbar in `cax` mapping repeat index -> the colour
    `plot_traces.plot_snippets_and_average` uses for that repeat (via
    `plot_traces.get_repeat_colors`), so the repeat-colour code used in the chirp/moving-bar/
    mouse-cam panels can be read off directly.
    """
    colors = plot_traces.get_repeat_colors(n_reps)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(0.5, n_reps + 1.5, 1.0), n_reps)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(1, n_reps + 1))
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return cb


def render_cell(idx, correct_by_cellclass=False):
    """Detail figure for row `idx`: DS-on-morph (left, full height), chirp (top right), mouse-cam snippet (bottom right)."""
    row = df.iloc[idx]
    fig = plt.figure(figsize=(14, 5.5))
    try:
        gs = fig.add_gridspec(2, 4, width_ratios=(1, 1.2, 1.1, 0.06), wspace=0.5)

        ax_morph, ax_ipl, _ = render_morph_panel(fig, gs[:, 0], row, correct_by_cellclass=correct_by_cellclass)

        plot_traces.plot_bar_dir_grid(fig, gs[:, 1], row)  # sets its own DSI/OSI suptitle

        ax_chirp = fig.add_subplot(gs[0, 2])
        plot_traces.plot_chirp(ax_chirp, row)
        ax_chirp.set_title('Chirp')

        ax_mc = fig.add_subplot(gs[1, 2])
        plot_traces.plot_mc_test_snippets(ax_mc, row)
        ax_mc.set_title('Mouse cam (3 test reps)')

        for ax in (ax_chirp, ax_mc):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # The Chirp and MB/MC colorbars share column 3 -- Chirp's next to `ax_chirp` (top),
        # MB/MC's next to `ax_mc` (bottom) -- so the two are column-aligned with each other; when
        # they'd be identical (same repeat count), draw one colorbar spanning both rows instead.
        n_chirp_reps = row['chirp_snippets'].shape[1]
        if n_chirp_reps == N_MB_REPS:
            cax_shared = fig.add_subplot(gs[:, 3])
            add_rep_colorbar(fig, cax_shared, N_MB_REPS, label='Repetition\n(Chirp, MB, MC)')
        else:
            cax_chirp = fig.add_subplot(gs[0, 3])
            add_rep_colorbar(fig, cax_chirp, n_chirp_reps, label='Repetition\n(Chirp)')
            cax_mb_mc = fig.add_subplot(gs[1, 3])
            add_rep_colorbar(fig, cax_mb_mc, N_MB_REPS, label='Repetition\n(MB, MC)')

        fig.tight_layout()
        if row['skel'] is not None:
            align_ipl_panel_box(ax_morph, ax_ipl)
    except Exception as e:
        fig.clear()
        fig.text(0.5, 0.5, f"Could not plot {row['label']}: {e}", ha='center', va='center')
    return fig


# %%
def roi_options_for_field(field):
    """``{"ROI <id> - <cell type>": global df index}`` for one field, ordered by roi_id."""
    sub = df[df['field'] == field].sort_values('roi_id')
    return {f"ROI {r['roi_id']} - {r['Cell Type']}": idx for idx, r in sub.iterrows()}


field_dropdown = pn.widgets.Select(name='Field', options=FIELDS, value=FIELDS[0])
roi_dropdown = pn.widgets.Select(name='ROI', options=roi_options_for_field(FIELDS[0]))
stim_dropdown = pn.widgets.Select(name='Stimulus outline', options=STIM_OPTIONS, value='Chirp')

def make_scatter_legend():
    """Caption explaining `make_scatter_figure`'s ROI dot colours."""
    return pn.pane.Markdown(
        "🔴 RGC&nbsp;&nbsp;&nbsp;🔵 AC (dAC)&nbsp;&nbsp;&nbsp;⚫ selected",
        styles={'font-size': '11px', 'margin-top': '-8px'},
    )


# The classifier confuses RGCs and dACs fairly often (both are recorded from the ganglion cell
# layer and driven through the same functional-response pipeline); this toggle applies
# `baden16_utils.correct_probs_by_cellclass` to zero out/renormalize each cell's predicted-type
# probabilities to be consistent with its known ground-truth `Cell Class` (RGC vs. AC/dAC).
# Shared across both tab-1 views (main detail panel and the timeline tab's morphology panel) --
# switching it re-renders whichever cell is currently shown in each.
CELLCLASS_CORRECTION_OPTIONS = {'Raw model prediction': False, 'Corrected by Cell Class (RGC vs. dAC)': True}
cellclass_correction_dropdown = pn.widgets.Select(name='Predicted type', options=CELLCLASS_CORRECTION_OPTIONS,
                                                   value=False)

scatter_pane = pn.pane.Plotly(make_scatter_figure(selected_idx=roi_dropdown.value, stim_type=stim_dropdown.value))
detail_pane = pn.pane.Matplotlib(render_cell(roi_dropdown.value, correct_by_cellclass=cellclass_correction_dropdown.value),
                                  tight=True, format='png', dpi=100)


def select_index(idx):
    """Update the scatter highlight/overlay and the detail panel for global row `idx`.

    This is the *only* place that actually redraws the detail panel. Both the
    ROI dropdown and the scatter click handler just set `roi_dropdown.value`;
    it's the dropdown's own 'value' watcher (`on_roi_change`, below) that
    calls this -- so there's exactly one codepath that updates the detail
    panel, regardless of how the selection was made. The scatter itself is
    redrawn here too (its stimulus footprint follows the newly selected
    ROI's field), reusing the currently selected stimulus type.
    """
    scatter_pane.object = make_scatter_figure(selected_idx=idx, stim_type=stim_dropdown.value)

    old_fig = detail_pane.object
    detail_pane.object = render_cell(idx, correct_by_cellclass=cellclass_correction_dropdown.value)
    plt.close(old_fig)


def on_roi_change(event):
    select_index(event.new)


def on_stim_change(event):
    scatter_pane.object = make_scatter_figure(selected_idx=roi_dropdown.value, stim_type=event.new)


# Set True while `on_scatter_click` is repointing `field_dropdown` at a
# different field so it already knows the target ROI -- suppresses
# `on_field_change`'s normal "reset to that field's first ROI" reaction,
# which would otherwise race with (and get overwritten by) the click's own
# explicit selection a moment later.
_suppress_field_reset = False


def on_field_change(event):
    if _suppress_field_reset:
        return
    options = roi_options_for_field(event.new)
    # Set options and value together (one atomic Param update) so there's no
    # intermediate state where `.value` is invalid for the new `.options` --
    # that would otherwise make Param auto-correct `.value` to the new
    # options' first entry as its own separate change, firing `on_roi_change`
    # an extra time before our intended value took effect.
    roi_dropdown.param.update(options=options, value=next(iter(options.values())))


def on_scatter_click(event):
    global _suppress_field_reset

    click_data = event.new
    scatter_pane.click_data = None  # reset so a stale replay of this event is a no-op
    if not click_data or not click_data.get('points'):
        return
    point = click_data['points'][0]
    if point.get('curveNumber', 0) != 0:
        return  # ignore clicks on the skeleton/stimulus-overlay traces
    idx = point.get('pointIndex', point.get('pointNumber'))
    field = df.iloc[idx]['field']

    if field_dropdown.value != field:
        _suppress_field_reset = True
        try:
            field_dropdown.value = field
        finally:
            _suppress_field_reset = False
        # Same atomic-update reasoning as in `on_field_change` above.
        roi_dropdown.param.update(options=roi_options_for_field(field), value=idx)
    else:
        roi_dropdown.value = idx  # triggers on_roi_change above


def on_cellclass_correction_change(event):
    """Re-render whichever cell is currently shown in each tab-1 view under the new correction setting."""
    select_index(roi_dropdown.value)
    if tl_roi_dropdown.value is not None:
        redraw_tl_morph(tl_roi_dropdown.value)


field_dropdown.param.watch(on_field_change, 'value')
roi_dropdown.param.watch(on_roi_change, 'value')
stim_dropdown.param.watch(on_stim_change, 'value')
scatter_pane.param.watch(on_scatter_click, 'click_data')
cellclass_correction_dropdown.param.watch(on_cellclass_correction_change, 'value')

cell_explorer_layout = pn.Row(pn.Column(scatter_pane, make_scatter_legend(), field_dropdown, roi_dropdown, stim_dropdown),
                pn.Column(detail_pane, cellclass_correction_dropdown))

# %% [markdown]
# ## Tab 2: Stimulus timeline
#
# A second view: scrub absolute session time and see the ROI traces recorded
# at that instant (heatmap), the field/stimulus footprint that was on screen
# at that instant (reusing `make_scatter_figure` from above), and that
# field's cells -- individually selectable, independent of the timeline.
#
# Session timing comes from `experiment-overview_consolidated.csv` (loaded once, above --
# same file used in `scripts/analysis/light_exposure/EW2_stim_history.py`). Each field's
# chirp/moving-bar/mouse-cam recording is its own acquisition with an independent local
# clock (`*_pp_trace_t0`/`*_pp_trace_dt`) -- there is no single continuous trace spanning the
# whole session, so time outside the currently active recording block (a field switch, a
# setup gap, or -- for GCL3's Chirp, which was recorded twice -- the second of the two
# logged presentations, since only one of them made it into the released `chirp_trace`) is
# treated as "nothing recorded" rather than an error.

# %%
HEATMAP_PRE_S = 5.0
HEATMAP_POST_S = 15.0

STIM_LABELS = {v: k for k, v in STIM_OPTIONS.items()}
STIM_TRACE_PREFIX = {'DS': 'bar', 'Chirp': 'chirp', 'MouseCam_Right': 'mc'}


def build_session_blocks():
    """One entry per (field, stim type) recording block: the earliest logged presentation
    that actually has a recording file, with its real duration taken from the matching
    `*_pp_trace` array length (the logged `t_dur_s` can be shorter than what's actually
    in the trace -- e.g. GCL3's Chirp block runs ~181 s of trace vs. 165 s logged).
    """
    blocks = []
    for field in FIELDS:
        field_idx = int(field[3:])
        rep_row = df[df['field'] == field].iloc[0]
        for stim_type, col_prefix in STIM_TRACE_PREFIX.items():
            candidates = df_stim_log[
                (df_stim_log['fieldID'] == field_idx)
                & (df_stim_log['stimFileName'] == stim_type)
                & (df_stim_log['dataFileName'].notna())
            ]
            if not len(candidates):
                continue
            pick = candidates.loc[candidates['t_abs_s'].idxmin()]
            trace_dt = rep_row[f'{col_prefix}_pp_trace_dt']
            n_samples = len(rep_row[f'{col_prefix}_pp_trace'])

            if col_prefix == 'mc':
                triggertimes = rep_row['mc_triggertimes']
            else:
                triggertimes = rep_row[f'{col_prefix}_triggertimes_snippets']
            blocks.append(dict(
                field=field, stim_type=stim_type, col_prefix=col_prefix,
                t_start=float(pick['t_abs_s']), t_end=float(pick['t_abs_s']) + n_samples * trace_dt,
                triggertimes=triggertimes,
            ))
    return sorted(blocks, key=lambda b: b['t_start'])


SESSION_BLOCKS = build_session_blocks()
SESSION_T_MIN = SESSION_BLOCKS[0]['t_start']
SESSION_T_MAX = SESSION_BLOCKS[-1]['t_end']


def active_block_at(t):
    """The session block (field + stim type) recording at absolute time `t`, or `None`
    if `t` falls between recordings (field switch / setup gap)."""
    for block in SESSION_BLOCKS:
        if block['t_start'] <= t < block['t_end']:
            return block
    return None


# Fixed, shared color scale for the heatmap (99th percentile magnitude across all pp
# traces, symmetric around 0) so the colors don't rescale/jump every time the slider moves.
_all_pp_vals = np.concatenate([
    np.concatenate(df[f'{p}_pp_trace'].apply(np.asarray).values) for p in STIM_TRACE_PREFIX.values()
])
HEATMAP_VMAX = np.nanpercentile(np.abs(_all_pp_vals), 99)
HEATMAP_VMIN = -HEATMAP_VMAX


# Canonical stimulus movies (same file for every presentation of that stim type -- a
# Chirp/DS presentation always shows the same movie, just positioned differently per field,
# which is already handled by the footprint overlay in the scatter panel). Mouse-cam has no
# single canonical movie: each field was shown a different one of 20 pre-shuffled sequences,
# recorded in `df`'s own `mc` column (e.g. `"mc16"` -> sequence 16).
STIM_MOVIE_NPZ = {
    'Chirp': ('global_chirp', 'chirp1000_setup3_movie_and_trigger.npz'),
    'DS': ('moving_bar', 'DS_setup3_movie_and_trigger.npz'),
}
MC_ARRAY_DIR = os.path.join(data_loader.DATA_2P, 'stimuli', 'mouse_cam_movies', 'mc_arrays')
MC_FRAME_RATE_HZ = 30.0  # matches stimulus_tools.FRAMES_PER_SECOND

_stim_npz_cache = {}
_mc_movie_cache = {}


def load_stim_movie_npz(stim_type):
    """(time, intensity) for `stim_type`'s canonical movie -- `intensity` has shape (T, 1),
    the frame mean at each `time` sample. Loaded once per stim type, then cached.

    Valid as a literal field-mean intensity for Chirp (a full-field spot). For DS it's a
    coarser proxy -- the bar only covers part of the field at any instant -- but the bar's
    actual footprint/position is already drawn separately in the scatter panel, so this is
    just meant to show *when* stimulation happens, not *where*.
    """
    if stim_type not in _stim_npz_cache:
        subdir, fname = STIM_MOVIE_NPZ[stim_type]
        movie = np.load(os.path.join(data_loader.DATA_2P, 'stimuli', subdir, fname))
        trigger = movie['trigger']
        trigger0 = np.argmax(trigger) # Start stimulus at first trigger
        intensity = movie['stimulus'].mean(axis=(1, 2))[trigger0:, None]
        _stim_npz_cache[stim_type] = (movie['time'], intensity)
    return _stim_npz_cache[stim_type]


def load_mc_movie(field):
    """(time, intensity) for `field`'s actual mouse-cam sequence -- `intensity` has shape
    (T, 2), the frame mean of the [Green, UV] channels at each `time` sample.

    `MC{n}.npy` (`data/stimuli/mouse_cam_movies/mc_to_numpy.py`) was built with
    `color_sequence='BGR'`, which merges the source montage's own (R, G, B) into an "RGB"-mode
    image as (B, G, R) -- so in the saved array, index 0 ("R" slot) holds the source's *blue*
    channel, index 1 ("G" slot) is the source's green channel unchanged, and index 2 ("B" slot)
    holds the source's *red* channel. Empirically (checked on MC15), index 0 is near-blank
    (mean ~2.5, std ~5.7) -- consistent with it being the setup's unused "red" channel -- while
    indices 1 and 2 both carry real structured content (mean ~85-90, std ~80-95), i.e. Green
    and UV respectively. So the real channels are 1 (Green) and 2 (UV); index 0 is dropped.

    Loaded once per distinct sequence (there are only 5, one per field, some repeated), then cached.
    """
    seq = int(df.loc[df['field'] == field, 'mc'].iloc[0][2:])  # "mc16" -> 16
    if seq not in _mc_movie_cache:
        arr = np.load(os.path.join(MC_ARRAY_DIR, f'MC{seq}.npy'), mmap_mode='r')
        intensity = np.asarray(arr[:, :, :, 1:3].mean(axis=(1, 2)), dtype=np.float64)  # [Green, UV]
        time = np.arange(arr.shape[0]) / MC_FRAME_RATE_HZ
        _mc_movie_cache[seq] = (time, intensity)
    return _mc_movie_cache[seq]


def stim_intensity_trace(block, t_center):
    """(t_rel, intensity, channel_labels) -- mean stimulus intensity around `t_center`.

    Aligned by treating the movie's own time 0 as `block['t_start']` (the presentation's
    logged absolute start) -- the same anchor `heatmap_matrix` uses for the raw pp traces.
    Cross-checked against both movies' own trigger arrays: each movie's first trigger falls
    within ~0.03 s of its own t=0 for Chirp (full pre-roll before the sweep starts is part of
    the DS movie itself, not a misalignment), consistent with the <=0.13 s `*_pp_trace_t0`
    pre-roll already seen on the raw traces -- so this approximation is accurate to well
    under one heatmap sample (`dt` ~0.128 s).

    Chirp/DS's canonical movie is only a single repeat (~33 s / ~36 s) while the recorded
    block covers all of its repeats back-to-back (5x / 3x, ~181 s / ~118 s) -- so within the
    block, time is wrapped onto one repeat's cycle (`% movie_duration`) rather than indexed
    directly; outside the block's own start/end (a different presentation, or a gap) the
    result is blanked to NaN, same as `heatmap_matrix`.
    """
    if block['stim_type'] in STIM_MOVIE_NPZ:
        time, intensity = load_stim_movie_npz(block['stim_type'])
        labels = ['Intensity']
    else:
        time, intensity = load_mc_movie(block['field'])
        labels = ['Green', 'UV']

    dt_movie = time[1] - time[0]
    movie_duration = time[-1] + dt_movie
    t_rel = np.arange(-HEATMAP_PRE_S, HEATMAP_POST_S, dt_movie)
    t_abs = t_center + t_rel

    t0_stim = block['t_start'] + block['triggertimes'].flat[0]

    t_local = (t_abs - t0_stim) % movie_duration
    idx = np.clip(np.round(t_local / dt_movie).astype(int), 0, len(time) - 1)

    out = np.full((len(t_rel), intensity.shape[1]), np.nan)
    valid = (t_abs >= t0_stim) & (t_abs < block['t_end'])
    out[valid] = intensity[idx[valid]]
    return t_rel, out, labels


def heatmap_matrix(block, t_center):
    """(t_rel, matrix, sub) -- ROI x time matrix of `block`'s pp trace around `t_center`.

    `matrix[i]` is `sub.iloc[i]`'s trace resampled onto the common `t_rel` grid
    (`-HEATMAP_PRE_S`..`+HEATMAP_POST_S` relative to `t_center`); samples that fall
    outside `block`'s own recorded span are left as NaN. `sub` keeps `df`'s own row
    labels (not reset), so a global ROI index can be located in it via `sub.index.get_loc`.
    """
    sub = df[df['field'] == block['field']].sort_values('roi_id')
    col_prefix = block['col_prefix']
    dt = sub[f'{col_prefix}_pp_trace_dt'].iloc[0]
    t_rel = np.arange(-HEATMAP_PRE_S, HEATMAP_POST_S, dt)
    t_abs = t_center + t_rel

    mat = np.full((len(sub), len(t_rel)), np.nan)
    for i, (_, row) in enumerate(sub.iterrows()):
        trace = row[f'{col_prefix}_pp_trace']
        trace_dt = row[f'{col_prefix}_pp_trace_dt']
        trace_t0 = row[f'{col_prefix}_pp_trace_t0']
        idx = np.round((t_abs - block['t_start'] - trace_t0) / trace_dt).astype(int)
        valid = (idx >= 0) & (idx < len(trace))
        mat[i, valid] = trace[idx[valid]]
    return t_rel, mat, sub


def render_heatmap(t, selected_idx=None):
    """Figure for the block active at `t` (or a placeholder if none): field-mean stimulus
    intensity (top), the selected ROI's own trace (middle, if any), and the full ROI x time
    heatmap (bottom) with `selected_idx`'s row outlined -- all three sharing one time axis.

    The heatmap's colorbar lives in its own gridspec column so it only shrinks the bottom
    panel's *colorbar column* rather than stealing width from the panel itself -- otherwise
    the top two panels (which have no colorbar) end up wider than the heatmap and the shared
    time axis visually misaligns between them.
    """
    block = active_block_at(t)
    if block is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.text(0.5, 0.5, 'No recording active at this time\n(field switch / setup gap)',
                ha='center', va='center', wrap=True)
        ax.axis('off')
        return fig

    fig = plt.figure(figsize=(6.5, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 4), width_ratios=(1, 0.03), hspace=0.15, wspace=0.05)
    ax_stim = fig.add_subplot(gs[0, 0])
    ax_trace = fig.add_subplot(gs[1, 0], sharex=ax_stim)
    ax = fig.add_subplot(gs[2, 0], sharex=ax_stim)
    cax = fig.add_subplot(gs[2, 1])

    t_rel, mat, sub = heatmap_matrix(block, t)
    im = ax.imshow(mat, aspect='auto', origin='lower', cmap='RdBu_r',
                    vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, interpolation='none',
                    extent=(t_rel[0], t_rel[-1], -0.5, len(sub) - 0.5))
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel(f'Time relative to t = {t:.1f} s [s]')
    ax.set_ylabel('ROI')
    step = max(1, len(sub) // 15)
    ax.set_yticks(range(0, len(sub), step))
    ax.set_yticklabels(sub['roi_id'].iloc[::step])
    fig.colorbar(im, cax=cax, label='Norm. Ca. (pp trace)')

    if selected_idx is not None and selected_idx in sub.index:
        pos = sub.index.get_loc(selected_idx)
        sel_row = sub.loc[selected_idx]

        # Other ROIs of the same morphology type ('Cell Type') recorded in the same field --
        # drawn first, thin and translucent, so the selected ROI's own trace (below) stays on top.
        same_type = sub[(sub['Cell Type'] == sel_row['Cell Type']) & (sub.index != selected_idx)]
        for other_idx in same_type.index:
            other_pos = sub.index.get_loc(other_idx)
            ax_trace.plot(t_rel, mat[other_pos], color='#1a7a1a', lw=0.8, alpha=0.5)

        ax.add_patch(Rectangle((t_rel[0], pos - 0.5), t_rel[-1] - t_rel[0], 1,
                                fill=False, edgecolor='lime', linewidth=2, zorder=5))
        ax_trace.plot(t_rel, mat[pos], color='#1a7a1a', lw=1.2)
        ax_trace.set_ylabel(f"ROI {sub.loc[selected_idx, 'roi_id']}")
        if len(same_type):
            other_ids = ', '.join(str(rid) for rid in same_type['roi_id'])
            ax_trace.set_title(f"{sel_row['Cell Type']} -- other ROI(s) in field: {other_ids}", fontsize=7)
        else:
            ax_trace.set_title(str(sel_row['Cell Type']), fontsize=7)
    else:
        ax_trace.text(0.5, 0.5, 'No cell selected', ha='center', va='center', transform=ax_trace.transAxes)
        ax_trace.set_ylabel('Selected\nROI')
    ax_trace.axvline(0, color='black', lw=1, ls='--')
    ax_trace.tick_params(labelbottom=False)

    stim_t_rel, stim_intensity, stim_labels = stim_intensity_trace(block, t)
    stim_colors = {'Intensity': 'black', 'UV': 'purple', 'Green': 'green'}
    for ch, label in enumerate(stim_labels):
        ax_stim.plot(stim_t_rel, stim_intensity[:, ch], color=stim_colors[label], lw=1, label=label)
    ax_stim.axvline(0, color='black', lw=1, ls='--')
    ax_stim.set_ylabel('Stimulus\nintensity')
    ax_stim.set_title(f"{block['field']} – {STIM_LABELS[block['stim_type']]}")
    if len(stim_labels) > 1:
        ax_stim.legend(loc='upper right', fontsize=7, frameon=False, handlelength=1.2)
    ax_stim.tick_params(labelbottom=False)

    with warnings.catch_warnings():
        # tight_layout warns that the manually-added `cax` (not created via the ax=
        # convenience path) is "not compatible" -- checked empirically that the three main
        # panels still end up with identical x0/x1 regardless, so the warning is benign.
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout()
    return fig


def render_morph_only(idx, correct_by_cellclass=False):
    """Standalone morphology panel for row `idx` (EM skeleton + IPL profile + predicted-type bars,
    via `render_morph_panel`; no chirp/bar/mc detail plots), or a placeholder if `idx` is `None`
    (no cell selected)."""
    fig = plt.figure(figsize=(4.5, 5))
    if idx is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No cell selected', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        return fig

    row = df.iloc[idx]
    gs = fig.add_gridspec(1, 1)
    ax_morph, ax_ipl, _ = render_morph_panel(fig, gs[0, 0], row, correct_by_cellclass=correct_by_cellclass)
    fig.tight_layout()
    if row['skel'] is not None:
        align_ipl_panel_box(ax_morph, ax_ipl)
    return fig


def tl_roi_options_for_field(field):
    """Like `roi_options_for_field`, but with a leading "None" entry -- the timeline tab
    defaults to no cell selected (rather than auto-picking one) whenever the slider moves
    into a new field, so the ROI panel/morphology panel only show a specific cell once the
    user actually asks for one."""
    return {'None': None, **roi_options_for_field(field)}


# %% [markdown]
# ### Session map
#
# The slider alone gives no sense of *where* the actual recordings are along an ~8300 s
# session -- this draws every block as a colored segment (color = stim type), highlights the
# gaps between them (field switches / setup time), labels each field, and marks the current
# slider position, so scrubbing has visual context instead of blind guess-and-search.

# %%
STIM_TYPE_COLORS = {'Chirp': '#4c72b0', 'DS': '#dd8452', 'MouseCam_Right': '#55a868'}


def format_hms(seconds):
    """"H:MM:SS" (or "M:SS" under an hour) for a session-absolute time in seconds."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


SESSION_MAP_FIGSIZE = (9, 1.3)
SESSION_MAP_DPI = 100
SESSION_MAP_WIDTH_PX = SESSION_MAP_FIGSIZE[0] * SESSION_MAP_DPI  # matches the slider's own width, below


def render_session_map(t):
    """Horizontal map of the whole session (see markdown above), with a vertical marker at `t`."""
    fig, ax = plt.subplots(figsize=SESSION_MAP_FIGSIZE, dpi=SESSION_MAP_DPI)

    for block in SESSION_BLOCKS:
        ax.broken_barh([(block['t_start'], block['t_end'] - block['t_start'])], (0, 1),
                        facecolors=STIM_TYPE_COLORS[block['stim_type']], edgecolor='white', linewidth=0.5)

    for prev, nxt in zip(SESSION_BLOCKS[:-1], SESSION_BLOCKS[1:]):
        gap = nxt['t_start'] - prev['t_end']
        if gap > 0:
            ax.broken_barh([(prev['t_end'], gap)], (0, 1), facecolors='none', edgecolor='0.5',
                            hatch='///', linewidth=0.5)

    for field in FIELDS:
        field_blocks = [b for b in SESSION_BLOCKS if b['field'] == field]
        mid = (field_blocks[0]['t_start'] + field_blocks[-1]['t_end']) / 2
        ax.text(mid, 1.1, field, ha='center', va='bottom', fontsize=8)

    ax.axvline(t, color='black', lw=2, zorder=5)
    ax.set_xlim(SESSION_T_MIN - 30, SESSION_T_MAX + 30)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: format_hms(val)))
    ax.set_xlabel('Session time [h:mm:ss]')
    for spine in ('top', 'left', 'right'):
        ax.spines[spine].set_visible(False)

    legend_handles = [Patch(facecolor=STIM_TYPE_COLORS[s], label=STIM_LABELS[s]) for s in STIM_TYPE_COLORS]
    legend_handles.append(Patch(facecolor='none', edgecolor='0.5', hatch='///', label='gap / setup'))
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.6),
              ncol=4, frameon=False, fontsize=7, handlelength=1.2)

    fig.tight_layout()
    return fig


# %%
_initial_block = active_block_at(SESSION_T_MIN)
assert _initial_block is not None, f"no recording block found at session start ({SESSION_T_MIN:.1f} s)"
tl_state = dict(field=_initial_block['field'], stim_type=_initial_block['stim_type'])

time_slider = pn.widgets.FloatSlider(
    name='Session time [s]', start=SESSION_T_MIN, end=SESSION_T_MAX, step=1.0, value=SESSION_T_MIN,
    width=SESSION_MAP_WIDTH_PX,
)
JUMP_OPTIONS = {
    f"{b['field']} – {STIM_LABELS[b['stim_type']]}  ({format_hms(b['t_start'] + b['triggertimes'].flat[0])})": b['t_start'] + b['triggertimes'].flat[0]
    for b in SESSION_BLOCKS
}
jump_dropdown = pn.widgets.Select(name='Jump to recording', options=JUMP_OPTIONS, value=SESSION_T_MIN)
tl_status = pn.pane.Markdown(f"**t = {SESSION_T_MIN:.1f} s** – {tl_state['field']}, {STIM_LABELS[tl_state['stim_type']]}")
tl_roi_dropdown = pn.widgets.Select(name='ROI', options=tl_roi_options_for_field(tl_state['field']), value=None)

session_map_pane = pn.pane.Matplotlib(render_session_map(time_slider.value), tight=True, format='png',
                                       dpi=SESSION_MAP_DPI, width=SESSION_MAP_WIDTH_PX)
heatmap_pane = pn.pane.Matplotlib(render_heatmap(time_slider.value, selected_idx=tl_roi_dropdown.value),
                                   tight=True, format='png', dpi=100)
tl_scatter_pane = pn.pane.Plotly(
    make_scatter_figure(selected_idx=tl_roi_dropdown.value, stim_type=tl_state['stim_type'], field=tl_state['field']))
tl_morph_pane = pn.pane.Matplotlib(
    render_morph_only(tl_roi_dropdown.value, correct_by_cellclass=cellclass_correction_dropdown.value),
    tight=True, format='png', dpi=100)


def redraw_tl_scatter(idx):
    tl_scatter_pane.object = make_scatter_figure(
        selected_idx=idx, stim_type=tl_state['stim_type'], field=tl_state['field'])


def redraw_tl_morph(idx):
    old_fig = tl_morph_pane.object
    tl_morph_pane.object = render_morph_only(idx, correct_by_cellclass=cellclass_correction_dropdown.value)
    plt.close(old_fig)


def redraw_tl_heatmap():
    old_fig = heatmap_pane.object
    heatmap_pane.object = render_heatmap(time_slider.value, selected_idx=tl_roi_dropdown.value)
    plt.close(old_fig)


def redraw_session_map():
    old_fig = session_map_pane.object
    session_map_pane.object = render_session_map(time_slider.value)
    plt.close(old_fig)


def on_tl_roi_change(event):
    redraw_tl_scatter(event.new)
    redraw_tl_morph(event.new)
    redraw_tl_heatmap()


def on_tl_scatter_click(event):
    """Same click-to-select as the main tab's scatter, except a click on a ROI outside
    the field currently active on the timeline is ignored -- field here always follows
    the slider, not the click."""
    click_data = event.new
    tl_scatter_pane.click_data = None  # reset so a stale replay of this event is a no-op
    if not click_data or not click_data.get('points'):
        return
    point = click_data['points'][0]
    if point.get('curveNumber', 0) != 0:
        return  # ignore clicks on the skeleton/stimulus-overlay traces
    idx = point.get('pointIndex', point.get('pointNumber'))
    if df.iloc[idx]['field'] != tl_state['field']:
        return
    tl_roi_dropdown.value = idx  # triggers on_tl_roi_change above


def on_time_slider_change(event):
    t = event.new
    block = active_block_at(t)

    if block is None:
        tl_status.object = f"**t = {t:.1f} s** – no recording active (field switch / setup gap)"
    else:
        tl_status.object = f"**t = {t:.1f} s** – {block['field']}, {STIM_LABELS[block['stim_type']]}"
        if block['field'] != tl_state['field']:
            tl_state['field'] = block['field']
            tl_state['stim_type'] = block['stim_type']
            options = tl_roi_options_for_field(block['field'])
            # Reset to "None" (no auto-selected cell) on every field change. Also redraw
            # explicitly right after: if the dropdown's value was already `None` (no ROI had
            # been selected), this update is a value no-op and won't itself fire
            # `on_tl_roi_change`, so the scatter/morph wouldn't otherwise pick up the new field.
            tl_roi_dropdown.param.update(options=options, value=None)
            redraw_tl_scatter(None)
            redraw_tl_morph(None)
        else:
            tl_state['stim_type'] = block['stim_type']
            redraw_tl_scatter(tl_roi_dropdown.value)  # same ROI selection, new stim overlay

    redraw_tl_heatmap()
    redraw_session_map()


def on_jump_change(event):
    time_slider.value = event.new  # triggers on_time_slider_change above


tl_roi_dropdown.param.watch(on_tl_roi_change, 'value')
tl_scatter_pane.param.watch(on_tl_scatter_click, 'click_data')
time_slider.param.watch(on_time_slider_change, 'value')
jump_dropdown.param.watch(on_jump_change, 'value')

timeline_layout = pn.Column(
    session_map_pane,
    time_slider,
    jump_dropdown,
    tl_status,
    pn.Row(
        heatmap_pane,
        pn.Column(tl_scatter_pane, tl_roi_dropdown),
        pn.Column(tl_morph_pane, cellclass_correction_dropdown),
    ),
)

# %% [markdown]
# ## Tab 3: Cell type explorer
#
# Pick a cell class (RGC / dAC / Other -- see `Cell Class`), a morphological cell type
# within that class (`Cell Type`, the ground-truth EM/proofreading label -- see
# `data_loader.load_df_rois_morph`), and either one recording field (`GCL0`..`GCL4`) or
# all of them, and see every matching cell's:
# - EM skeleton (left, click a cell to select it -- see below), overlaid at its true
#   position in the shared retinal frame (same placement logic as `scripts/tutorial/
#   plot_retinal_outline/plot_retinal_outline.py` and tab 1 above).
# - Chirp, moving-bar, natural-movie (mouse-cam), and moving-bar direction-tuning
#   responses (right), each averaged per cell (chirp: `chirp_average_norm`; moving bar:
#   `bar_time_component`, the SVD time kernel already used this way in `plot_dataframe.
#   plot_df_chirp_and_bar`; natural movie: the mean of the 3 repeated test-clip
#   presentations, same snippets as `plot_traces.plot_mc_test_snippets`; direction tuning:
#   `bar_dir_component`), all overlaid on their own axes plus the across-cell mean +/- SD.
#
# Click a cell's skeleton/marker in the morphology panel to highlight it (bold/black, on
# top) across all 5 panels, dimming every other cell to gray -- click the background (or
# the retinal/field outline) to clear the selection. Changing any filter above also clears
# it, since a selection made under one field/cell-type/class combination may not exist
# under another.
#
# This tab's own widgets/callbacks are prefixed `ct_` where they'd otherwise collide with
# tab 1's identically-purposed ones (its own field selector, its own "correct by cell
# class" toggle) -- the two tabs' selections are independent.

# %%
# The 'All fields' sentinel is a plain string, not `None` -- Panel's `Select` widget can't
# tell "nothing selected" apart from "the selected option's value happens to be `None`", so a
# `None`-valued option never renders as selected even when it's `value`, leaving the widget
# looking empty. A distinct string value sidesteps that.
ALL_FIELDS = 'All fields'
FIELD_OPTIONS = [ALL_FIELDS] + FIELDS

# Per-cell (not per-field) colour cycle for `make_morph_figure`/`make_ipl_figure` -- cells
# recorded in the same field are common (most `cell_type`/`field` selections have several),
# and would otherwise all render in one indistinguishable colour. Wraps around past 10 cells
# in one selection.
CELL_TAB10_HEX = [mcolors.to_hex(c) for c in plt.get_cmap('tab10').colors]


def cell_colors(sub):
    """``{df row index: hex colour}``, one per row of `sub`, cycling through `CELL_TAB10_HEX`
    in `sub`'s own (field, then roi_id -- see `selected_cells`) row order. Built once from
    `sub` and passed to both `make_morph_figure` and `make_ipl_figure` so the same cell gets
    the same colour in both panels, even though `make_ipl_figure` silently skips cells with no
    cached profile."""
    return {key: CELL_TAB10_HEX[i % len(CELL_TAB10_HEX)] for i, key in enumerate(sub.index)}

MORPH_MIN_ZOOM_UM = 100
# CSS pixel size of the (Plotly) morphology pane -- kept roughly square, close to the
# height of the (Matplotlib) response-panel figure below, so the two panes sit right next
# to each other at a similar visual scale.
MORPH_PLOTLY_SIZE = dict(width=480, height=480)

# Figure size (inches) of the response-panel figure (chirp/moving-bar/natural-movie/
# direction-tuning/colorbar) -- shared with `add_bar_dir_panel_and_colorbar`, which needs
# the actual width/height ratio to size the direction-tuning polar panel as a physical
# square. Morphology is a separate (Plotly) pane, not part of this figure -- see
# `make_morph_figure`. Height is half of the panel's "natural" (one-per-row) size, so the
# 3 stacked response panels take up less vertical space overall.
TRACES_FIGSIZE = (7, 4.5)

# %% [markdown]
# ### Cell-class filter (RGC / dAC / Other)
#
# The ground-truth `Cell Class` column only ever holds `'RGC'` or `'AC'` (the released
# ROI-matched data's only two populated classes -- `'AC'` here means displaced amacrine
# cell, dAC; see `CELL_CLASS_COLORS` above) plus a handful of `'Fragment'`/missing rows,
# bucketed here as 'Other'. A handful of `Cell Type` labels (e.g. "F-mini-ON", "Other") are
# used for cells of more than one class, so the class filter is applied at the row level
# (`cell_class_mask`), not just used to prune the type list -- otherwise switching class
# with such a type still selected would silently mix classes back in.

# %%
CELL_CLASS_OPTIONS = ['RGC', 'dAC', 'Other']


def cell_class_mask(df_, cell_class):
    """Boolean mask selecting `df_`'s rows matching the UI's `cell_class` filter."""
    if cell_class == 'RGC':
        return df_['Cell Class'] == 'RGC'
    elif cell_class == 'dAC':
        return df_['Cell Class'] == 'AC'
    else:
        return ~df_['Cell Class'].isin(['RGC', 'AC'])


CELL_TYPES_BY_CLASS = {
    cell_class: sorted(df.loc[cell_class_mask(df, cell_class), 'Cell Type'].dropna().unique())
    for cell_class in CELL_CLASS_OPTIONS
}

# %% [markdown]
# ### Recording time (session-absolute), for colouring the chirp overlay
#
# Every ROI in a field is recorded simultaneously, so "when was this cell recorded" is a
# per-field quantity -- the field's own Chirp acquisition start time, read from the shared
# `df_stim_log` session log loaded above (same logic as tab 2's `build_session_blocks`).
# Where a field's Chirp was logged more than once (e.g. GCL3, restarted once), the earliest
# logged presentation with an actual recording file is used, matching that same logic.

# %%
def field_chirp_start_time(field):
    """Session-absolute time (s) of `field`'s Chirp acquisition start."""
    field_idx = int(field[3:])
    candidates = df_stim_log[
        (df_stim_log['fieldID'] == field_idx)
        & (df_stim_log['stimFileName'] == 'Chirp')
        & (df_stim_log['dataFileName'].notna())
    ]
    return float(candidates['t_abs_s'].min())


FIELD_CHIRP_T0 = {f: field_chirp_start_time(f) for f in FIELDS}
TIME_CMAP = plt.get_cmap('viridis')
TIME_NORM = mcolors.Normalize(vmin=min(FIELD_CHIRP_T0.values()), vmax=max(FIELD_CHIRP_T0.values()))

# One colour per Baden et al. 2016 functional cluster (e.g. '12a'), sampled continuously
# across `BADEN_CLUSTER_INFO`'s own row order -- clusters are already grouped contiguously
# by supergroup then group in that table, so nearby clusters (same supergroup) land near
# each other on the colormap too, while still being a continuous (not discrete/7-colour)
# scale over all 75. Used by `cluster_prob_bar_html`'s bar segments.
CLUSTER_CMAP = plt.get_cmap('turbo')
CLUSTER_NAME_TO_COLOR_HEX = {
    name: mcolors.to_hex(CLUSTER_CMAP(i / (len(baden16_utils.BADEN_CLUSTER_INFO) - 1)))
    for i, name in enumerate(baden16_utils.BADEN_CLUSTER_INFO[:, 1])
}


# %% [markdown]
# ### Cell log table
#
# A simple Excel log the user can append to from the GUI (see the "Add to table" button
# below) -- one row per cell of interest, with a free-text comment. Lives next to this
# script (not under `data/`, since it's user-curated output, not released/input data), so
# it's created fresh on first run rather than shipped with the repo.

# %%
LOG_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'celltype_explorer_log.xlsx')
LOG_TABLE_COLUMNS = ['field', 'ROI number', 'morphological cell type', 'passed quality filter', 'comment']

if not os.path.exists(LOG_TABLE_PATH):
    pd.DataFrame(columns=LOG_TABLE_COLUMNS).to_excel(LOG_TABLE_PATH, index=False)


# %% [markdown]
# ## Rendering

# %%
def selected_cells(field, cell_type, qfilt_only, cell_class):
    """Rows of `df` matching `cell_type` and `cell_class` (see `cell_class_mask` -- needed
    even though `cell_type` alone usually determines it, since a few `Cell Type` labels
    span more than one class) and `field` (unless 'All fields'), sorted by field then
    roi_id so overlay order is stable/reproducible across renders."""
    sub = df[df['Cell Type'] == cell_type]
    sub = sub[cell_class_mask(sub, cell_class)]
    if field != ALL_FIELDS:
        sub = sub[sub['field'] == field]
    if qfilt_only:
        sub = sub[sub['qfilt']]
    return sub.sort_values(['field', 'roi_id'])


def field_outline_rect(field):
    """``(xs, ys)`` of a closed rectangle outlining `field`'s actual 2p imaging window
    (`nxpix`/`nypix` pixels at `pixel_size_um`, centered on its registered
    `field_temporal_nasal_pos_um`/`field_ventral_dorsal_pos_um` position) -- i.e. the
    small (~94x94 um) patch the ROIs' somata were actually scanned in, as distinct from
    the much larger dendritic fields their morphologies can extend into, and from the
    (much bigger, ~1000 um) stimulus field of view used for the footprint overlays in
    tab 1 above."""
    frow = df_fields[df_fields['field'] == field].iloc[0]
    w = frow['nxpix'] * frow['pixel_size_um']
    h = frow['nypix'] * frow['pixel_size_um']
    cx, cy = frow['field_temporal_nasal_pos_um'], frow['field_ventral_dorsal_pos_um']
    xs = [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2]
    ys = [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2]
    return xs, ys


MORPH_VIEW_OPTIONS = ['Skeletons', 'Dendritic hull']


def dendritic_hull_xy(skel):
    """``(hull_x, hull_y)`` -- a closed polygon outline (last point repeats the first) of
    the convex hull of `skel`'s dendritic nodes (xy projection), excluding axon nodes
    (`skel.ntype == 2`, the same convention `plot_morph.plot_morph`'s auto-fit radius
    uses) when node-type info is available, so the hull reflects the *dendritic* arbor's
    footprint, not one stretched out by a far-reaching axon. Returns `None` if fewer than
    3 (non-collinear) points remain -- too degenerate for a hull."""
    pts = skel.nodes[:, :2]
    if skel.ntype is not None:
        non_axon = skel.ntype != 2
        if non_axon.any():
            pts = pts[non_axon]
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    hull_pts = pts[hull.vertices]
    hx = list(hull_pts[:, 0]) + [hull_pts[0, 0]]
    hy = list(hull_pts[:, 1]) + [hull_pts[0, 1]]
    return hx, hy


def make_morph_figure(sub, field, selected_key, view_mode='Skeletons'):
    """Plotly figure overlaying every row in `sub`'s EM skeleton at its true 2p position
    (via `align_and_place_skel`), colour-coded per cell (`cell_colors`, cycling
    `CELL_TAB10_HEX` -- not by field, since many cells in one selection typically share a
    field), and click-selectable (see `on_morph_click`). Cells without a skeleton file are
    still marked (as an 'x') at their 2p position, so the group's full spatial coverage is
    visible even where morphology is missing.

    The retinal outline is drawn as spatial context whenever `field` is `ALL_FIELDS`, and the
    recorded field(s)' own imaging-window outline(s) are always drawn
    (`field_outline_rect`) -- but the view always zooms dynamically to fit every plotted
    morphology (with a margin) rather than to the whole retina/outline, so cells are
    legible regardless of how tightly they cluster, whether that's one field or all five.
    The fit is always to each cell's *full* skeleton (every node, including axon), never to
    just what `view_mode` actually draws -- so switching `view_mode` alone never changes
    the axis range, even though a dendritic hull's own extent (axon excluded) is a subset
    of the full skeleton's.

    `view_mode` (one of `MORPH_VIEW_OPTIONS`) switches between drawing each cell's full
    skeleton (`'Skeletons'`, the default) or just the convex hull of its dendritic arbor
    (`'Dendritic hull'`, `dendritic_hull_xy` -- a translucent filled polygon, cheaper to
    read at a glance when overlaying many cells at once, at the cost of the actual branch
    structure). Cells whose skeleton is too degenerate for a hull (fewer than 3 non-
    collinear dendritic nodes) just have no hull drawn in that mode (still counted in the
    title's "n with skeleton" and still part of the axis-range fit, same as any other cell
    with a skeleton file).

    Multiple rows of `sub` occasionally share the same EM cell (`Latest SegID`) -- most likely
    because the 2p ROI segmentation oversplit what should have been one ROI into several, each
    independently matched to the same EM cell. Only one of them (the selected one, if any of
    them is selected, else the first by row order) actually gets its skeleton/hull drawn, so
    the same morphology is never rendered more than once; the others still get their own
    position marker (see below), just no skeleton/hull geometry of their own. When this
    happens, the panel title calls it out explicitly ("N ROIs share M morphologies (likely
    oversplit 2p ROIs)") rather than just silently reporting a skeleton count that would
    otherwise look unremarkable (e.g. "2/2 with skeleton").

    If `selected_key` (a `df` row index, or `None`) names one of `sub`'s cells, that cell's
    skeleton/hull/marker are drawn solid black (same line width as everyone else -- only
    colour changes) and every other cell is dimmed (rather than converted to gray, so each
    cell's own colour stays visible/legible); with no selection, all cells use their own
    colour at full opacity.

    Uses `go.Scattergl` (WebGL), not plain `go.Scatter` (SVG), throughout -- with up to
    ~90 cells x ~15k skeleton nodes each, SVG rendering of that many short disconnected
    line segments visibly degrades into a speckled/dotted mess rather than continuous
    dendrites (checked empirically); WebGL renders it cleanly.

    Every clickable/background trace uses `hoverinfo='none'`, not `'skip'`: per Plotly's
    own docs, `'skip'` excludes a trace from hover-distance computation entirely, which
    also silently makes it un-clickable -- `'none'` still participates in hit-testing (so
    `on_morph_click` gets a `curveNumber`) while simply not drawing a tooltip. Using
    `'skip'` here previously meant clicks on the outlines/background grid produced no
    event at all, so "click background to deselect" never fired.

    Returns ``(fig, curve_to_key)``: `curve_to_key[curveNumber]` is the `df` row index the
    trace at that Plotly curve number belongs to, or `None` for anything that isn't a cell
    (retinal/field outlines, the invisible background click-catcher) -- used by
    `on_morph_click` to resolve a click into a cell selection (`None` meaning "clicked the
    background", i.e. deselect).
    """
    fig = go.Figure()
    curve_to_key = {}
    fields_shown = FIELDS if field == ALL_FIELDS else [field]
    colors = cell_colors(sub)

    if field == ALL_FIELDS:
        fig.add_trace(go.Scattergl(
            x=df_outline['temporal_nasal_pos_um'], y=df_outline['ventral_dorsal_pos_um'], mode='lines',
            line=dict(color='dimgray', width=1, dash='dot'), hoverinfo='none', showlegend=False))
        curve_to_key[len(fig.data) - 1] = None

    for f in fields_shown:
        xs, ys = field_outline_rect(f)
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode='lines', line=dict(color='lightgray', width=1),
                                  hoverinfo='none', showlegend=False))
        curve_to_key[len(fig.data) - 1] = None

    # For cells sharing a `Latest SegID`, pick the one row whose skeleton/hull actually gets
    # drawn below -- the selected one if any of them is selected (so selecting a "hidden"
    # duplicate still shows its highlighted morphology), else the first by `sub`'s own row
    # order (field then roi_id, see `selected_cells`). Restricted to rows that actually have a
    # skeleton (`row['skel'] is not None`) -- otherwise a seg ID's first-seen row could be one
    # *without* a skeleton file and "own" a seg ID no row of it ever actually draws.
    skel_seg_id_counts = {}
    skel_owner_key = {}
    for key, row in sub.iterrows():
        if row['skel'] is None:
            continue
        seg_id = row['Latest SegID']
        skel_seg_id_counts[seg_id] = skel_seg_id_counts.get(seg_id, 0) + 1
        if seg_id not in skel_owner_key or key == selected_key:
            skel_owner_key[seg_id] = key
    # seg IDs more than one row of `sub` shares -- i.e. distinct 2p ROIs matched to the same
    # EM cell, most likely because the functional (2p) ROI segmentation oversplit what should
    # have been a single ROI. Surfaced in the panel title below.
    shared_seg_id_counts = {seg_id: n for seg_id, n in skel_seg_id_counts.items() if n > 1}

    xs_cells, ys_cells = [], []
    n_with_skel = 0
    for key, row in sub.iterrows():
        is_selected = key == selected_key
        dimmed = selected_key is not None and not is_selected
        color = 'black' if is_selected else colors[key]
        opacity = 0.25 if dimmed else 1.0
        x, y = row['temporal_nasal_pos_um'], row['ventral_dorsal_pos_um']

        if row['skel'] is not None:
            n_with_skel += 1

        if row['skel'] is not None and skel_owner_key[row['Latest SegID']] == key:
            skel = align_and_place_skel(row['skel'], reg, field=row['field'], target_xy=(x, y))
            # Always fit the view to the *full* skeleton (all nodes, including axon), not
            # to whatever's actually drawn below -- otherwise switching `view_mode` would
            # itself change the axis range (a dendritic hull's own extent, with the axon
            # excluded, is a la strict subset of the full skeleton's).
            xs_cells += list(skel.nodes[:, 0])
            ys_cells += list(skel.nodes[:, 1])

            if view_mode == 'Dendritic hull':
                hull_xy = dendritic_hull_xy(skel)
                if hull_xy is not None:
                    hx, hy = hull_xy
                    # `fillcolor`'s own alpha (independent of the trace-level `opacity`
                    # below, which instead handles selection-dimming) keeps the fill
                    # translucent while the outline stays fully opaque, so overlapping
                    # hulls remain distinguishable. A selected hull gets a more opaque
                    # fill (not a thicker outline -- matching `make_morph_figure`'s
                    # skeleton-mode convention of colour-only, not width, emphasis).
                    fill_alpha = 0.4 if is_selected else 0.18
                    fill_rgba = 'rgba({},{},{},{})'.format(*(int(c * 255) for c in mcolors.to_rgb(color)), fill_alpha)
                    fig.add_trace(go.Scattergl(
                        x=hx, y=hy, mode='lines', fill='toself', fillcolor=fill_rgba,
                        line=dict(color=color, width=1),
                        opacity=opacity, hoverinfo='none', showlegend=False))
                    curve_to_key[len(fig.data) - 1] = key
            else:
                seg_x, seg_y = [], []
                for (x0, y0), (x1, y1) in zip(skel.nodes[skel.edges[:, 0], :2], skel.nodes[skel.edges[:, 1], :2]):
                    seg_x += [x0, x1, None]
                    seg_y += [y0, y1, None]
                fig.add_trace(go.Scattergl(x=seg_x, y=seg_y, mode='lines', line=dict(color=color, width=1),
                                          opacity=opacity, hoverinfo='none', showlegend=False))
                curve_to_key[len(fig.data) - 1] = key
        else:
            xs_cells.append(x)
            ys_cells.append(y)

        fig.add_trace(go.Scattergl(
            x=[x], y=[y], mode='markers',
            marker=dict(size=11 if is_selected else 7, color=color,
                        symbol='x' if row['skel'] is None else 'circle', line=dict(width=0.5, color='white')),
            opacity=opacity, hoverinfo='text', text=f"{row['field']} - ROI {row['roi_id']}<br>{row['Cell Type']}",
            showlegend=False))
        curve_to_key[len(fig.data) - 1] = key

    pad = 50
    xmin, xmax = min(xs_cells) - pad, max(xs_cells) + pad
    ymin, ymax = min(ys_cells) - pad, max(ys_cells) + pad
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin, MORPH_MIN_ZOOM_UM) / 2
    x_range = (cx - half, cx + half)
    y_range = (cy - half, cy + half)

    # Invisible dense marker grid spanning the view -- Plotly doesn't fire a click event
    # for genuinely empty canvas, so without this, clicking the background somewhere with
    # no nearby cell/outline point would silently do nothing rather than deselect.
    grid_coord = np.linspace(-1, 1, 25)
    gx, gy = np.meshgrid(cx + grid_coord * half, cy + grid_coord * half)
    fig.add_trace(go.Scattergl(x=gx.ravel(), y=gy.ravel(), mode='markers',
                              marker=dict(size=18, color='rgba(0,0,0,0)'), hoverinfo='none', showlegend=False))
    curve_to_key[len(fig.data) - 1] = None

    title_text = f"Morphology ({n_with_skel}/{len(sub)} with skeleton)"
    if shared_seg_id_counts:
        n_shared_rois = sum(shared_seg_id_counts.values())
        n_shared_morphs = len(shared_seg_id_counts)
        morph_word = 'morphology' if n_shared_morphs == 1 else 'morphologies'
        title_text += (f" -- {n_shared_rois} ROIs share {n_shared_morphs} {morph_word} "
                       "(likely oversplit 2p ROIs)")

    fig.update_layout(
        width=MORPH_PLOTLY_SIZE['width'], height=MORPH_PLOTLY_SIZE['height'],
        margin=dict(l=50, r=10, t=30, b=40),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(title='Temporal <-> Nasal [um]', range=x_range, gridcolor='#eee', zerolinecolor='#eee'),
        yaxis=dict(title='Ventral <-> Dorsal [um]', range=y_range, scaleanchor='x', scaleratio=1,
                   gridcolor='#eee', zerolinecolor='#eee'),
        title=dict(text=title_text, x=0, xanchor='left', font=dict(size=13)),
    )
    return fig, curve_to_key


def row_color(row):
    """A row's colour for the response-overlay panels: when its field was recorded during
    the session (`FIELD_CHIRP_T0`/`TIME_CMAP`/`TIME_NORM`)."""
    return TIME_CMAP(TIME_NORM(FIELD_CHIRP_T0[row['field']]))


def top2_cluster_probs(row, correct_by_cellclass=False):
    """The predicted-type probability distribution over the 75 Baden et al. 2016 functional
    clusters (e.g. `'12a'` -- the same granularity tab 1's per-cell "Cluster" prediction bar
    shows, one level more specific than the 46 functional groups) for `row` (from its
    `probs_per_cluster` vector, aggregated via `baden16_utils.probs_per_cluster_to_level_probs`,
    whose `cluster_probs` output already carries the `'12a'`-style name directly -- no
    separate ID lookup needed, unlike the group level), sorted descending and truncated to
    the top 2.

    `correct_by_cellclass` matches tab 1's `cellclass_correction_dropdown`/
    `plot_morph.plot_type_prediction_bars`: if set, `row['probs_per_cluster']` is first passed
    through `baden16_utils.correct_probs_by_cellclass(row['Cell Class'], ...)`, zeroing out/
    renormalizing probability mass inconsistent with the cell's known ground-truth `Cell Class`
    (RGC vs. dAC) -- the classifier confuses the two fairly often. Off by default, same as there.

    Returns a list of up to 2 ``(cluster_name, supergroup, prob)`` tuples (fewer only if a
    cell somehow has fewer than 2 clusters with nonzero probability mass)."""
    probs = row['probs_per_cluster']
    if correct_by_cellclass:
        probs = baden16_utils.correct_probs_by_cellclass(row['Cell Class'], probs)
    _, _, cluster_probs = baden16_utils.probs_per_cluster_to_level_probs(probs)
    cluster_probs = [(name, sg, p) for name, sg, p in cluster_probs if p > 0]
    cluster_probs.sort(key=lambda entry: entry[2], reverse=True)
    return cluster_probs[:2]


def top2_group_probs(row, correct_by_cellclass=False):
    """The predicted-type probability distribution over the 46 Baden et al. 2016 functional
    groups (e.g. `'On-Off DS 1'` -- the human-readable "functional cell type" name, one
    level coarser than `top2_cluster_probs`'s clusters) for `row`, aggregated via
    `baden16_utils.probs_per_cluster_to_level_probs`, whose `group_probs` output already
    carries the shortened descriptive group name directly (each cluster's probability
    summed into its parent group first). `correct_by_cellclass` as in `top2_cluster_probs`.

    Note this is a *separate* ranking from `top2_cluster_probs`, not just "look up the
    top-2 clusters' parent groups": a group's total probability mass (summed across all its
    own clusters) can outrank another group whose single best cluster still beats any one of
    the first group's clusters individually.

    Returns a list of up to 2 ``(group_name, supergroup, prob)`` tuples (fewer only if a
    cell somehow has fewer than 2 groups with nonzero probability mass)."""
    probs = row['probs_per_cluster']
    if correct_by_cellclass:
        probs = baden16_utils.correct_probs_by_cellclass(row['Cell Class'], probs)
    _, group_probs, _ = baden16_utils.probs_per_cluster_to_level_probs(probs)
    group_probs = [(name, sg, p) for name, sg, p in group_probs if p > 0]
    group_probs.sort(key=lambda entry: entry[2], reverse=True)
    return group_probs[:2]


# IPL depth range (um) the cached per-cell profiles (`data_loader.load_ipl_profile`) were
# computed over -- matches `plot_morph.compute_ipl_z_profile`'s own default and
# `scripts/preprocessing/compute_ipl_profiles.py`, which produced the cache.
IPL_ZLIM = (-30, 30)
# CSS pixel size of the (Plotly) IPL-overlay pane -- half the width of the morphology pane
# above it (they sit in the same column, left-aligned), a good deal shorter since a depth
# profile only needs a tall-and-narrow plot, not a square one.
IPL_PLOTLY_SIZE = dict(width=MORPH_PLOTLY_SIZE['width'] // 2, height=220)


def make_ipl_figure(sub, selected_key):
    """Plotly figure overlaying every row in `sub`'s cached IPL depth-density profile
    (`data_loader.load_ipl_profile`, keyed by `Latest SegID`) as a horizontal density curve
    against IPL depth (y-axis) -- the same axis convention as `plot_morph.
    plot_ipl_profile_from_arrays`, just drawn with Plotly instead of Matplotlib so it can
    reuse the same click-to-select interaction as `make_morph_figure`.

    Cells with no cached profile (`load_ipl_profile` returns `None` -- `pywarper` failed or
    hasn't been run for that cell) are silently skipped, same as cells with no skeleton file
    are skipped in the morphology panel, rather than surfaced as gaps needing an explanation.
    Cells sharing the same EM cell (`Latest SegID`) share the same profile too -- only one of
    them (the selected one, if any, else the first by row order) gets a curve drawn, same
    dedup as `make_morph_figure`'s skeleton/hull.

    If `selected_key` (a `df` row index, or `None`) names one of `sub`'s cells with a cached
    profile, that cell's curve is drawn solid black (bold) and every other curve is dimmed
    (not converted to gray, matching `make_morph_figure`); with no selection, all curves use
    their own colour (`cell_colors`, matching `make_morph_figure`'s per-cell, not per-field,
    colouring) at full opacity. The ON/OFF SAC (ChAT band) reference lines
    (`plot_morph.plot_sac_lines`'s fixed z=0/z=12 convention) are drawn once, in data
    coordinates, spanning the full plotted density range.

    Returns ``(fig, curve_to_key)``, same contract as `make_morph_figure` -- `curve_to_key`
    maps this figure's own curve numbers back to `df` row indices (or `None` for the SAC
    lines/background), used by the same `on_morph_click`-style handler.
    """
    fig = go.Figure()
    curve_to_key = {}
    colors = cell_colors(sub)

    profiles = {}
    for key, row in sub.iterrows():
        prof = data_loader.load_ipl_profile(row['Latest SegID'])
        if prof is not None:
            profiles[key] = prof

    # Same "one owner per `Latest SegID`" dedup as `make_morph_figure`'s `skel_owner_key`.
    profile_owner_key = {}
    for key, row in sub.iterrows():
        if key not in profiles:
            continue
        seg_id = row['Latest SegID']
        if seg_id not in profile_owner_key or key == selected_key:
            profile_owner_key[seg_id] = key

    vmax = max((prof['dens'].max() for prof in profiles.values()), default=1.0)
    xlim = (-0.1 * vmax, vmax * 1.1)

    fig.add_trace(go.Scattergl(x=list(xlim), y=[0, 0], mode='lines',
                              line=dict(color='#FFC09F', width=1), hoverinfo='none', showlegend=False))
    curve_to_key[len(fig.data) - 1] = None
    fig.add_trace(go.Scattergl(x=list(xlim), y=[12, 12], mode='lines',
                              line=dict(color='#17CFB9', width=1), hoverinfo='none', showlegend=False))
    curve_to_key[len(fig.data) - 1] = None

    for key, row in sub.iterrows():
        if key not in profiles or profile_owner_key[row['Latest SegID']] != key:
            continue
        is_selected = key == selected_key
        dimmed = selected_key is not None and not is_selected
        color = 'black' if is_selected else colors[key]
        opacity = 0.25 if dimmed else 1.0
        prof = profiles[key]
        fig.add_trace(go.Scattergl(
            x=prof['dens'].to_numpy(), y=prof['z'].to_numpy(), mode='lines',
            line=dict(color=color, width=2.5 if is_selected else 1), opacity=opacity,
            hoverinfo='text', text=f"{row['field']} - ROI {row['roi_id']}<br>{row['Cell Type']}",
            showlegend=False))
        curve_to_key[len(fig.data) - 1] = key

    fig.update_layout(
        width=IPL_PLOTLY_SIZE['width'], height=IPL_PLOTLY_SIZE['height'],
        margin=dict(l=50, r=10, t=30, b=30),
        plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
        xaxis=dict(title='Density', range=xlim, gridcolor='#eee', zerolinecolor='#eee'),
        # No `autorange='reversed'` -- negative z (vitread) is drawn low and positive z
        # (scleral) high, matching `plot_morph.plot_ipl_profile_from_arrays`'s own
        # (unreversed) `ax.set_ylim(zlim)` convention.
        yaxis=dict(title='IPL depth [um]', range=IPL_ZLIM, gridcolor='#eee', zerolinecolor='#eee'),
        title=dict(text=f"IPL depth profile ({len(profiles)}/{len(sub)} with cached profile)",
                   x=0, xanchor='left', font=dict(size=13)),
    )
    return fig, curve_to_key


def message_figure(size, text):
    """A blank Plotly figure of `size` (one of the `*_PLOTLY_SIZE` dicts) with `text`
    centered on it and no axes -- shared by the "no cells match" (`empty_ipl_figure`/
    `empty_morph_figure`) and "still computing" (`loading_ipl_figure`/`loading_morph_figure`)
    placeholders for `ipl_pane`/`morph_pane`."""
    fig = go.Figure()
    fig.update_layout(
        width=size['width'], height=size['height'],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=text, showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)],
    )
    return fig, {}


def empty_ipl_figure():
    """Placeholder for `ipl_pane` when no cells match the current filters -- mirrors
    `empty_morph_figure`."""
    return message_figure(IPL_PLOTLY_SIZE, 'No cells match this selection')


def loading_ipl_figure():
    """Placeholder for `ipl_pane` while `blank_heavy_panes` is (re)computing the current
    selection -- see `redraw_filters`/`redraw_selection`."""
    return message_figure(IPL_PLOTLY_SIZE, 'Loading...')


def plot_overlay_with_mean(ax, time, traces, colors, keys, selected_key, title, vlines=()):
    """Shared drawing logic for the three response-overlay panels: each row's trace in
    `traces` (shape ``(n_cells, time)``, row order matching `keys`/`colors`) plotted
    thin/translucent in its own `colors` entry, plus the across-cell mean +/- SD on top in
    black.

    If `selected_key` (a `df` row index, or `None`) is one of `keys`, all *other* rows are
    dimmed to gray instead of their own colour, and the selected row is redrawn on top,
    bold and at full opacity, so it reads clearly against the rest -- the mean +/- SD is
    also dimmed in that case, so it doesn't visually compete with the highlighted trace.
    """
    highlighted = selected_key is not None and selected_key in keys
    selected_i = keys.get_loc(selected_key) if highlighted else None

    for i, (color, trace) in enumerate(zip(colors, traces)):
        if not highlighted:
            ax.plot(time, trace, color=color, lw=0.8, alpha=0.6, clip_on=False, zorder=1)
        elif i != selected_i:
            ax.plot(time, trace, color='#cccccc', lw=0.6, alpha=0.6, clip_on=False, zorder=1)

    mean = traces.mean(axis=0)
    sd = traces.std(axis=0)
    ax.plot(time, mean, color='black', lw=2, clip_on=False, zorder=2, alpha=0.3 if highlighted else 1.0)
    ax.fill_between(time, mean - sd, mean + sd, color='black', alpha=0.05 if highlighted else 0.15, zorder=0)

    if highlighted:
        ax.plot(time, traces[selected_i], color=colors[selected_i], lw=2.2, alpha=1.0, clip_on=False, zorder=3)

    ax.axhline(0, color='dimgray', ls='--', lw=1)
    for t in vlines:
        ax.axvline(t, color='gray', ls='--', lw=1)
    ax.set(xlabel='Time [s]', ylabel='Norm. Ca.')
    ax.set_title(title, loc='left')


def plot_chirp_overlay(ax, sub, selected_key):
    """Every row's mean-over-trials chirp response (`chirp_average_norm`) overlaid, colour-
    coded by recording time (see `row_color`), or highlighted/dimmed if `selected_key`
    names one of `sub`'s cells (see `plot_overlay_with_mean`). Returns the panel's
    time-axis duration (s), for `scale_panel_widths_by_duration`.

    Assumes a shared time axis/length across rows (true for the released data, same as
    `plot_dataframe.plot_df_chirp_and_bar`)."""
    dt = sub['chirp_average_dt'].iloc[0]
    n_t = sub['chirp_average_norm'].iloc[0].size
    time = np.arange(n_t) * dt
    traces = np.stack(sub['chirp_average_norm'].to_numpy())
    colors = [row_color(row) for _, row in sub.iterrows()]
    plot_overlay_with_mean(ax, time, traces, colors, sub.index, selected_key,
                           "Chirp, mean over trials per cell", vlines=(2, 5, 8, 30))
    return time[-1]


def plot_bar_overlay(ax, sub, selected_key):
    """Every row's moving-bar temporal response (`bar_time_component`, the SVD time kernel
    that collapses across direction and repeats -- same quantity `plot_dataframe.
    plot_df_chirp_and_bar` stacks per-row) overlaid, colour-coded by recording time, or
    highlighted/dimmed if `selected_key` names one of `sub`'s cells. Returns the panel's
    time-axis duration (s), for `scale_panel_widths_by_duration`.

    Assumes a shared time axis/length across rows (true for the released data)."""
    dt = sub['bar_snippets_dt'].iloc[0]
    n_t = sub['bar_time_component'].iloc[0].size
    time = np.arange(n_t) * dt
    traces = np.stack(sub['bar_time_component'].to_numpy())
    colors = [row_color(row) for _, row in sub.iterrows()]
    # Kept short ("Moving bar" rather than e.g. "Moving bar, time component") -- this
    # panel is narrowed by scale_panel_widths_by_duration (~4 s vs. chirp's ~33 s) and
    # sits right next to the direction-tuning polar panel (plot_bar_dir_overlay), so a
    # longer left-aligned title would overflow this panel's own width straight into it.
    plot_overlay_with_mean(ax, time, traces, colors, sub.index, selected_key, "Moving bar",
                           vlines=plot_traces.BAR_STIM_TIMES)
    return time[-1]


# `bar_dir_component`'s 8 entries are ordered by ascending direction (0, 45, 90, ..., 315
# deg), matching `plot_traces.preprocess_mb_snippets`'s own sort order -- not `MB_DIRS`'s
# raw stimulus-presentation order.
SORTED_DIR_RAD = np.deg2rad(sorted(MB_DIRS))


def close_loop(values):
    """Append `values[0]` to the end, so a per-direction curve plotted against
    `SORTED_DIR_RAD` (or with that same +1 appended) draws as a closed loop."""
    return np.append(values, values[0])


def plot_bar_dir_overlay(ax, sub, selected_key):
    """Every row's moving-bar direction-tuning curve (`bar_dir_component` -- precomputed,
    min-max normalized to peak at 1; the same SVD-derived quantity `plot_traces.
    plot_bar_dir` computes live as `dir_component`) overlaid on a shared polar axes,
    colour-coded by recording time, plus the across-cell mean curve on top in black -- or,
    if `selected_key` names one of `sub`'s cells, that cell's curve highlighted/bold and
    every other row (plus the mean) dimmed to gray, matching `plot_overlay_with_mean`."""
    angles = close_loop(SORTED_DIR_RAD)
    highlighted = selected_key is not None and selected_key in sub.index

    for key, row in sub.iterrows():
        if not highlighted:
            ax.plot(angles, close_loop(row['bar_dir_component']), color=row_color(row),
                     lw=0.8, alpha=0.6, clip_on=False, zorder=1)
        elif key != selected_key:
            ax.plot(angles, close_loop(row['bar_dir_component']), color='#cccccc',
                     lw=0.6, alpha=0.6, clip_on=False, zorder=1)

    traces = np.stack(sub['bar_dir_component'].to_numpy())
    mean = close_loop(traces.mean(axis=0))
    ax.plot(angles, mean, color='black', lw=2, clip_on=False, zorder=2, alpha=0.3 if highlighted else 1.0)

    if highlighted:
        sel_row = sub.loc[selected_key]
        ax.plot(angles, close_loop(sel_row['bar_dir_component']), color=row_color(sel_row),
                 lw=2.2, alpha=1.0, clip_on=False, zorder=3)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmin(0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([])
    cardinal_dirs = (0, 90, 180, 270)
    ax.set_xticks(np.deg2rad(cardinal_dirs))
    ax.set_xticklabels([MB_DIRS_SYMBOLS_D_UP[np.argmax(np.array(MB_DIRS) == d)] for d in cardinal_dirs],
                        fontsize=10, fontweight='bold', fontname='DejaVu Sans', color='#999999')
    # Kept short (no "Moving bar,"/"(n=...)" prefix/suffix) -- this panel sits right next
    # to the also-narrow `plot_bar_overlay` time-component panel (see its own comment), so
    # two long left-aligned titles this close together would run into each other.
    ax.set_title("Direction tuning", loc='left')


MC_TEST_INDICES = (0, 59, 118)


def mc_average_trace(row, test_indices=MC_TEST_INDICES):
    """(t_rel, average) -- mean of `row`'s 3 repeated natural-movie (mouse-cam) test-clip
    presentations, using the same snippet extraction as `plot_traces.plot_mc_test_snippets`
    (duplicated here rather than reused, since that function only plots -- it doesn't
    return the averaged trace)."""
    mc_trace = row['mc_pp_trace']
    mc_time = np.arange(mc_trace.size) * row['mc_trace_dt'] + row['mc_trace_t0']
    mc_tt = row['mc_triggertimes']
    mc_tt = np.append(mc_tt, mc_tt[-1] + np.median(np.diff(mc_tt)))

    t_common = None
    snippets = []
    for test_i in test_indices:
        t0, t1 = mc_tt[test_i], mc_tt[test_i + 5]
        ilim = (mc_time >= t0) & (mc_time <= t1)
        t_rel = mc_time[ilim] - t0
        if t_common is None:
            t_common = t_rel
        snippets.append(np.interp(t_common, t_rel, mc_trace[ilim]))
    snippets = np.stack(snippets, axis=1)
    return t_common, snippets.mean(axis=1)


def plot_mc_overlay(ax, sub, selected_key):
    """Every row's mean-over-test-reps natural-movie (mouse-cam) response overlaid,
    colour-coded by recording time, or highlighted/dimmed if `selected_key` names one of
    `sub`'s cells. Returns the panel's time-axis duration (s), for
    `scale_panel_widths_by_duration`."""
    per_row = [mc_average_trace(row) for _, row in sub.iterrows()]
    # A test-clip window's exact sample count can be off by one between rows (float
    # boundary rounding in `mc_average_trace`'s `<=` comparison) -- trim to the shortest
    # so all rows stack into one array.
    min_len = min(len(t) for t, _ in per_row)
    time = per_row[0][0][:min_len]
    traces = np.stack([avg[:min_len] for _, avg in per_row])
    colors = [row_color(row) for _, row in sub.iterrows()]

    first_tt = sub['mc_triggertimes'].iloc[0]
    first_tt = np.append(first_tt, first_tt[-1] + np.median(np.diff(first_tt)))
    vlines = first_tt[MC_TEST_INDICES[0]:MC_TEST_INDICES[0] + 6] - first_tt[MC_TEST_INDICES[0]]

    plot_overlay_with_mean(ax, time, traces, colors, sub.index, selected_key,
                           "Natural movie (mouse cam), mean over 3 test reps", vlines=vlines)
    return time[-1]


def draw_time_colorbar(cax):
    """Horizontal colorbar in `cax` translating `row_color`'s colour back to session-
    elapsed recording time in minutes, increasing left-to-right (matplotlib's default for
    a horizontal bar, `TIME_NORM`'s low/high end at the left/right respectively -- no flip
    needed). Ticked at each field's own recording time."""
    sm = plt.cm.ScalarMappable(cmap=TIME_CMAP, norm=TIME_NORM)
    cb = cax.figure.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Recording time [min]', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    tick_times = list(FIELD_CHIRP_T0.values())
    cb.set_ticks(tick_times)
    cb.set_ticklabels([f"{t / 60:.0f}" for t in tick_times])


def add_bar_dir_panel_and_colorbar(fig, ax_bar, ax_chirp, sub, selected_key, gap=0.02, cbar_height_frac=0.25):
    """Add the moving-bar direction-tuning polar overlay (`plot_bar_dir_overlay`) and the
    time colorbar (`draw_time_colorbar`) side by side, in the width `scale_panel_widths_
    by_duration` freed up to the right of the (duration-narrowed) moving-bar panel: the
    polar panel first, sized to a physical square (using `TRACES_FIGSIZE`'s aspect ratio)
    matching the row's height, then the colorbar in whatever's left -- considerably
    shorter than when it had that whole freed-up width to itself. Must be called after
    `scale_panel_widths_by_duration`, since `ax_bar`'s width (and thus the free space) is
    only finalized there.
    """
    pos_bar = ax_bar.get_position()
    pos_chirp = ax_chirp.get_position()
    right_edge = pos_chirp.x0 + pos_chirp.width

    polar_width = pos_bar.height * (TRACES_FIGSIZE[1] / TRACES_FIGSIZE[0])
    polar_x0 = pos_bar.x0 + pos_bar.width + gap
    ax_polar = fig.add_axes([polar_x0, pos_bar.y0, polar_width, pos_bar.height], projection='polar')
    plot_bar_dir_overlay(ax_polar, sub, selected_key)

    # matplotlib's PolarAxes.clear() hardcodes its title to y=1.05 (axes-fraction, to clear
    # the theta=0 tick label we rely on -- see `plot_bar_dir_overlay`'s cardinal-direction
    # ticks), sitting noticeably higher than ax_bar's own (Cartesian, default-positioned)
    # title even at the identical nominal box position. Measure the actual rendered gap
    # and shrink ax_polar's box from the top by that amount so the two titles line up.
    fig.canvas.draw()
    delta_px = ax_polar.title.get_window_extent().y0 - ax_bar.title.get_window_extent().y0
    if delta_px > 0:
        delta_frac = delta_px / (fig.dpi * TRACES_FIGSIZE[1])
        pos_polar = ax_polar.get_position()
        ax_polar.set_position([pos_polar.x0, pos_polar.y0, pos_polar.width, pos_polar.height - delta_frac])

    cax_x0 = polar_x0 + polar_width + gap
    cax_width = right_edge - cax_x0
    cax_height = pos_bar.height * cbar_height_frac
    cax_y0 = pos_bar.y0 + (pos_bar.height - cax_height) / 2
    cax = fig.add_axes([cax_x0, cax_y0, cax_width, cax_height])
    draw_time_colorbar(cax)


def scale_panel_widths_by_duration(response_axes, durations):
    """Shrink each of `response_axes`' rendered width so that seconds-per-inch is the same
    across all of them, instead of each stimulus filling the same panel width regardless
    of how long it actually lasted (chirp ~33 s vs. moving bar ~4 s vs. natural movie
    ~25 s) -- so panel width becomes directly readable as relative duration. `durations`
    (s) must be in the same order as `response_axes`; each axes keeps its own left edge
    (`x0`), so t=0 stays aligned across panels, and the longest-duration one keeps its
    full original width (the new common seconds-per-inch scale)."""
    max_duration = max(durations)
    for ax, duration in zip(response_axes, durations):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * (duration / max_duration), pos.height])


def render_traces_fig(sub, selected_key):
    """The response-panel figure (chirp/moving-bar/natural-movie overlays, the moving-bar
    direction-tuning polar overlay, and the shared time colorbar) for `sub`, with
    `selected_key`'s cell (a `df` row index, or `None`) highlighted across all of them.
    Morphology is a separate (Plotly) pane -- see `make_morph_figure` -- not part of this
    figure, so unlike the very first version of this tool, there's no need to match this
    figure's height to anything; the 3 response panels simply divide it evenly.
    """
    fig = plt.figure(figsize=TRACES_FIGSIZE)
    # hspace is relative to each panel's own (now much shorter, see TRACES_FIGSIZE) height,
    # but the title/tick-label text between panels needs roughly the same *absolute* room
    # regardless -- so this needs to be considerably larger than it would for a taller figure.
    gs = fig.add_gridspec(3, 1, hspace=1.3)
    ax_chirp = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_mc = fig.add_subplot(gs[2, 0])

    if len(sub) == 0:
        for ax in (ax_chirp, ax_bar, ax_mc):
            ax.text(0.5, 0.5, 'No cells match this selection', ha='center', va='center')
            ax.axis('off')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # see the other tight_layout call below
            fig.tight_layout()
        return fig

    chirp_duration = plot_chirp_overlay(ax_chirp, sub, selected_key)
    bar_duration = plot_bar_overlay(ax_bar, sub, selected_key)
    mc_duration = plot_mc_overlay(ax_mc, sub, selected_key)
    with warnings.catch_warnings():
        # The explicit `hspace` above makes matplotlib treat every axes in this
        # gridspec as "not from an unmodified gridspec" (see `matplotlib._tight_layout.
        # get_subplotspec_list`'s `locally_modified_subplot_params` check), which tight_
        # layout warns about generically -- checked empirically that layout still comes
        # out correct regardless (same as interactive_explorer.py's render_heatmap).
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout()
    scale_panel_widths_by_duration([ax_chirp, ax_bar, ax_mc], [chirp_duration, bar_duration, mc_duration])
    add_bar_dir_panel_and_colorbar(fig, ax_bar, ax_chirp, sub, selected_key)
    return fig


def loading_traces_fig():
    """Placeholder for `traces_pane` while `blank_heavy_panes` is (re)computing the current
    selection -- see `redraw_filters`/`redraw_selection`."""
    fig = plt.figure(figsize=TRACES_FIGSIZE)
    fig.text(0.5, 0.5, 'Loading...', ha='center', va='center')
    return fig


# Rendered CSS pixel width of `cluster_prob_bar_html`'s bar itself (excludes the field/ROI/
# cell-type columns, and the top-1/top-2 group-name text to its right) -- a bit wider than
# a bare percentage bar would need, so a small segment's in-bar cluster-name label (e.g.
# '22b') still has room to stay legible.
CLUSTER_PROB_BAR_WIDTH_PX = 260


def cluster_prob_bar_html(row, correct_by_cellclass=False, height_px=18, width_px=CLUSTER_PROB_BAR_WIDTH_PX):
    """A horizontal stacked bar -- 2 segments for `row`'s top-2 predicted functional
    *clusters* (`top2_cluster_probs`, e.g. `'12b'`), each labelled with its cluster name and
    coloured continuously by cluster identity (`CLUSTER_NAME_TO_COLOR_HEX`, not discretely by
    supergroup like tab 1's per-cell prediction bars/`SUPERGROUP_COLORS`), plus a gray
    remainder covering the rest of the probability mass -- followed by text naming the top-1
    predicted functional *group* (`top2_group_probs`, e.g. `'On-Off DS 1'`, the human-readable
    "functional cell type" -- a separate, coarser ranking than the cluster-level bar, not just
    the bar's top cluster's parent group), then the top-2 group in gray in brackets.
    `correct_by_cellclass` is forwarded to both.
    """
    cluster_segments = top2_cluster_probs(row, correct_by_cellclass=correct_by_cellclass)
    spans = []
    for name, supergroup, prob in cluster_segments:
        color = CLUSTER_NAME_TO_COLOR_HEX[name]
        spans.append(
            f'<div title="Cluster {name} ({supergroup}): {prob:.0%}" '
            f'style="flex:{max(prob, 0.001)}; background:{color}; color:white; '
            f'display:flex; align-items:center; justify-content:center; overflow:hidden; '
            f'font-size:10px; white-space:nowrap;">{name}</div>'
        )
    remainder = 1.0 - sum(p for _, _, p in cluster_segments)
    if remainder > 0:
        spans.append(f'<div style="flex:{remainder}; background:#e0e0e0;"></div>')
    bar_html = (f'<div style="display:flex; width:{width_px}px; height:{height_px}px; '
                f'border-radius:3px; overflow:hidden;">{"".join(spans)}</div>')

    group_segments = top2_group_probs(row, correct_by_cellclass=correct_by_cellclass)
    if group_segments:
        label_html = f'<span style="margin-left:8px;">{group_segments[0][0]}</span>'
        if len(group_segments) > 1:
            label_html += f' <span style="color:#999999;">({group_segments[1][0]})</span>'
    else:
        label_html = ''
    return f'<div style="display:flex; align-items:center;">{bar_html}{label_html}</div>'


def render_cell_list_html(sub, selected_key, correct_by_cellclass=False):
    """HTML table listing every row of `sub` -- field, ROI number, morphological cell type,
    and its predicted-type probability bar (top-2 clusters on the bar, top-2 functional
    groups as text -- `cluster_prob_bar_html`) -- below the response-overlay panels
    (`render_traces_fig`).

    `selected_key` (a `df` row index, or `None`) highlights that row (light-blue background),
    matching the highlight-on-selection convention used across the other panels, so the
    selected cell can also be found by scrolling this list. `correct_by_cellclass` is
    forwarded to `cluster_prob_bar_html`.

    Rows that fail the response-quality filter (`row['qfilt']` False -- i.e. would be
    excluded if `qfilt_dropdown` were set to only show quality-filtered cells, regardless of
    its actual current value here) have their Field/ROI/Cell type text grayed out, so a cell
    whose predicted-type bar can't be trusted as much stands out at a glance without needing
    to actually filter it out of the list."""
    rows_html = []
    for key, row in sub.iterrows():
        bg = '#eaf2fb' if key == selected_key else 'transparent'
        name_color = 'inherit' if row['qfilt'] else '#999999'
        name_title = '' if row['qfilt'] else ' title="Did not pass the response-quality filter (qfilt)"'
        rows_html.append(
            f'<tr style="background:{bg};">'
            f'<td style="padding:2px 8px; color:{name_color};"{name_title}>{row["field"]}</td>'
            f'<td style="padding:2px 8px; color:{name_color};"{name_title}>{row["roi_id"]}</td>'
            f'<td style="padding:2px 8px; color:{name_color};"{name_title}>{row["Cell Type"]}</td>'
            f'<td style="padding:2px 8px;">{cluster_prob_bar_html(row, correct_by_cellclass=correct_by_cellclass)}</td>'
            f'</tr>'
        )
    if not rows_html:
        return '<p>No cells match this selection.</p>'
    return (
        '<div style="max-height:300px; overflow-y:auto;">'
        '<table style="border-collapse:collapse; font-size:12px; width:100%;">'
        '<thead><tr style="text-align:left;">'
        '<th style="padding:2px 8px;">Field</th><th style="padding:2px 8px;">ROI</th>'
        '<th style="padding:2px 8px;">Cell type</th>'
        '<th style="padding:2px 8px;">Predicted functional type (cluster bar; group name)</th>'
        '</tr></thead><tbody>' + ''.join(rows_html) + '</tbody></table></div>'
    )


# %% [markdown]
# ### Widgets

# %%
# Fixed, fairly narrow widths -- laid out in two rows above the panels (rather than a column
# to their left) so they don't eat into the panels' own width. Row 1 (which cells match --
# `ct_field_dropdown`.. `qfilt_dropdown`, see `selected_cells`) then row 2 (how to display/
# highlight that already-selected group -- `morph_view_dropdown`/`cell_dropdown`/
# `add_to_table_button`).
ct_field_dropdown = pn.widgets.Select(name='Field', options=FIELD_OPTIONS, value=ALL_FIELDS, width=140)
cellclass_dropdown = pn.widgets.Select(name='Cell class', options=CELL_CLASS_OPTIONS, value='RGC', width=140)
celltype_dropdown = pn.widgets.Select(name='Morphological cell type',
                                       options=CELL_TYPES_BY_CLASS['RGC'], value=CELL_TYPES_BY_CLASS['RGC'][0],
                                       width=260)
# A `Select`, not a `Checkbox`, to stay visually consistent with the other cell-selection
# dropdowns on this row.
QFILT_OPTIONS = {'Only quality-filtered (qfilt)': True, 'All cells': False}
qfilt_dropdown = pn.widgets.Select(name='Quality filter', options=QFILT_OPTIONS, value=True, width=200)
# Switches `morph_pane` between full skeletons and just their dendritic-hull outline (see
# `make_morph_figure`'s `view_mode`) -- a display choice, not a data filter, so changing it
# doesn't reset `selected_key` (unlike the filter widgets above).
morph_view_dropdown = pn.widgets.Select(name='Morphology view', options=MORPH_VIEW_OPTIONS,
                                        value=MORPH_VIEW_OPTIONS[0], width=160)
# Options rebuilt on every `redraw_filters` (see `cell_dropdown_options`) to match the
# current filtered `sub` -- an alternative, keyboard/list-friendly way to pick the
# highlighted cell alongside clicking it in `morph_pane`/`ipl_pane`.
cell_dropdown = pn.widgets.Select(name='Highlight cell', options={'(none)': None}, value=None, width=200)
# Same toggle/purpose as tab 1's `cellclass_correction_dropdown`, but independent (this
# tab's own filters/selection are independent of tab 1's) -- applies
# `baden16_utils.correct_probs_by_cellclass` to `cell_list_pane`'s predicted-type bars (see
# `top2_cluster_probs`/`top2_group_probs`). Placed (see `celltype_layout` below) between
# `traces_pane` and `cell_list_pane`, right next to the bars it controls, rather than up in
# the cell-selection rows.
CT_CELLCLASS_CORRECTION_OPTIONS = {'Raw model prediction': False, 'Corrected by Cell Class (RGC vs. dAC)': True}
ct_cellclass_correction_dropdown = pn.widgets.Select(name='Predicted type', options=CT_CELLCLASS_CORRECTION_OPTIONS,
                                                  value=False, width=220)
add_to_table_button = pn.widgets.Button(name='Add to table', button_type='primary', width=120)

# The currently-selected cell (a `df` row index, or `None`) -- click a cell in `morph_pane`/
# `ipl_pane`, or pick it from `cell_dropdown`, to set it; click the background to clear it.
# Any filter change resets it too (`on_filter_change`), since a selection made under one
# field/cell-type/class combination wouldn't necessarily mean anything under another.
selected_key = None
# `curve_to_key[curveNumber]` for the currently-shown `morph_pane`/`ipl_pane` figures --
# rebuilt by `set_morph_pane`/`set_ipl_pane` on every render (see `make_morph_figure`/
# `make_ipl_figure`), read by `on_morph_click`/`on_ipl_click`.
CURVE_TO_KEY = {}
IPL_CURVE_TO_KEY = {}
# Guards `cell_dropdown`'s 'value' watcher while `sync_cell_dropdown` sets it programmatically
# (to reflect a selection made by clicking a panel, or to reset it on a filter change) --
# without this, that assignment would itself fire `on_cell_dropdown_change`, which is only
# meant to react to the user's own picks.
_updating_cell_dropdown = False


def cell_dropdown_options(sub):
    """``{label: df row index}`` for `cell_dropdown`, one entry per row of `sub` (field + ROI
    number, matching the hover text in `make_morph_figure`/`make_ipl_figure`), plus a leading
    '(none)' entry (`None`) for "nothing selected"."""
    options = {'(none)': None}
    for key, row in sub.iterrows():
        options[f"{row['field']} - ROI {row['roi_id']}"] = key
    return options


def empty_morph_figure():
    """Placeholder for `morph_pane` when no cells match the current filters -- unlike
    `render_traces_fig`, `make_morph_figure` assumes at least one cell (it fits the view to
    the cells' own extent), so this is a separate, trivial figure rather than a code path
    through it."""
    return message_figure(MORPH_PLOTLY_SIZE, 'No cells match this selection')


def loading_morph_figure():
    """Placeholder for `morph_pane` while it's being (re)computed -- see `blank_heavy_panes`/
    `redraw_morph_view`."""
    return message_figure(MORPH_PLOTLY_SIZE, 'Loading...')


_initial_sub = selected_cells(ct_field_dropdown.value, celltype_dropdown.value, qfilt_dropdown.value,
                               cellclass_dropdown.value)
if len(_initial_sub):
    _initial_morph_fig, CURVE_TO_KEY = make_morph_figure(_initial_sub, ct_field_dropdown.value, selected_key,
                                                         view_mode=morph_view_dropdown.value)
    _initial_ipl_fig, IPL_CURVE_TO_KEY = make_ipl_figure(_initial_sub, selected_key)
else:
    _initial_morph_fig, CURVE_TO_KEY = empty_morph_figure()
    _initial_ipl_fig, IPL_CURVE_TO_KEY = empty_ipl_figure()
cell_dropdown.options = cell_dropdown_options(_initial_sub)

morph_pane = pn.pane.Plotly(_initial_morph_fig, margin=0)
ipl_pane = pn.pane.Plotly(_initial_ipl_fig, margin=0)
traces_pane = pn.pane.Matplotlib(render_traces_fig(_initial_sub, selected_key), tight=True, format='png', dpi=110,
                                  margin=0)
title_pane = pn.pane.Markdown(
    f"### {cellclass_dropdown.value} -- {celltype_dropdown.value} -- "
    f"{ct_field_dropdown.value} -- n={len(_initial_sub)}",
    margin=(0, 0, 10, 0),
)
# Shows the selected cell's field/ROI once one is picked in `morph_pane`/`ipl_pane` (see
# `on_morph_click`/`on_ipl_click`) -- empty (rather than a placeholder string) when nothing
# is selected, matching the tool's existing "no selection" states elsewhere (e.g. all cells
# at full opacity in make_morph_figure).
selection_pane = pn.pane.Markdown('', margin=(0, 0, 0, 10))
# Per-cell list (field/ROI/cell type + top-2 predicted functional-group probability bar)
# below the response-overlay panels -- see `render_cell_list_html`.
cell_list_pane = pn.pane.HTML(
    render_cell_list_html(_initial_sub, selected_key, correct_by_cellclass=ct_cellclass_correction_dropdown.value),
    margin=(10, 0, 0, 0))

# The `sub` behind the panels currently on screen -- set only by `redraw_filters` (the only
# thing that can change which cells match). Every other redraw path (`redraw_selection`/
# `redraw_morph_view`/`redraw_cell_list`) reuses this instead of recomputing it via
# `selected_cells`, since none of the widgets they react to change which cells match.
current_sub = _initial_sub


def set_morph_pane(sub, field, selected_key):
    """(Re)render `morph_pane` for `sub`/`selected_key` -- the only thing that depends on
    `morph_view_dropdown` too (read directly, not passed in, same as the other `set_*_pane`
    helpers below read their own widget dependencies)."""
    global CURVE_TO_KEY
    if len(sub):
        fig, CURVE_TO_KEY = make_morph_figure(sub, field, selected_key, view_mode=morph_view_dropdown.value)
    else:
        fig, CURVE_TO_KEY = empty_morph_figure()
    morph_pane.object = fig


def set_ipl_pane(sub, selected_key):
    global IPL_CURVE_TO_KEY
    if len(sub):
        fig, IPL_CURVE_TO_KEY = make_ipl_figure(sub, selected_key)
    else:
        fig, IPL_CURVE_TO_KEY = empty_ipl_figure()
    ipl_pane.object = fig


def set_traces_pane(sub, selected_key):
    old_fig = traces_pane.object
    traces_pane.object = render_traces_fig(sub, selected_key)
    plt.close(old_fig)


def set_title_pane(field, sub):
    title_pane.object = (f"### {cellclass_dropdown.value} -- {celltype_dropdown.value} -- "
                         f"{field} -- n={len(sub)}")


def set_selection_pane(sub, selected_key):
    if selected_key is not None and selected_key in sub.index:
        sel_row = sub.loc[selected_key]
        selection_pane.object = f"**Field:** {sel_row['field']}  \n**ROI:** {sel_row['roi_id']}"
    else:
        selection_pane.object = ''


def set_cell_list_pane(sub, selected_key):
    cell_list_pane.object = render_cell_list_html(sub, selected_key,
                                                  correct_by_cellclass=ct_cellclass_correction_dropdown.value)


def sync_cell_dropdown(sub, selected_key, rebuild_options):
    """Reflect `selected_key` in `cell_dropdown.value` (set here whether the selection came
    from `cell_dropdown` itself, a panel click, or a filter reset) -- guarded so this
    assignment doesn't recursively trigger `on_cell_dropdown_change`. `rebuild_options` is
    only needed from `redraw_filters`: `sub`'s own membership only changes there, everywhere
    else `cell_dropdown`'s existing options already match `sub`."""
    global _updating_cell_dropdown
    _updating_cell_dropdown = True
    try:
        value = selected_key if selected_key in sub.index else None
        if rebuild_options:
            cell_dropdown.param.update(options=cell_dropdown_options(sub), value=value)
        else:
            cell_dropdown.value = value
    finally:
        _updating_cell_dropdown = False


async def blank_heavy_panes():
    """Blank `morph_pane`/`ipl_pane`/`traces_pane` to a "Loading..." placeholder -- the only
    panels expensive enough (`make_morph_figure` aligns a skeleton per cell, `make_ipl_figure`
    reads a cached profile per cell, `render_traces_fig` replots every cell's response traces)
    to need one -- then yields to the event loop (`await asyncio.sleep(...)`) just long enough
    for Panel to actually push that blanked state to the browser before the recompute that
    follows, so a change visibly does *something* immediately rather than silently appearing
    to hang until the new figures are ready."""
    morph_pane.object = loading_morph_figure()[0]
    ipl_pane.object = loading_ipl_figure()[0]
    old_traces_fig = traces_pane.object
    traces_pane.object = loading_traces_fig()
    plt.close(old_traces_fig)
    await asyncio.sleep(0.01)


async def redraw_filters():
    """Triggered when a cell-*selection* filter (field/cell class/cell type/qfilt) changes --
    the only thing that changes `sub` (which cells match), so it's the only redraw that
    touches every panel. `title_pane`/`selection_pane`/`cell_dropdown` are cheap (no file I/O
    or per-cell skeleton alignment) and set directly from the freshly computed `sub`, with no
    "Loading..." placeholder. `cell_list_pane` is *not* cheap to skip a placeholder for even
    though it's fast to actually recompute: a different `sub` means different cells/
    morphologies, and `cell_list_pane`'s predicted-type bars are per-cell, so they're
    guaranteed to change too -- showing the *old* group's predictions while the rest of the
    view is blanked/updating would be actively misleading, not just stale. So it's blanked
    up front alongside `morph_pane`/`ipl_pane`/`traces_pane` (`blank_heavy_panes`), even
    though it isn't itself "heavy"."""
    global current_sub

    cell_list_pane.object = '<p>Loading...</p>'
    await blank_heavy_panes()

    field = ct_field_dropdown.value
    sub = selected_cells(field, celltype_dropdown.value, qfilt_dropdown.value, cellclass_dropdown.value)
    current_sub = sub

    set_morph_pane(sub, field, selected_key)
    set_ipl_pane(sub, selected_key)
    set_traces_pane(sub, selected_key)
    set_title_pane(field, sub)
    set_selection_pane(sub, selected_key)
    set_cell_list_pane(sub, selected_key)
    sync_cell_dropdown(sub, selected_key, rebuild_options=True)


async def redraw_selection():
    """Triggered when only `selected_key` changes (a panel click or a `cell_dropdown` pick)
    -- `sub` (which cells match the current filters) is unchanged, so it's reused from
    `current_sub` rather than recomputed, and `title_pane` (whose text doesn't depend on
    `selected_key`) is left untouched entirely. `morph_pane`/`ipl_pane`/`traces_pane` still
    redraw *every* cell's geometry, not just the (de)selected one (see `make_morph_figure`
    etc.), so they're still worth blanking first via `blank_heavy_panes`; `selection_pane`/
    `cell_list_pane` are cheap and set directly."""
    sub = current_sub
    await blank_heavy_panes()

    set_morph_pane(sub, ct_field_dropdown.value, selected_key)
    set_ipl_pane(sub, selected_key)
    set_traces_pane(sub, selected_key)
    set_selection_pane(sub, selected_key)
    set_cell_list_pane(sub, selected_key)
    sync_cell_dropdown(sub, selected_key, rebuild_options=False)


async def redraw_morph_view():
    """Triggered by `morph_view_dropdown` -- only `morph_pane` depends on it (`view_mode`);
    nothing else is touched."""
    morph_pane.object = loading_morph_figure()[0]
    await asyncio.sleep(0.01)
    set_morph_pane(current_sub, ct_field_dropdown.value, selected_key)


def redraw_cell_list():
    """Triggered by `ct_cellclass_correction_dropdown` -- only `cell_list_pane`'s predicted-type
    bars depend on it, via a pure in-memory recompute (`top2_cluster_probs`/
    `top2_group_probs` over data already loaded, no file I/O), fast enough not to need a
    loading placeholder or to be async."""
    set_cell_list_pane(current_sub, selected_key)


async def on_filter_change(event=None):
    global selected_key
    selected_key = None
    await redraw_filters()


NO_TYPES_PLACEHOLDER = '(no cell types for this class)'


def on_cellclass_change(event):
    """Switching cell class swaps the type list to that class's own types (see
    `CELL_TYPES_BY_CLASS`) and jumps to its first entry -- set together via one atomic
    Param update so there's no intermediate state where `celltype_dropdown.value` is
    invalid for its new `options`. 'Other' has none in the released data (every
    `Fragment`/missing-class row also has a missing `Cell Type`) -- fall back to a
    placeholder rather than crashing on an empty options list; `selected_cells` then
    naturally matches zero rows against it. Doesn't call `redraw_filters` itself: setting
    `celltype_dropdown`'s value fires its own 'value' watcher (`on_filter_change`), which
    does."""
    options = CELL_TYPES_BY_CLASS[event.new] or [NO_TYPES_PLACEHOLDER]
    celltype_dropdown.param.update(options=options, value=options[0])


async def on_morph_click(event):
    """Resolve a click in `morph_pane` to a cell (or the background) via `CURVE_TO_KEY`,
    and redraw with that cell selected -- or cleared, for a background click (`CURVE_TO_
    KEY` maps background/decoration traces to `None`, same value as "nothing selected").
    A click that hits no trace at all leaves `click_data` empty and is a no-op; `make_
    morph_figure`'s invisible background marker grid means that should be rare for clicks
    inside the plotted range.
    """
    global selected_key
    click_data = event.new
    morph_pane.click_data = None  # reset so a stale replay of this event is a no-op
    if not click_data or not click_data.get('points'):
        return
    point = click_data['points'][0]
    selected_key = CURVE_TO_KEY.get(point.get('curveNumber', 0))
    await redraw_selection()


async def on_ipl_click(event):
    """Same click-to-select behaviour as `on_morph_click`, but for `ipl_pane` (via
    `IPL_CURVE_TO_KEY`) -- so a cell can be selected either from its morphology or from its
    IPL depth-profile curve."""
    global selected_key
    click_data = event.new
    ipl_pane.click_data = None  # reset so a stale replay of this event is a no-op
    if not click_data or not click_data.get('points'):
        return
    point = click_data['points'][0]
    selected_key = IPL_CURVE_TO_KEY.get(point.get('curveNumber', 0))
    await redraw_selection()


async def on_cell_dropdown_change(event):
    """Set `selected_key` from the user's own pick in `cell_dropdown` -- a keyboard/list-
    friendly alternative to clicking the cell in `morph_pane`/`ipl_pane`. Ignored while
    `sync_cell_dropdown` is itself updating `cell_dropdown.value` to reflect a selection made
    some other way (`_updating_cell_dropdown`), since that's not a new user choice."""
    global selected_key
    if _updating_cell_dropdown:
        return
    selected_key = event.new
    await redraw_selection()


async def on_morph_view_change(event):
    await redraw_morph_view()


def on_ct_cellclass_correction_change(event):
    redraw_cell_list()


ct_field_dropdown.param.watch(on_filter_change, 'value')
cellclass_dropdown.param.watch(on_cellclass_change, 'value')
celltype_dropdown.param.watch(on_filter_change, 'value')
qfilt_dropdown.param.watch(on_filter_change, 'value')
# A display choice, not a filter -- redraws (to pick up the new view_mode) without
# resetting selected_key, unlike the filter widgets above.
morph_view_dropdown.param.watch(on_morph_view_change, 'value')
morph_pane.param.watch(on_morph_click, 'click_data')
ipl_pane.param.watch(on_ipl_click, 'click_data')
cell_dropdown.param.watch(on_cell_dropdown_change, 'value')
# Only `cell_list_pane`'s bars depend on this (see `redraw_cell_list`) -- it doesn't reset
# `selected_key`, since toggling the correction isn't a filter change.
ct_cellclass_correction_dropdown.param.watch(on_ct_cellclass_correction_change, 'value')

# %% [markdown]
# #### "Add to table" logging
#
# Clicking `add_to_table_button` opens `log_modal`, prompting for a free-text comment
# (`log_comment_input`) about the currently-selected cell; `log_save_button` appends that
# row (field/ROI/cell type/qfilt, plus the comment) to `LOG_TABLE_PATH` and closes the
# modal. Nothing is written if no cell is selected (`log_status_pane` explains why instead).

# %%
log_comment_input = pn.widgets.TextAreaInput(name='Comment', placeholder='Enter a comment for this cell...',
                                              auto_grow=True, rows=3)
log_save_button = pn.widgets.Button(name='Save', button_type='primary', width=100)
log_status_pane = pn.pane.Markdown('', margin=(0, 0, 0, 5))
log_modal = pn.Modal(pn.Column(log_status_pane, log_comment_input, log_save_button), width=400)

# The cell (a `df` row index) `log_modal` is currently prompting a comment for -- captured
# from `selected_key` when the modal is opened, since `selected_key` itself could in
# principle change while the modal is up (e.g. a stray background click).
_log_target_key = None


def on_add_to_table_click(event):
    """Open `log_modal` to collect a comment for the currently-selected cell -- or, with no
    cell selected, show that as an inline message instead of opening the modal, since there
    would be nothing to log a comment against."""
    global _log_target_key
    if selected_key is None:
        log_status_pane.object = "*Select a cell first (click it in the morphology panel).*"
        log_modal.open = True
        return
    _log_target_key = selected_key
    log_status_pane.object = ''
    log_comment_input.value = ''
    log_modal.open = True


def on_log_save_click(event):
    """Append one row for `_log_target_key` to `LOG_TABLE_PATH` and close `log_modal`. Reads
    the sheet fresh from disk (rather than keeping it in memory) so concurrently-running
    copies of this tool, or manual edits, aren't clobbered by a stale in-memory copy."""
    if _log_target_key is None:
        log_modal.open = False
        return
    row = df.loc[_log_target_key]
    log_df = pd.read_excel(LOG_TABLE_PATH)
    new_row = pd.DataFrame([{
        'field': row['field'],
        'ROI number': row['roi_id'],
        'morphological cell type': row['Cell Type'],
        'passed quality filter': bool(row['qfilt']),
        'comment': log_comment_input.value,
    }])
    log_df = pd.concat([log_df, new_row], ignore_index=True)
    log_df.to_excel(LOG_TABLE_PATH, index=False)
    log_modal.open = False


add_to_table_button.on_click(on_add_to_table_click)
log_save_button.on_click(on_log_save_click)

celltype_layout = pn.Column(
    title_pane,
    # Row 1: cell selection (which cells match). Row 2: display/highlight options for the
    # already-selected group, below it.
    pn.Row(ct_field_dropdown, cellclass_dropdown, celltype_dropdown, qfilt_dropdown),
    pn.Row(morph_view_dropdown, cell_dropdown, add_to_table_button),
    # `margin=0` on the panes themselves plus here on their Row -- morphology and the
    # response panels sit right next to each other, no gap. `selection_pane`/`ipl_pane` sit
    # below the morphology pane, and `ct_cellclass_correction_dropdown`/`cell_list_pane` below
    # `traces_pane` (left-aligned with it, not spanning the full row) -- the dropdown sits
    # directly between the traces and the predicted-type bars it controls -- each in its own
    # column, so they don't affect that spacing.
    pn.Row(pn.Column(morph_pane, selection_pane, ipl_pane, margin=0),
          pn.Column(traces_pane, ct_cellclass_correction_dropdown, cell_list_pane, margin=0), margin=0),
    log_modal,
    margin=(0, 0, 0, 32),
)

# %%
app = pn.Tabs(
    ('Cell explorer', cell_explorer_layout),
    ('Stimulus timeline', timeline_layout),
    ('Cell type explorer', celltype_layout),
)
app.servable()
