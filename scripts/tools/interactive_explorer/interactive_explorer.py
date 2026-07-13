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
# Pick a 2p ROI, either from the field/ROI dropdowns or by clicking a dot in
# the XY scatter, and see its EM skeleton (also overlaid on the XY scatter
# itself) next to its direction-selectivity/morphology, chirp, and mouse-cam
# responses.
#
# This is a **quick first cut**, deliberately not fancy yet. Not included
# here (see `BRAINSTORM.md` in this folder for the full plan):
# - the session-timeline slider
# - the light-exposure/stimulus-history overlay
# - the stimulus footprint drawn behind the cell mosaic
#
# Built with [Panel](https://panel.holoviz.org/) rather than `ipywidgets`
# (see `BRAINSTORM.md` for why) -- two ways to run this:
# - Opened in Jupyter (`uv run jupyter lab`, right-click -> Open With ->
#   Notebook) for cell-by-cell exploration.
# - As a standalone local app, no Jupyter needed:
#   `uv run panel serve scripts/tools/interactive_explorer/interactive_explorer.py --show`

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import plotly.graph_objects as go

from eyewire2_functional_analysis import data_loader, plot_morph, plot_traces, registration
from eyewire2_functional_analysis.space_mapping import align_and_place_skel
from eyewire2_functional_analysis.stimulus import stim_outlines

pn.extension('plotly')

# %% [markdown]
# ## Load data
#
# Restrict to ROIs that matched a proofread EM cell (`load_df_rois_morph`).
# Not all of those have an EM skeleton file on disk (`add_skels` leaves
# `skel` as `None` when the SWC file is missing) -- those are still
# selectable, just without a morphology panel (see `render_cell` below).

# %%
df_rois = data_loader.load_df_rois()
df_fields = data_loader.load_df_fields()
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = data_loader.add_skels(df, inplace=True)
df = df.reset_index(drop=True)

n_with_skel = df['skel'].notnull().sum()
print(f"{n_with_skel} / {len(df)} EM-matched ROI(s) have a skeleton file; the rest are still selectable")

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
reg = registration.load_registration(REG_FILE)

df['label'] = df.apply(lambda r: f"{r['field']} - ROI {r['roi_id']}\n{r['Cell Type']}", axis=1)

# Fixed XY range for the cell-picker scatter, from the ROI cloud alone (with
# some padding) -- kept constant regardless of which cell's skeleton is
# overlaid, so a large skeleton gets cropped rather than the view zooming out.
_x_pad = 0.1 * (df['temporal_nasal_pos_um'].max() - df['temporal_nasal_pos_um'].min())
_y_pad = 0.1 * (df['ventral_dorsal_pos_um'].max() - df['ventral_dorsal_pos_um'].min())
X_RANGE = (df['temporal_nasal_pos_um'].min() - _x_pad, df['temporal_nasal_pos_um'].max() + _x_pad)
Y_RANGE = (df['ventral_dorsal_pos_um'].min() - _y_pad, df['ventral_dorsal_pos_um'].max() + _y_pad)

# Morphology-panel scale bar / minimum zoom, so a 100 um scale bar always
# fits comfortably even for a small cell.
MORPH_SCALE_BAR_UM = 100
MORPH_MARGIN_UM = 10
MORPH_MIN_RAD_UM = MORPH_SCALE_BAR_UM / 2 + MORPH_MARGIN_UM

FIELDS = sorted(df['field'].unique())

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
# ## Cell picker (scatter + field/ROI/stimulus dropdowns) and detail panel

# %%
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


def make_scatter_figure(selected_idx=None, stim_type='Chirp'):
    """Build the XY cell-picker scatter, highlighting `selected_idx`, overlaying its skeleton, and
    drawing `stim_type`'s footprint at the selected ROI's field centre.

    Always emits exactly 4 traces (ROI dots, skeleton, stimulus footprint,
    bar-direction outlines -- empty traces when there's nothing to overlay)
    so the figure's trace count never changes between renders. Panel/Plotly's
    `Plotly.react`-based update otherwise gets confused about which trace is
    which when the trace count changes from one selection to the next, which
    was causing stray/delayed click events to resolve against the wrong point.
    """
    colors = ['steelblue'] * len(df)
    sizes = [8] * len(df)
    if selected_idx is not None:
        colors[selected_idx] = 'crimson'
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


def render_cell(idx):
    """Detail figure for row `idx`: DS-on-morph (left, full height), chirp (top right), mouse-cam snippet (bottom right)."""
    row = df.iloc[idx]
    fig = plt.figure(figsize=(13, 5))
    try:
        gs = fig.add_gridspec(2, 3, width_ratios=(1, 1.2, 1.1))

        ax_morph = fig.add_subplot(gs[:, 0])
        if row['skel'] is not None:
            plot_morph.plot_morph(ax_morph, row, reg=reg, rad=None, min_rad=MORPH_MIN_RAD_UM, margin=MORPH_MARGIN_UM,
                            scale_bar_um=MORPH_SCALE_BAR_UM, annotate_orientation=True)
        else:
            ax_morph.text(0.5, 0.5, f"no skeleton found for {row['Latest SegID']}",
                          ha='center', va='center', wrap=True)
            ax_morph.axis('off')
        ax_morph.set_title(row['label'])

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

        fig.tight_layout()
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

scatter_pane = pn.pane.Plotly(make_scatter_figure(selected_idx=roi_dropdown.value, stim_type=stim_dropdown.value))
detail_pane = pn.pane.Matplotlib(render_cell(roi_dropdown.value), tight=True, format='png', dpi=100)


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
    detail_pane.object = render_cell(idx)
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


field_dropdown.param.watch(on_field_change, 'value')
roi_dropdown.param.watch(on_roi_change, 'value')
stim_dropdown.param.watch(on_stim_change, 'value')
scatter_pane.param.watch(on_scatter_click, 'click_data')

layout = pn.Row(pn.Column(scatter_pane, field_dropdown, roi_dropdown, stim_dropdown), detail_pane)
layout.servable()