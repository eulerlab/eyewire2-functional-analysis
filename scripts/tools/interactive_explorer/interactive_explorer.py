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
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

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

# %% [markdown]
# ## Stimulus-timeline tab
#
# A second view: scrub absolute session time and see the ROI traces recorded
# at that instant (heatmap), the field/stimulus footprint that was on screen
# at that instant (reusing `make_scatter_figure` from above), and that
# field's cells -- individually selectable, independent of the timeline.
#
# Session timing comes from `experiment-overview_consolidated.csv` (same
# file used in `scripts/analysis/light_exposure/EW2_stim_history.py`). Each
# field's chirp/moving-bar/mouse-cam recording is its own acquisition with an
# independent local clock (`*_pp_trace_t0`/`*_pp_trace_dt`) -- there is no
# single continuous trace spanning the whole session, so time outside the
# currently active recording block (a field switch, a setup gap, or -- for
# GCL3's Chirp, which was recorded twice -- the second of the two logged
# presentations, since only one of them made it into the released
# `chirp_trace`) is treated as "nothing recorded" rather than an error.

# %%
HEATMAP_PRE_S = 5.0
HEATMAP_POST_S = 15.0

STIM_LABELS = {v: k for k, v in STIM_OPTIONS.items()}
STIM_TRACE_PREFIX = {'DS': 'bar', 'Chirp': 'chirp', 'MouseCam_Right': 'mc'}

CONSOL_PATH = os.path.join(data_loader.DATA_ROOT, 'experiment-overview_consolidated.csv')
df_stim_log = pd.read_csv(CONSOL_PATH, sep=';', on_bad_lines='warn')


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
MC_ARRAY_DIR = os.path.join(data_loader.DATA_ROOT, 'stimuli', 'mouse_cam_movies', 'mc_arrays')
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
        movie = np.load(os.path.join(data_loader.DATA_ROOT, 'stimuli', subdir, fname))
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
        ax.add_patch(Rectangle((t_rel[0], pos - 0.5), t_rel[-1] - t_rel[0], 1,
                                fill=False, edgecolor='lime', linewidth=2, zorder=5))
        ax_trace.plot(t_rel, mat[pos], color='#1a7a1a', lw=1.2)
        ax_trace.set_ylabel(f"ROI {sub.loc[selected_idx, 'roi_id']}")
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


def render_morph_only(idx):
    """Standalone EM-skeleton-on-morphology panel for row `idx` (no chirp/bar/mc detail plots),
    or a placeholder if `idx` is `None` (no cell selected)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    if idx is None:
        ax.text(0.5, 0.5, 'No cell selected', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        return fig

    row = df.iloc[idx]
    if row['skel'] is not None:
        plot_morph.plot_morph(ax, row, reg=reg, rad=None, min_rad=MORPH_MIN_RAD_UM, margin=MORPH_MARGIN_UM,
                        scale_bar_um=MORPH_SCALE_BAR_UM, annotate_orientation=True)
    else:
        ax.text(0.5, 0.5, f"no skeleton found for {row['Latest SegID']}", ha='center', va='center', wrap=True)
        ax.axis('off')
    ax.set_title(row['label'])
    fig.tight_layout()
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
tl_morph_pane = pn.pane.Matplotlib(render_morph_only(tl_roi_dropdown.value), tight=True, format='png', dpi=100)


def redraw_tl_scatter(idx):
    tl_scatter_pane.object = make_scatter_figure(
        selected_idx=idx, stim_type=tl_state['stim_type'], field=tl_state['field'])


def redraw_tl_morph(idx):
    old_fig = tl_morph_pane.object
    tl_morph_pane.object = render_morph_only(idx)
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
        tl_morph_pane,
    ),
)

# %%
app = pn.Tabs(('Cell explorer', layout), ('Stimulus timeline', timeline_layout))
app.servable()