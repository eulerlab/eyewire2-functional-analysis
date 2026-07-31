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
# # Cell-type explorer (v0)
#
# Pick a cell class (RGC / dAC / Other -- see `Cell Class`), a morphological cell type
# within that class (`Cell Type`, the ground-truth EM/proofreading label -- see
# `data_loader.load_df_rois_morph`), and either one recording field (`GCL0`..`GCL4`) or
# all of them, and see every matching cell's:
# - EM skeleton (left, click a cell to select it -- see below), overlaid at its true
#   position in the shared retinal frame (same placement logic as `scripts/tutorial/
#   plot_retinal_outline/plot_retinal_outline.py` and the main `scripts/tools/
#   interactive_explorer/interactive_explorer.py`).
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
# This is a first cut -- not yet included: a functional-classification reliability summary
# for the group, or a quality-index toggle beyond the simple `qfilt` filter below.
#
# Built with [Panel](https://panel.holoviz.org/), matching `interactive_explorer.py` --
# including its use of a clickable Plotly pane (here, for the morphology panel) to drive
# cell selection.
# Two ways to run this:
# - Opened in Jupyter (`uv run jupyter lab`, right-click -> Open With -> Notebook)
#   for cell-by-cell exploration.
# - As a standalone local app: `uv run panel serve scripts/tools/celltype_explorer/celltype_explorer.py --show`

# %%
import os
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from eyewire2_functional_analysis import baden16_utils, data_loader, plot_morph, plot_traces, registration, style
from eyewire2_functional_analysis.ds import MB_DIRS, MB_DIRS_SYMBOLS_D_UP
from eyewire2_functional_analysis.space_mapping import align_and_place_skel

style.set_rc_params()
pn.extension('plotly', 'modal')

# %% [markdown]
# ## Load data
#
# Restrict to ROIs that matched a proofread EM cell (`load_df_rois_morph`) --
# `Cell Type` (the column selected on below) only exists for those. Not all of
# those have an EM skeleton file on disk (`add_skels` leaves `skel` as `None`
# when the SWC file is missing) -- those cells are still counted/included in
# the response overlays, just skipped in the morphology panel.

# %%
df_rois = data_loader.load_df_rois()
df_outline = data_loader.load_df_outline()
df_fields = data_loader.load_df_fields()
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = data_loader.add_skels(df, inplace=True)
df = df.reset_index(drop=True)

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
reg = registration.load_registration(REG_FILE)

FIELDS = sorted(df['field'].unique())
FIELD_COLORS_HEX = {f: mcolors.to_hex(c) for f, c in zip(FIELDS, plt.get_cmap('tab10').colors)}
FIELD_OPTIONS = {'All fields': None, **{f: f for f in FIELDS}}

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
# cell, dAC; see `interactive_explorer.py`'s `CELL_CLASS_COLORS`) plus a handful of
# `'Fragment'`/missing rows, bucketed here as 'Other'. A handful of `Cell Type` labels
# (e.g. "F-mini-ON", "Other") are used for cells of more than one class, so the class
# filter is applied at the row level (`cell_class_mask`), not just used to prune the type
# list -- otherwise switching class with such a type still selected would silently mix
# classes back in.

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
# per-field quantity -- the field's own Chirp acquisition start time, read from the same
# `experiment-overview_consolidated.csv` session log used by
# `interactive_explorer.py`'s stimulus-timeline tab (`build_session_blocks`). Where a
# field's Chirp was logged more than once (e.g. GCL3, restarted once), the earliest logged
# presentation with an actual recording file is used, matching that same logic.

# %%
CONSOL_PATH = os.path.join(data_loader.DATA_ROOT, 'experiment-overview_consolidated.csv')
df_stim_log = pd.read_csv(CONSOL_PATH, sep=';', on_bad_lines='warn')


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
    if field is not None:
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
    `interactive_explorer.py`."""
    frow = df_fields[df_fields['field'] == field].iloc[0]
    w = frow['nxpix'] * frow['pixel_size_um']
    h = frow['nypix'] * frow['pixel_size_um']
    cx, cy = frow['field_temporal_nasal_pos_um'], frow['field_ventral_dorsal_pos_um']
    xs = [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2]
    ys = [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2]
    return xs, ys


def make_morph_figure(sub, field, selected_key):
    """Plotly figure overlaying every row in `sub`'s EM skeleton at its true 2p position
    (via `align_and_place_skel`), colour-coded by field, and click-selectable (see
    `on_morph_click`). Cells without a skeleton file are still marked (as an 'x') at their
    2p position, so the group's full spatial coverage is visible even where morphology is
    missing.

    The retinal outline is drawn as spatial context whenever `field` is `None` ('All
    fields'), and the recorded field(s)' own imaging-window outline(s) are always drawn
    (`field_outline_rect`) -- but the view always zooms dynamically to fit every plotted
    morphology (with a margin) rather than to the whole retina/outline, so cells are
    legible regardless of how tightly they cluster, whether that's one field or all five.

    If `selected_key` (a `df` row index, or `None`) names one of `sub`'s cells, that cell's
    skeleton/marker are drawn solid black (same line width as everyone else -- only colour
    changes) and every other cell is dimmed (rather than converted to gray, so the
    field-colour legend stays meaningful); with no selection, all cells use their normal
    field colour at full opacity.

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
    (retinal/field outlines, the invisible background click-catcher, legend swatches) --
    used by `on_morph_click` to resolve a click into a cell selection (`None` meaning
    "clicked the background", i.e. deselect).
    """
    fig = go.Figure()
    curve_to_key = {}
    fields_shown = FIELDS if field is None else [field]

    if field is None:
        fig.add_trace(go.Scattergl(
            x=df_outline['temporal_nasal_pos_um'], y=df_outline['ventral_dorsal_pos_um'], mode='lines',
            line=dict(color='dimgray', width=1, dash='dot'), hoverinfo='none', showlegend=False))
        curve_to_key[len(fig.data) - 1] = None

    for f in fields_shown:
        xs, ys = field_outline_rect(f)
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode='lines', line=dict(color='lightgray', width=1),
                                  hoverinfo='none', showlegend=False))
        curve_to_key[len(fig.data) - 1] = None

    xs_cells, ys_cells = [], []
    n_with_skel = 0
    for key, row in sub.iterrows():
        is_selected = key == selected_key
        dimmed = selected_key is not None and not is_selected
        color = 'black' if is_selected else FIELD_COLORS_HEX[row['field']]
        opacity = 0.25 if dimmed else 1.0
        x, y = row['temporal_nasal_pos_um'], row['ventral_dorsal_pos_um']

        if row['skel'] is not None:
            n_with_skel += 1
            skel = align_and_place_skel(row['skel'], reg, field=row['field'], target_xy=(x, y))
            seg_x, seg_y = [], []
            for (x0, y0), (x1, y1) in zip(skel.nodes[skel.edges[:, 0], :2], skel.nodes[skel.edges[:, 1], :2]):
                seg_x += [x0, x1, None]
                seg_y += [y0, y1, None]
            fig.add_trace(go.Scattergl(x=seg_x, y=seg_y, mode='lines', line=dict(color=color, width=1),
                                      opacity=opacity, hoverinfo='none', showlegend=False))
            curve_to_key[len(fig.data) - 1] = key
            xs_cells += list(skel.nodes[:, 0])
            ys_cells += list(skel.nodes[:, 1])
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

    if field is None:
        for f in FIELDS:
            if f in sub['field'].unique():
                fig.add_trace(go.Scattergl(x=[None], y=[None], mode='markers',
                                          marker=dict(size=10, color=FIELD_COLORS_HEX[f]), name=f, showlegend=True))
                curve_to_key[len(fig.data) - 1] = None

    fig.update_layout(
        width=MORPH_PLOTLY_SIZE['width'], height=MORPH_PLOTLY_SIZE['height'],
        margin=dict(l=50, r=10, t=30, b=40),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=field is None,
        legend=dict(title='Field', font=dict(size=10)),
        xaxis=dict(title='Temporal <-> Nasal [um]', range=x_range, gridcolor='#eee', zerolinecolor='#eee'),
        yaxis=dict(title='Ventral <-> Dorsal [um]', range=y_range, scaleanchor='x', scaleratio=1,
                   gridcolor='#eee', zerolinecolor='#eee'),
        title=dict(text=f"Morphology ({n_with_skel}/{len(sub)} with skeleton)", x=0, xanchor='left',
                   font=dict(size=13)),
    )
    return fig, curve_to_key


def row_color(row):
    """A row's colour for the response-overlay panels: when its field was recorded during
    the session (`FIELD_CHIRP_T0`/`TIME_CMAP`/`TIME_NORM`)."""
    return TIME_CMAP(TIME_NORM(FIELD_CHIRP_T0[row['field']]))


# `group_id` per entry of `baden16_utils.BADEN_CLUSTER_INFO`, in the same (first-occurrence)
# order `baden16_utils.probs_per_cluster_to_level_probs` builds its `group_probs` list in --
# zipped against that list below to recover each group's numeric ID, since that function
# itself only returns `(name, supergroup, prob)` tuples.
_GROUP_IDS_IN_LEVEL_PROBS_ORDER = list(dict.fromkeys(
    baden16_utils.BADEN_CLUSTER_INFO[:, 2].astype(int)))


def top2_group_probs(row, correct_by_cellclass=False):
    """The predicted-type probability distribution over the 46 Baden et al. 2016 functional
    groups for `row` (from its `probs_per_cluster` cluster-level vector, aggregated via
    `baden16_utils.probs_per_cluster_to_level_probs` -- see `plot_morph.
    plot_type_prediction_bars`, which uses the same aggregation for its group-level bar),
    sorted descending and truncated to the top 2.

    `correct_by_cellclass` matches `interactive_explorer.py`'s `cellclass_correction_dropdown`/
    `plot_morph.plot_type_prediction_bars`: if set, `row['probs_per_cluster']` is first passed
    through `baden16_utils.correct_probs_by_cellclass(row['Cell Class'], ...)`, zeroing out/
    renormalizing probability mass inconsistent with the cell's known ground-truth `Cell Class`
    (RGC vs. dAC) -- the classifier confuses the two fairly often. Off by default, same as there.

    Returns a list of up to 2 ``(group_id, supergroup, prob)`` tuples (fewer only if a cell
    somehow has fewer than 2 groups with nonzero probability mass)."""
    probs = row['probs_per_cluster']
    if correct_by_cellclass:
        probs = baden16_utils.correct_probs_by_cellclass(row['Cell Class'], probs)
    _, group_probs, _ = baden16_utils.probs_per_cluster_to_level_probs(probs)
    group_probs = [(gid, sg, p) for gid, (_, sg, p) in zip(_GROUP_IDS_IN_LEVEL_PROBS_ORDER, group_probs) if p > 0]
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

    If `selected_key` (a `df` row index, or `None`) names one of `sub`'s cells with a cached
    profile, that cell's curve is drawn solid black (bold) and every other curve is dimmed
    (not converted to gray, matching `make_morph_figure`); with no selection, all curves use
    their normal field colour at full opacity. The ON/OFF SAC (ChAT band) reference lines
    (`plot_morph.plot_sac_lines`'s fixed z=0/z=12 convention) are drawn once, in data
    coordinates, spanning the full plotted density range.

    Returns ``(fig, curve_to_key)``, same contract as `make_morph_figure` -- `curve_to_key`
    maps this figure's own curve numbers back to `df` row indices (or `None` for the SAC
    lines/background), used by the same `on_morph_click`-style handler.
    """
    fig = go.Figure()
    curve_to_key = {}

    profiles = {}
    for key, row in sub.iterrows():
        prof = data_loader.load_ipl_profile(row['Latest SegID'])
        if prof is not None:
            profiles[key] = prof

    vmax = max((prof['dens'].max() for prof in profiles.values()), default=1.0)
    xlim = (-0.1 * vmax, vmax * 1.1)

    fig.add_trace(go.Scattergl(x=list(xlim), y=[0, 0], mode='lines',
                              line=dict(color='#FFC09F', width=1), hoverinfo='none', showlegend=False))
    curve_to_key[len(fig.data) - 1] = None
    fig.add_trace(go.Scattergl(x=list(xlim), y=[12, 12], mode='lines',
                              line=dict(color='#17CFB9', width=1), hoverinfo='none', showlegend=False))
    curve_to_key[len(fig.data) - 1] = None

    for key, row in sub.iterrows():
        if key not in profiles:
            continue
        is_selected = key == selected_key
        dimmed = selected_key is not None and not is_selected
        color = 'black' if is_selected else FIELD_COLORS_HEX[row['field']]
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
        yaxis=dict(title='IPL depth [um]', range=IPL_ZLIM, autorange='reversed',
                   gridcolor='#eee', zerolinecolor='#eee'),
        title=dict(text=f"IPL depth profile ({len(profiles)}/{len(sub)} with cached profile)",
                   x=0, xanchor='left', font=dict(size=13)),
    )
    return fig, curve_to_key


def empty_ipl_figure():
    """Placeholder for `ipl_pane` when no cells match the current filters -- mirrors
    `empty_morph_figure`."""
    fig = go.Figure()
    fig.update_layout(
        width=IPL_PLOTLY_SIZE['width'], height=IPL_PLOTLY_SIZE['height'],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text='No cells match this selection', showarrow=False,
                          xref='paper', yref='paper', x=0.5, y=0.5)],
    )
    return fig, {}


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


# Rendered CSS pixel width of `group_prob_bar_html`'s bar (excludes the field/ROI/cell-type
# columns) -- generous, since a narrow bar leaves too little room for its group-ID labels to
# be legible once a segment's probability share is small.
GROUP_PROB_BAR_WIDTH_PX = 400


def group_prob_bar_html(row, correct_by_cellclass=False, height_px=18, width_px=GROUP_PROB_BAR_WIDTH_PX):
    """A horizontal stacked-bar HTML snippet, always spanning the full 0-100% probability
    range (2 coloured segments for `row`'s top-2 predicted functional groups, plus a gray
    remainder segment covering the rest of the mass, together always summing to `width_px`
    -- flexbox-based, no plotting library involved, cheap to build per row of a long list),
    each segment sized by its probability and labelled with its (numeric) group ID, coloured
    by its parent supergroup (`plot_morph.SUPERGROUP_COLORS`, the same convention `plot_morph.
    plot_type_prediction_bars` uses) -- so the same supergroup reads as the same colour here
    and in `interactive_explorer.py`'s per-cell prediction bars. `correct_by_cellclass` is
    forwarded to `top2_group_probs`."""
    segments = top2_group_probs(row, correct_by_cellclass=correct_by_cellclass)
    spans = []
    for group_id, supergroup, prob in segments:
        color = plot_morph.SUPERGROUP_COLORS.get(supergroup, '#999999')
        spans.append(
            f'<div title="Group {group_id} ({supergroup}): {prob:.0%}" '
            f'style="flex:{max(prob, 0.001)}; background:{color}; color:white; '
            f'display:flex; align-items:center; justify-content:center; overflow:hidden; '
            f'font-size:10px; white-space:nowrap;">{group_id}</div>'
        )
    remainder = 1.0 - sum(p for _, _, p in segments)
    if remainder > 0:
        spans.append(f'<div style="flex:{remainder}; background:#e0e0e0;"></div>')
    return (f'<div style="display:flex; width:{width_px}px; height:{height_px}px; '
            f'border-radius:3px; overflow:hidden;">{"".join(spans)}</div>')


def render_cell_list_html(sub, selected_key, correct_by_cellclass=False):
    """HTML table listing every row of `sub` -- field, ROI number, morphological cell type,
    and its top-2 predicted functional-group probabilities as a horizontal stacked bar
    (`group_prob_bar_html`) -- below the response-overlay panels (`render_traces_fig`).

    `selected_key` (a `df` row index, or `None`) highlights that row (light-blue background),
    matching the highlight-on-selection convention used across the other panels, so the
    selected cell can also be found by scrolling this list. `correct_by_cellclass` is
    forwarded to `group_prob_bar_html`/`top2_group_probs`."""
    rows_html = []
    for key, row in sub.iterrows():
        bg = '#eaf2fb' if key == selected_key else 'transparent'
        rows_html.append(
            f'<tr style="background:{bg};">'
            f'<td style="padding:2px 8px;">{row["field"]}</td>'
            f'<td style="padding:2px 8px;">{row["roi_id"]}</td>'
            f'<td style="padding:2px 8px;">{row["Cell Type"]}</td>'
            f'<td style="padding:2px 8px;">{group_prob_bar_html(row, correct_by_cellclass=correct_by_cellclass)}</td>'
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
        '<th style="padding:2px 8px;">Top-2 predicted functional group (0-100%)</th>'
        '</tr></thead><tbody>' + ''.join(rows_html) + '</tbody></table></div>'
    )


# %% [markdown]
# ## Widgets

# %%
# Fixed, fairly narrow widths -- laid out in a row above the panels (rather than a column
# to their left) so they don't eat into the panels' own width.
field_dropdown = pn.widgets.Select(name='Field', options=FIELD_OPTIONS, value=None, width=140)
cellclass_dropdown = pn.widgets.Select(name='Cell class', options=CELL_CLASS_OPTIONS, value='RGC', width=140)
celltype_dropdown = pn.widgets.Select(name='Morphological cell type',
                                       options=CELL_TYPES_BY_CLASS['RGC'], value=CELL_TYPES_BY_CLASS['RGC'][0],
                                       width=260)
qfilt_checkbox = pn.widgets.Checkbox(name='Quality filter only (qfilt)', value=True)
# Options rebuilt on every `full_redraw` (see `cell_dropdown_options`) to match the current
# filtered `sub` -- an alternative, keyboard/list-friendly way to pick the highlighted cell
# alongside clicking it in `morph_pane`/`ipl_pane`.
cell_dropdown = pn.widgets.Select(name='Highlight cell', options={'(none)': None}, value=None, width=200)
# Same toggle/options/default (raw, uncorrected) as `interactive_explorer.py`'s
# `cellclass_correction_dropdown` -- applies `baden16_utils.correct_probs_by_cellclass` to
# `cell_list_pane`'s top-2 group-probability bars (see `top2_group_probs`).
CELLCLASS_CORRECTION_OPTIONS = {'Raw model prediction': False, 'Corrected by Cell Class (RGC vs. dAC)': True}
cellclass_correction_dropdown = pn.widgets.Select(name='Predicted type', options=CELLCLASS_CORRECTION_OPTIONS,
                                                  value=False, width=220)
add_to_table_button = pn.widgets.Button(name='Add to table', button_type='primary', width=120)

# The currently-selected cell (a `df` row index, or `None`) -- click a cell in `morph_pane`/
# `ipl_pane`, or pick it from `cell_dropdown`, to set it; click the background to clear it.
# Any filter change resets it too (`on_filter_change`), since a selection made under one
# field/cell-type/class combination wouldn't necessarily mean anything under another.
selected_key = None
# `curve_to_key[curveNumber]` for the currently-shown `morph_pane`/`ipl_pane` figures --
# rebuilt by `full_redraw` on every render (see `make_morph_figure`/`make_ipl_figure`), read
# by `on_morph_click`/`on_ipl_click`.
CURVE_TO_KEY = {}
IPL_CURVE_TO_KEY = {}
# Guards `cell_dropdown`'s 'value' watcher while `full_redraw` sets it programmatically (to
# reflect a selection made by clicking a panel, or to reset it on a filter change) -- without
# this, that assignment would itself fire `on_cell_dropdown_change`, which is only meant to
# react to the user's own picks.
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
    fig = go.Figure()
    fig.update_layout(
        width=MORPH_PLOTLY_SIZE['width'], height=MORPH_PLOTLY_SIZE['height'],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text='No cells match this selection', showarrow=False,
                          xref='paper', yref='paper', x=0.5, y=0.5)],
    )
    return fig, {}


_initial_sub = selected_cells(field_dropdown.value, celltype_dropdown.value, qfilt_checkbox.value,
                               cellclass_dropdown.value)
if len(_initial_sub):
    _initial_morph_fig, CURVE_TO_KEY = make_morph_figure(_initial_sub, field_dropdown.value, selected_key)
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
    f"{field_dropdown.value or 'All fields'} -- n={len(_initial_sub)}",
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
    render_cell_list_html(_initial_sub, selected_key, correct_by_cellclass=cellclass_correction_dropdown.value),
    margin=(10, 0, 0, 0))


def full_redraw():
    """The only place that actually redraws `morph_pane`/`ipl_pane`/`traces_pane`/
    `title_pane`/`selection_pane`/`cell_dropdown`/`cell_list_pane` -- every widget/click
    handler below just updates `selected_key` (or a filter widget) and calls this, so
    there's exactly one code path that rebuilds the view, regardless of how the change was
    triggered (matches `interactive_explorer.py`'s `select_index`)."""
    global CURVE_TO_KEY, IPL_CURVE_TO_KEY, _updating_cell_dropdown
    field = field_dropdown.value
    sub = selected_cells(field, celltype_dropdown.value, qfilt_checkbox.value, cellclass_dropdown.value)

    if len(sub):
        morph_fig, CURVE_TO_KEY = make_morph_figure(sub, field, selected_key)
        ipl_fig, IPL_CURVE_TO_KEY = make_ipl_figure(sub, selected_key)
    else:
        morph_fig, CURVE_TO_KEY = empty_morph_figure()
        ipl_fig, IPL_CURVE_TO_KEY = empty_ipl_figure()
    morph_pane.object = morph_fig
    ipl_pane.object = ipl_fig

    old_traces_fig = traces_pane.object
    traces_pane.object = render_traces_fig(sub, selected_key)
    plt.close(old_traces_fig)

    field_label = field if field is not None else 'All fields'
    title_pane.object = (f"### {cellclass_dropdown.value} -- {celltype_dropdown.value} -- "
                         f"{field_label} -- n={len(sub)}")

    if selected_key is not None and selected_key in sub.index:
        sel_row = sub.loc[selected_key]
        selection_pane.object = f"**Field:** {sel_row['field']}  \n**ROI:** {sel_row['roi_id']}"
    else:
        selection_pane.object = ''

    cell_list_pane.object = render_cell_list_html(sub, selected_key,
                                                  correct_by_cellclass=cellclass_correction_dropdown.value)

    # Rebuild to match `sub` (options change with every filter change) and reflect
    # `selected_key` (set here whether the selection came from `cell_dropdown` itself, a
    # panel click, or a filter reset) -- guarded so this assignment doesn't recursively
    # trigger `on_cell_dropdown_change`.
    _updating_cell_dropdown = True
    try:
        cell_dropdown.param.update(options=cell_dropdown_options(sub),
                                   value=selected_key if selected_key in sub.index else None)
    finally:
        _updating_cell_dropdown = False


def on_filter_change(event=None):
    global selected_key
    selected_key = None
    full_redraw()


NO_TYPES_PLACEHOLDER = '(no cell types for this class)'


def on_cellclass_change(event):
    """Switching cell class swaps the type list to that class's own types (see
    `CELL_TYPES_BY_CLASS`) and jumps to its first entry -- set together via one atomic
    Param update so there's no intermediate state where `celltype_dropdown.value` is
    invalid for its new `options`. 'Other' has none in the released data (every
    `Fragment`/missing-class row also has a missing `Cell Type`) -- fall back to a
    placeholder rather than crashing on an empty options list; `selected_cells` then
    naturally matches zero rows against it. Doesn't call `full_redraw` itself: setting
    `celltype_dropdown`'s value fires its own 'value' watcher (`on_filter_change`), which
    does."""
    options = CELL_TYPES_BY_CLASS[event.new] or [NO_TYPES_PLACEHOLDER]
    celltype_dropdown.param.update(options=options, value=options[0])


def on_morph_click(event):
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
    full_redraw()


def on_ipl_click(event):
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
    full_redraw()


def on_cell_dropdown_change(event):
    """Set `selected_key` from the user's own pick in `cell_dropdown` -- a keyboard/list-
    friendly alternative to clicking the cell in `morph_pane`/`ipl_pane`. Ignored while
    `full_redraw` is itself updating `cell_dropdown.value` to reflect a selection made some
    other way (`_updating_cell_dropdown`), since that's not a new user choice."""
    global selected_key
    if _updating_cell_dropdown:
        return
    selected_key = event.new
    full_redraw()


field_dropdown.param.watch(on_filter_change, 'value')
cellclass_dropdown.param.watch(on_cellclass_change, 'value')
celltype_dropdown.param.watch(on_filter_change, 'value')
qfilt_checkbox.param.watch(on_filter_change, 'value')
morph_pane.param.watch(on_morph_click, 'click_data')
ipl_pane.param.watch(on_ipl_click, 'click_data')
cell_dropdown.param.watch(on_cell_dropdown_change, 'value')
# Only `cell_list_pane`'s bars depend on this (unlike the filter widgets above), but there's
# a single `full_redraw` code path, so it also gets called here -- it doesn't reset
# `selected_key`, since toggling the correction isn't a filter change.
cellclass_correction_dropdown.param.watch(lambda event: full_redraw(), 'value')

# %% [markdown]
# ### "Add to table" logging
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

layout = pn.Column(
    title_pane,
    pn.Row(field_dropdown, cellclass_dropdown, celltype_dropdown, qfilt_checkbox, cell_dropdown,
          cellclass_correction_dropdown, add_to_table_button),
    # `margin=0` on the panes themselves plus here on their Row -- morphology and the
    # response panels sit right next to each other, no gap. `selection_pane`/`ipl_pane` sit
    # below the morphology pane, in its own column, so they don't affect that spacing.
    pn.Row(pn.Column(morph_pane, selection_pane, ipl_pane, margin=0), traces_pane, margin=0),
    cell_list_pane,
    log_modal,
    margin=(0, 0, 0, 32),
)

# %%
app = layout
app.servable()
