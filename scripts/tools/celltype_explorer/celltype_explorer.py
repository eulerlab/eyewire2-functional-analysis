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
# - EM skeleton, overlaid at its true position in the shared retinal frame
#   (same placement logic as `scripts/tutorial/plot_retinal_outline/plot_retinal_outline.py`
#   and the main `scripts/tools/interactive_explorer/interactive_explorer.py`).
# - Chirp, moving-bar, and natural-movie (mouse-cam) responses, each averaged per cell
#   (chirp: `chirp_average_norm`; moving bar: `bar_time_component`, the SVD time kernel
#   already used this way in `plot_dataframe.plot_df_chirp_and_bar`; natural movie: the
#   mean of the 3 repeated test-clip presentations, same snippets as
#   `plot_traces.plot_mc_test_snippets`), all overlaid on their own axes plus the
#   across-cell mean +/- SD. The three response panels are stacked to together match the
#   morphology panel's height.
#
# This is a first cut -- not yet included: per-direction/DS-tuning detail for the moving
# bar, a functional-classification reliability summary for the group, or a quality-index
# toggle beyond the simple `qfilt` filter below.
#
# Built with [Panel](https://panel.holoviz.org/), matching `interactive_explorer.py`.
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
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle

from eyewire2_functional_analysis import data_loader, plot_traces, registration, style
from eyewire2_functional_analysis.space_mapping import align_and_place_skel

style.set_rc_params()
pn.extension()

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
FIELD_COLORS = dict(zip(FIELDS, plt.get_cmap('tab10').colors))
FIELD_OPTIONS = {'All fields': None, **{f: f for f in FIELDS}}

MORPH_MIN_ZOOM_UM = 100

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


def plot_field_outlines(ax, fields):
    """Light-grey rectangle marking each field's actual 2p imaging window (`nxpix`/`nypix`
    pixels at `pixel_size_um`, centered on its registered `field_temporal_nasal_pos_um`/
    `field_ventral_dorsal_pos_um` position) -- i.e. the small (~94x94 um) patch the ROIs'
    somata were actually scanned in, as distinct from the much larger dendritic fields
    their morphologies can extend into, and from the (much bigger, ~1000 um) stimulus
    field of view used for the footprint overlays in `interactive_explorer.py`."""
    for f in fields:
        frow = df_fields[df_fields['field'] == f].iloc[0]
        w = frow['nxpix'] * frow['pixel_size_um']
        h = frow['nypix'] * frow['pixel_size_um']
        x0 = frow['field_temporal_nasal_pos_um'] - w / 2
        y0 = frow['field_ventral_dorsal_pos_um'] - h / 2
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor='lightgray', linewidth=1, zorder=1))


def plot_morph_overlay(ax, sub, field):
    """Overlay every row in `sub`'s EM skeleton (rotated + placed at its true 2p position,
    via `align_and_place_skel`), colour-coded by field. Cells without a skeleton file are
    still marked (as an 'x') at their 2p position, so the group's full spatial coverage is
    visible even where morphology is missing.

    The retinal outline is drawn as spatial context whenever `field` is `None` ('All
    fields'), and the recorded field(s)' own imaging-window outline(s) are always drawn
    (see `plot_field_outlines`) -- but the view always zooms dynamically to fit every
    plotted morphology (with a margin) rather than to the whole retina/outline, so cells
    are legible regardless of how tightly they cluster, whether that's one field or all
    five.
    """
    if field is None:
        ax.plot(df_outline['temporal_nasal_pos_um'], df_outline['ventral_dorsal_pos_um'],
                c='dimgray', ls=':', lw=1, zorder=0)
    plot_field_outlines(ax, FIELDS if field is None else [field])

    xs, ys = [], []
    n_with_skel = 0
    for _, row in sub.iterrows():
        color = FIELD_COLORS[row['field']]
        x, y = row['temporal_nasal_pos_um'], row['ventral_dorsal_pos_um']
        if row['skel'] is None:
            ax.plot(x, y, marker='x', color=color, ms=6, mew=1.2, zorder=3)
            xs.append(x)
            ys.append(y)
            continue
        n_with_skel += 1
        skel = align_and_place_skel(row['skel'], reg, field=row['field'], target_xy=(x, y))
        seg = np.stack([skel.nodes[skel.edges[:, 0], :2], skel.nodes[skel.edges[:, 1], :2]], axis=1)
        ax.add_collection(LineCollection(seg, colors=color, linewidths=0.7, alpha=0.6, zorder=2))
        ax.scatter(*skel.soma.center[:2], color=color, s=12, alpha=0.85, zorder=3)
        xs += list(skel.nodes[:, 0])
        ys += list(skel.nodes[:, 1])

    pad = 50
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    # Keep it square-ish (aspect='equal' below would otherwise letterbox unevenly).
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin, MORPH_MIN_ZOOM_UM) / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    if field is None:
        legend_fields = [f for f in FIELDS if f in sub['field'].unique()]
        handles = [Patch(facecolor=FIELD_COLORS[f], label=f) for f in legend_fields]
        ax.legend(handles=handles, loc='upper right', fontsize=7, frameon=False, title='Field')

    ax.set(xlabel='Temporal <-> Nasal [um]', ylabel='Ventral <-> Dorsal [um]', aspect='equal')
    ax.set_title(f"Morphology ({n_with_skel}/{len(sub)} with skeleton)")


def row_color(row):
    """A row's colour for the response-overlay panels: when its field was recorded during
    the session (`FIELD_CHIRP_T0`/`TIME_CMAP`/`TIME_NORM`)."""
    return TIME_CMAP(TIME_NORM(FIELD_CHIRP_T0[row['field']]))


def plot_overlay_with_mean(ax, time, traces, colors, title, vlines=()):
    """Shared drawing logic for the three response-overlay panels: each row's trace in
    `traces` (shape ``(n_cells, time)``) plotted thin/translucent in its own `colors`
    entry, plus the across-cell mean +/- SD on top in black."""
    for color, trace in zip(colors, traces):
        ax.plot(time, trace, color=color, lw=0.8, alpha=0.6, clip_on=False)

    mean = traces.mean(axis=0)
    sd = traces.std(axis=0)
    ax.plot(time, mean, color='black', lw=2, clip_on=False)
    ax.fill_between(time, mean - sd, mean + sd, color='black', alpha=0.15)

    ax.axhline(0, color='dimgray', ls='--', lw=1)
    for t in vlines:
        ax.axvline(t, color='gray', ls='--', lw=1)
    ax.set(xlabel='Time [s]', ylabel='Norm. Ca.')
    ax.set_title(title, loc='left')


def plot_chirp_overlay(ax, sub):
    """Every row's mean-over-trials chirp response (`chirp_average_norm`) overlaid, colour-
    coded by recording time (see `row_color`). Returns the panel's time-axis duration (s),
    for `scale_panel_widths_by_duration`.

    Assumes a shared time axis/length across rows (true for the released data, same as
    `plot_dataframe.plot_df_chirp_and_bar`)."""
    dt = sub['chirp_average_dt'].iloc[0]
    n_t = sub['chirp_average_norm'].iloc[0].size
    time = np.arange(n_t) * dt
    traces = np.stack(sub['chirp_average_norm'].to_numpy())
    colors = [row_color(row) for _, row in sub.iterrows()]
    plot_overlay_with_mean(ax, time, traces, colors, f"Chirp, mean over trials per cell (n={len(sub)})",
                           vlines=(2, 5, 8, 30))
    return time[-1]


def plot_bar_overlay(ax, sub):
    """Every row's moving-bar temporal response (`bar_time_component`, the SVD time kernel
    that collapses across direction and repeats -- same quantity `plot_dataframe.
    plot_df_chirp_and_bar` stacks per-row) overlaid, colour-coded by recording time.
    Returns the panel's time-axis duration (s), for `scale_panel_widths_by_duration`.

    Assumes a shared time axis/length across rows (true for the released data)."""
    dt = sub['bar_snippets_dt'].iloc[0]
    n_t = sub['bar_time_component'].iloc[0].size
    time = np.arange(n_t) * dt
    traces = np.stack(sub['bar_time_component'].to_numpy())
    colors = [row_color(row) for _, row in sub.iterrows()]
    plot_overlay_with_mean(ax, time, traces, colors, f"Moving bar, time component (n={len(sub)})",
                           vlines=plot_traces.BAR_STIM_TIMES)
    return time[-1]


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


def plot_mc_overlay(ax, sub):
    """Every row's mean-over-test-reps natural-movie (mouse-cam) response overlaid,
    colour-coded by recording time. Returns the panel's time-axis duration (s), for
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

    plot_overlay_with_mean(ax, time, traces, colors, f"Natural movie (mouse cam), mean over 3 test reps (n={len(sub)})",
                           vlines=vlines)
    return time[-1]


def add_time_colorbar(fig, ax_bar, ax_chirp, height_frac=0.4, gap=0.015):
    """Horizontal colorbar translating `row_color`'s colour back to session-elapsed
    recording time in minutes, increasing left-to-right (matplotlib's default for a
    horizontal bar, `TIME_NORM`'s low/high end at the left/right respectively -- no flip
    needed). Ticked at each field's own recording time.

    Placed to the right of the (duration-narrowed, see `scale_panel_widths_by_duration`)
    moving-bar panel, filling the freed-up width up to the chirp panel's right edge --
    chirp has the longest duration of the 3 response panels, so it keeps its full
    original width and marks where that freed-up space ends. Must be called after
    `scale_panel_widths_by_duration`, since `ax_bar`'s width (and thus how much space is
    actually free to its right) is only finalized there.
    """
    pos_bar = ax_bar.get_position()
    pos_chirp = ax_chirp.get_position()
    x0 = pos_bar.x0 + pos_bar.width + gap
    width = (pos_chirp.x0 + pos_chirp.width) - x0
    height = pos_bar.height * height_frac
    y0 = pos_bar.y0 + (pos_bar.height - height) / 2
    cax = fig.add_axes([x0, y0, width, height])

    sm = plt.cm.ScalarMappable(cmap=TIME_CMAP, norm=TIME_NORM)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Recording time [min]', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    tick_times = list(FIELD_CHIRP_T0.values())
    cb.set_ticks(tick_times)
    cb.set_ticklabels([f"{t / 60:.0f}" for t in tick_times])
    return cax


def align_response_panels_to_morph(ax_morph, response_axes, gap=0.08):
    """Shrink/reposition the stacked `response_axes` so their combined height matches
    `ax_morph`'s actual rendered height.

    `ax_morph` is drawn with `aspect='equal'` on (square) data limits (see
    `plot_morph_overlay`), which -- via matplotlib's default `adjustable='box'` -- shrinks
    its rendered box to the largest square that fits its gridspec slot; since that slot
    spans 3 rows but only 1 column, the square ends up width-limited and shorter than the
    slot, leaving the response panels (which keep their full gridspec-assigned height)
    visibly taller than the morphology panel. Must be called after the figure's layout is
    finalized (e.g. right after `fig.tight_layout()`), since the aspect-driven box
    shrinking is only resolved at draw time -- same reasoning as
    `interactive_explorer.py`'s `align_ipl_panel_box`.
    """
    morph_pos = ax_morph.get_position()
    n = len(response_axes)
    panel_height = (morph_pos.height - gap * (n - 1)) / n

    y_top = morph_pos.y0 + morph_pos.height
    for ax in response_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, y_top - panel_height, pos.width, panel_height])
        y_top -= panel_height + gap


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


def render_celltype(field, cell_type, qfilt_only, cell_class):
    sub = selected_cells(field, cell_type, qfilt_only, cell_class)

    fig = plt.figure(figsize=(12.5, 9))
    # Morphology (`ax_morph`) spans all 3 rows; the response panels start out the same way
    # (so they divide its full height) but get shrunk to actually match it in
    # `align_response_panels_to_morph` below, once that height is known post-layout. No
    # gridspec slot is reserved for the colorbar -- it's placed freehand afterwards (see
    # `add_time_colorbar`), in whatever width `scale_panel_widths_by_duration` frees up to
    # the right of the moving-bar panel.
    gs = fig.add_gridspec(3, 2, width_ratios=(1, 1.15), hspace=0.5, wspace=0.35)
    ax_morph = fig.add_subplot(gs[:, 0])
    # `aspect='equal'` (see plot_morph_overlay) shrinks ax_morph's box to fit its slot,
    # normally centering the leftover space above+below -- anchor it to the top instead,
    # so that leftover space ends up below the whole morph+response block (where
    # bbox_inches='tight'/Panel's tight=True can just crop it) rather than as a gap
    # between the suptitle and the block.
    ax_morph.set_anchor('N')
    ax_chirp = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_mc = fig.add_subplot(gs[2, 1])

    if len(sub) == 0:
        for ax in (ax_morph, ax_chirp, ax_bar, ax_mc):
            ax.text(0.5, 0.5, 'No cells match this selection', ha='center', va='center')
            ax.axis('off')
    else:
        plot_morph_overlay(ax_morph, sub, field)
        chirp_duration = plot_chirp_overlay(ax_chirp, sub)
        bar_duration = plot_bar_overlay(ax_bar, sub)
        mc_duration = plot_mc_overlay(ax_mc, sub)

    field_label = field if field is not None else 'All fields'
    fig.suptitle(f"{cell_class} -- {cell_type} -- {field_label} -- n={len(sub)}")
    with warnings.catch_warnings():
        # The legend added in plot_morph_overlay (when field is None) isn't created via the
        # ax= convenience path tight_layout expects; checked empirically that layout still
        # comes out correct regardless, so the warning is benign (same as
        # interactive_explorer.py's render_heatmap).
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout()
    if len(sub) > 0:
        align_response_panels_to_morph(ax_morph, [ax_chirp, ax_bar, ax_mc])
        scale_panel_widths_by_duration([ax_chirp, ax_bar, ax_mc], [chirp_duration, bar_duration, mc_duration])
        add_time_colorbar(fig, ax_bar, ax_chirp)
    return fig


# %% [markdown]
# ## Widgets

# %%
field_dropdown = pn.widgets.Select(name='Field', options=FIELD_OPTIONS, value=None)
cellclass_dropdown = pn.widgets.Select(name='Cell class', options=CELL_CLASS_OPTIONS, value='RGC')
celltype_dropdown = pn.widgets.Select(name='Morphological cell type',
                                       options=CELL_TYPES_BY_CLASS['RGC'], value=CELL_TYPES_BY_CLASS['RGC'][0])
qfilt_checkbox = pn.widgets.Checkbox(name='Quality filter only (qfilt)', value=True)

detail_pane = pn.pane.Matplotlib(
    render_celltype(field_dropdown.value, celltype_dropdown.value, qfilt_checkbox.value, cellclass_dropdown.value),
    tight=True, format='png', dpi=110,
)


def redraw(event=None):
    old_fig = detail_pane.object
    detail_pane.object = render_celltype(field_dropdown.value, celltype_dropdown.value, qfilt_checkbox.value,
                                          cellclass_dropdown.value)
    plt.close(old_fig)


NO_TYPES_PLACEHOLDER = '(no cell types for this class)'


def on_cellclass_change(event):
    """Switching cell class swaps the type list to that class's own types (see
    `CELL_TYPES_BY_CLASS`) and jumps to its first entry -- set together via one atomic
    Param update so there's no intermediate state where `celltype_dropdown.value` is
    invalid for its new `options`. 'Other' has none in the released data (every
    `Fragment`/missing-class row also has a missing `Cell Type`) -- fall back to a
    placeholder rather than crashing on an empty options list; `selected_cells` then
    naturally matches zero rows against it."""
    options = CELL_TYPES_BY_CLASS[event.new] or [NO_TYPES_PLACEHOLDER]
    celltype_dropdown.param.update(options=options, value=options[0])


field_dropdown.param.watch(redraw, 'value')
cellclass_dropdown.param.watch(on_cellclass_change, 'value')
celltype_dropdown.param.watch(redraw, 'value')
qfilt_checkbox.param.watch(redraw, 'value')

layout = pn.Row(
    pn.Column(field_dropdown, cellclass_dropdown, celltype_dropdown, qfilt_checkbox, width=250),
    detail_pane,
)

# %%
app = layout
app.servable()
