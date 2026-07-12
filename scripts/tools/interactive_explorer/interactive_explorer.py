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
# %load_ext autoreload
# %autoreload 2

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import plotly.graph_objects as go

from eyewire2_functional_analysis import data_loader, plot, plot_traces, registration
from eyewire2_functional_analysis.space_mapping import align_and_place_skel

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
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = data_loader.add_skels(df, inplace=True)
df = df.reset_index(drop=True)

n_with_skel = df['skel'].notnull().sum()
print(f"{n_with_skel} / {len(df)} EM-matched ROI(s) have a skeleton file; the rest are still selectable")

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
reg = registration.load_registration(REG_FILE)

df['label'] = df.apply(lambda r: f"{r['field']} / ROI {r['roi_id']} - {r['Cell Type']}", axis=1)

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
# ## Cell picker (scatter + field/ROI dropdowns) and detail panel

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


def make_scatter_figure(selected_idx=None):
    """Build the XY cell-picker scatter, highlighting `selected_idx` and overlaying its skeleton.

    Always emits exactly 2 traces (ROI dots, then skeleton -- an empty one
    when there's nothing to overlay) so the figure's trace count never
    changes between renders. Panel/Plotly's `Plotly.react`-based update
    otherwise gets confused about which trace is which when the trace count
    changes from one selection to the next, which was causing stray/delayed
    click events to resolve against the wrong point.
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
        skel_trace = go.Scatter(x=[], y=[], mode='lines', hoverinfo='skip', showlegend=False)

    fig = go.Figure(data=[roi_trace, skel_trace])
    fig.update_layout(
        width=480, height=480,
        margin=dict(l=50, r=10, t=10, b=40),
        showlegend=False,
        xaxis_title='Temporal <-> Nasal [um]',
        yaxis_title='Ventral <-> Dorsal [um]',
        xaxis=dict(range=X_RANGE),
        yaxis=dict(range=Y_RANGE, scaleanchor='x', scaleratio=1),
    )
    return fig


def plot_mc_snippet(ax, row, test_indices=(0, 59, 118)):
    """Zoomed mouse-cam response snippets, overlaid across all 3 test-clip repetitions.

    Adapts the ``axs['C']`` panel of `scripts/tutorial/stimuli/mouse_cam_movies.py`
    (which only showed one repetition) to overlay all of them, aligned to a
    common local time axis, plus their average, via the same
    `plot_traces.plot_snippets_and_average` helper used by `plot.plot_chirp`/
    `plot.plot_bar`/`plot.plot_bar_dir_grid`.
    """
    mc_trace = row.mc_trace
    mc_time = np.arange(mc_trace.size) * row.mc_trace_dt + row.mc_trace_t0
    mc_tt = row.mc_triggertimes
    mc_tt = np.append(mc_tt, mc_tt[-1] + np.median(np.diff(mc_tt)))
    mc_ylim = (mc_trace.min(), mc_trace.max())

    ax.axis('off')

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

    rel_tt = mc_tt[test_indices[0]:test_indices[0] + 6] - mc_tt[test_indices[0]]
    plot_traces.plot_snippets_and_average(
        ax, t_common, snippets,
        vlines=rel_tt, vline_ymin=mc_ylim[1] - np.diff(mc_ylim)[0] * 0.1, vline_ymax=mc_ylim[1],
    )
    plot.plot_scale_bar(ax=ax, x0=rel_tt[0] + 2.5, y0=mc_ylim[0] + 2, size=5, text=True,
                        unit='s', tdist=1, fontsize=8)


def render_cell(idx):
    """Detail figure for row `idx`: DS-on-morph (left, full height), chirp (top right), mouse-cam snippet (bottom right)."""
    row = df.iloc[idx]
    fig = plt.figure(figsize=(13, 5))
    try:
        gs = fig.add_gridspec(2, 3, width_ratios=(1, 1.2, 1.1))

        ax_morph = fig.add_subplot(gs[:, 0])
        if row['skel'] is not None:
            plot.plot_morph(ax_morph, row, reg=reg, rad=None, min_rad=MORPH_MIN_RAD_UM, margin=MORPH_MARGIN_UM,
                            scale_bar_um=MORPH_SCALE_BAR_UM, annotate_orientation=False)
        else:
            ax_morph.text(0.5, 0.5, f"no skeleton found for {row['Latest SegID']}",
                          ha='center', va='center', wrap=True, fontsize=9)
            ax_morph.axis('off')
        ax_morph.set_title(row['label'], fontsize=10)

        plot.plot_bar_dir_grid(fig, gs[:, 1], row)  # sets its own DSI/OSI suptitle

        ax_chirp = fig.add_subplot(gs[0, 2])
        plot.plot_chirp(ax_chirp, row)
        ax_chirp.set_title('Chirp')

        ax_mc = fig.add_subplot(gs[1, 2])
        plot_mc_snippet(ax_mc, row)
        ax_mc.set_title('Mouse cam (3 test reps)', fontsize=9)

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

scatter_pane = pn.pane.Plotly(make_scatter_figure(selected_idx=roi_dropdown.value))
detail_pane = pn.pane.Matplotlib(render_cell(roi_dropdown.value), tight=True, format='png', dpi=100)


def select_index(idx):
    """Update the scatter highlight/overlay and the detail panel for global row `idx`.

    This is the *only* place that actually redraws anything. Both the ROI
    dropdown and the scatter click handler just set `roi_dropdown.value`;
    it's the dropdown's own 'value' watcher (`on_roi_change`, below) that
    calls this -- so there's exactly one codepath that updates the UI,
    regardless of how the selection was made.
    """
    scatter_pane.object = make_scatter_figure(selected_idx=idx)

    old_fig = detail_pane.object
    detail_pane.object = render_cell(idx)
    plt.close(old_fig)


def on_roi_change(event):
    select_index(event.new)


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
        return  # ignore clicks on the skeleton-overlay trace
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
scatter_pane.param.watch(on_scatter_click, 'click_data')

layout = pn.Row(pn.Column(scatter_pane, field_dropdown, roi_dropdown), detail_pane)
layout.servable()

# %%
