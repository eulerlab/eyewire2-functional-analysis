# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive EM <-> function explorer (v0)
#
# Pick a 2p ROI, either from the dropdown or by clicking a dot in the XY
# scatter, and see its EM skeleton next to its chirp / moving-bar responses.
#
# This is a **quick first cut**, deliberately not fancy yet. Not included
# here (see `BRAINSTORM.md` in this folder for the full plan):
# - the session-timeline slider
# - the light-exposure/stimulus-history overlay
# - the stimulus footprint drawn behind the cell mosaic
#
# Needs a live Jupyter kernel to render the widgets -- open this in
# `uv run jupyter lab`, don't just `uv run python` it.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display

from eyewire2_functional_analysis import data_loader, plot, registration

# %% [markdown]
# ## Load data
#
# Restrict to ROIs that both (a) matched a proofread EM cell
# (`load_df_rois_morph`) and (b) have an EM skeleton file on disk
# (`add_skels`) -- those are the only ROIs this tool can show a skeleton for.

# %%
df_rois = data_loader.load_df_rois()
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = data_loader.add_skels(df, inplace=True)

n_matched = len(df)
df = df[df['skel'].notnull()].reset_index(drop=True)
print(f"{len(df)} / {n_matched} EM-matched ROI(s) have a skeleton file and will show up in the picker")

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
reg = registration.load_registration(REG_FILE)

df['label'] = df.apply(lambda r: f"{r['field']} / ROI {r['roi_id']} - {r['Cell Type']}", axis=1)

# %% [markdown]
# ## Cell picker (scatter + dropdown) and detail panel

# %%
scatter = go.FigureWidget(
    data=[go.Scatter(
        x=df['temporal_nasal_pos_um'],
        y=df['ventral_dorsal_pos_um'],
        mode='markers',
        marker=dict(size=8, color='steelblue', line=dict(width=0.5, color='white')),
        text=df['label'],
        hoverinfo='text',
    )]
)
scatter.update_layout(
    width=480, height=480,
    margin=dict(l=50, r=10, t=10, b=40),
    xaxis_title='Temporal <-> Nasal [um]',
    yaxis_title='Ventral <-> Dorsal [um]',
    yaxis=dict(scaleanchor='x', scaleratio=1),
)

dropdown = widgets.Dropdown(
    options=[(label, i) for i, label in enumerate(df['label'])],
    description='Cell:',
    layout=widgets.Layout(width='420px'),
)

output = widgets.Output()


def highlight_point(idx):
    colors = ['steelblue'] * len(df)
    sizes = [8] * len(df)
    colors[idx] = 'crimson'
    sizes[idx] = 14
    with scatter.batch_update():
        scatter.data[0].marker.color = colors
        scatter.data[0].marker.size = sizes


def render_cell(idx):
    row = df.iloc[idx]
    output.clear_output(wait=True)
    with output:
        try:
            fig = plt.figure(figsize=(12, 3.2))
            gs = fig.add_gridspec(1, 4, width_ratios=(1, 1.3, 1.6, 0.9))

            ax_morph = fig.add_subplot(gs[0, 0])
            plot.plot_morph(ax_morph, row, reg=reg, rad=200)
            ax_morph.set_title('EM skeleton')

            ax_chirp = fig.add_subplot(gs[0, 1])
            plot.plot_chirp(ax_chirp, row)
            ax_chirp.set_title('Chirp')

            ax_bar = fig.add_subplot(gs[0, 2])
            plot.plot_bar(ax_bar, row, annotate_symbols=True)
            ax_bar.set_title('Moving bar')

            ax_polar = fig.add_subplot(gs[0, 3], projection='polar')
            plot.plot_bar_dir(ax_polar, row)
            ax_polar.set_title('DS tuning', pad=15)

            fig.suptitle(row['label'])
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot {row['label']}: {e}")


def on_dropdown_change(change):
    if change['name'] == 'value' and change['new'] is not None:
        idx = change['new']
        highlight_point(idx)
        render_cell(idx)


dropdown.observe(on_dropdown_change, names='value')


def on_scatter_click(trace, points, state):
    if points.point_inds:
        dropdown.value = points.point_inds[0]  # triggers on_dropdown_change above


scatter.data[0].on_click(on_scatter_click)

ui = widgets.VBox([widgets.HBox([scatter, dropdown]), output])
display(ui)

dropdown.value = 0  # trigger the initial render

# %%
