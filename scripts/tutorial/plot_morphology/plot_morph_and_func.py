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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
from eyewire2_functional_analysis import data_loader, registration, plot_traces, plot_utils

df_rois, df_fields, df_outline = data_loader.load_all_dfs()
df = data_loader.load_df_rois_morph(df_rois=df_rois)

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
reg = registration.load_registration(REG_FILE)

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 12))
sns.countplot(ax=ax, data=df, y='Cell Type')
plt.savefig(os.path.join(FIG_DIR, 'cell_type_counts.pdf'))
plt.show()

# %%
from eyewire2_functional_analysis import plot_morph

import skeliner as sk


def plot_cells(df, reg):
    fig, axs = plt.subplots(len(df), 5, figsize=(6, len(df) * 0.8), width_ratios=(0.9, 0.5, 1.6, 1.6, 0.6))

    for i, ax_to_replace in enumerate(axs[:, -1]):
        ax_to_replace.remove()
        polar_ax = fig.add_subplot(ax_to_replace.get_subplotspec(), polar=True)
        axs[i, -1] = polar_ax

    all_xmin, all_xmax = [], []
    all_ymin, all_ymax = [], []

    for seg_id, row in df.iterrows():
        skel_rot = registration.align_skel(row['skel'], reg, field=row['field'], direction='em_to_2p')

        nodes = skel_rot.nodes[skel_rot.ntype == 3]
        nodes -= skel_rot.soma.center

        xmax, ymax, zmax = np.max(nodes, axis=0)
        xmin, ymin, zmin = np.min(nodes, axis=0)

        all_xmin.append(xmin)
        all_xmax.append(xmax)
        all_ymin.append(ymin)
        all_ymax.append(ymax)

    # Compute global limits with padding
    global_xlim = (min(all_xmin) - 13, max(all_xmax) + 3)
    global_ylim = (min(all_ymin) - 3, max(all_ymax) + 3)

    for i, (seg_id, row) in enumerate(df.iterrows()):
        ax = axs[i, 0]
        skel_rot = registration.align_skel(row['skel'], reg, field=row['field'], direction='em_to_2p')
        skel_rot.nodes -= skel_rot.soma.center

        sk.plot.projection(skel_rot,
                           ax=ax, xlim=global_xlim, ylim=global_ylim, plane='xy', draw_cylinders=False)
        plot_utils.plot_scale_bar(ax=ax, x0=global_xlim[0] + 5, y0=np.mean(global_ylim),
                            size=100, text=False, unit='µm', tdist=0, orientation='v')
        plot_morph.plot_em_axis_indicator(ax, reg, field=row['field'], direction='em_to_2p',
                                    center=(0, 0), lw=1, labels=('', ''))
        ax.set_rasterized(True)

        ax = axs[i, 1]
        plot_morph.plot_ipl_profile(ax=ax, row=row)

        ax = axs[i, 2]
        plot_traces.plot_chirp(ax=ax, row=row)
        if i == (df.shape[0] - 1):
            plot_utils.plot_scale_bar(ax=ax, x0=1, y0=-0.4, size=2, text=True, tdist=0.05, unit='s')
        ax.set_ylim(-0.5, +1.1)

        ax = axs[i, 3]
        plot_traces.plot_bar(ax=ax, row=row, annotate_dirs=False, annotate_symbols=i == 0, ventral_up=False)
        if i == (df.shape[0] - 1):
            plot_utils.plot_scale_bar(ax=ax, x0=1, y0=-0.4, size=2, text=True, tdist=0.05, unit='s')
        ax.set_ylim(-0.5, +1.1)

        ax = axs[i, 4]
        plot_traces.plot_bar_dir(ax=ax, row=row, ventral_up=False)

    for ax in axs[:, :-1].flat:
        ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[])
        ax.axis('off')
        ax.set_facecolor((1, 1, 1, 0))

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    return fig, axs


# %%
df['swc_path'] = ''
df['skel'] = None

# %%
from eyewire2_functional_analysis.data_loader import DATA_ROOT


skel_dir = os.path.join(DATA_ROOT, 'swc')


# %%
def add_skels(df):
    df = df.copy()
    df['swc_path'] = df['Latest SegID'].apply(lambda x: os.path.join(skel_dir, f"{x}.swc"))
    df['skel'] = df.apply(lambda row: sk.io.load_swc(row['swc_path']) if os.path.isfile(row['swc_path']) else None, axis=1)
    return df


# %%
df_plot = add_skels(df[
    ((df.chirp_qidx > 0.45) | (df.bar_qidx > 0.6))  # Responsive
    & (df['Cell Class'] == 'RGC')  # RGCs only
].iloc[:10])
df_plot = df_plot[df_plot.skel.notnull()]

fig, axs = plot_cells(df_plot, reg)
plt.savefig(os.path.join(FIG_DIR, 'cells_RGC_responsive.pdf'))
plt.show()

# %%
df_plot = add_skels(df[
    ((df.chirp_qidx > 0.45) | (df.bar_qidx > 0.6))  # Responsive
    & (df['Cell Type'] == 'F-mini-ON')  # F-mini-ON only
].iloc[:10])
df_plot = df_plot[df_plot.skel.notnull()]

fig, axs = plot_cells(df_plot, reg)
plt.savefig(os.path.join(FIG_DIR, 'cells_F-mini-ON.pdf'))
plt.show()

# %%
