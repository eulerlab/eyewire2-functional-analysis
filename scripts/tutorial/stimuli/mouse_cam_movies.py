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
# # Setup

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

# %%
from eyewire2_functional_analysis import data_loader

df_rois, df_fields, df_outline= data_loader.load_all_dfs()
df = data_loader.load_df_rois_morph(df_rois=df_rois)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# ## Stimuli

# %%
# You have to run mc_to_numpy.py first, before loading the mc array

mc15 = np.load(os.path.join(data_loader.SHARED_DATA_ROOT, "stimuli", "mouse_cam_movies", "mc_arrays", "MC15.npy"))
assert mc15.shape[0] == 123 * 5 * 30
mc15.shape

# %%
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axs.flat):
    frame = i * 150
    ax.set(xticks=[], yticks=[], title=f"z={frame}")
    ax.imshow(mc15[frame, :, :])
    ax.axis('off')

plt.savefig(os.path.join(FIG_DIR, 'mc_frames.pdf'))

# %% [markdown]
# # Plot

# %%
from eyewire2_functional_analysis import plot_utils

# %%
row = df.iloc[0]

# %%
fig, axs = plt.subplot_mosaic(
    """
    EFGHI
    CCCCC
    CCCCC
    CCCCC
    DDDDD
    """,
    figsize=(12, 4),
    width_ratios=(0.3, 0.3, 0.3, 0.3, 0.3),
    height_ratios=(1, 0.2, 0.2, 0.2, 1),
)

ax = axs['D']
mc_trace = row.mc_trace
mc_time = np.arange(mc_trace.size) * row.mc_trace_dt + row.mc_trace_t0
mc_tt = row.mc_triggertimes
mc_tt = np.append(mc_tt, mc_tt[-1] + np.median(np.diff(mc_tt)))
mc_ylim = (mc_trace.min(), mc_trace.max())

ax.axis('off')
for test_i in [0, 59, 118]:
    ax.fill_between(x=[mc_tt[test_i], mc_tt[test_i + 5]], y1=mc_ylim[0], y2=mc_ylim[1], color='C0', alpha=0.3, lw=0,
                    label='test' if test_i == 0 else '_')

ax.fill_between(x=[mc_tt[5], mc_tt[5 + 54]], y1=mc_ylim[0], y2=mc_ylim[1], color='C1', alpha=0.3, lw=0, label='train1')
ax.fill_between(x=[mc_tt[59 + 5], mc_tt[59 + 54 + 5]], y1=mc_ylim[0], y2=mc_ylim[1], color='C2', alpha=0.3, lw=0,
                label='train2')

ax.plot(mc_time, mc_trace, c='dimgray', lw=1)
ax.vlines(mc_tt, ymin=mc_ylim[1] - np.diff(mc_ylim) * 0.1, ymax=mc_ylim[1], colors='k', linestyles='-', lw=0.3,
          label='Trigger')

ax.legend(bbox_to_anchor=(0.8, 0.5), loc='center left', borderaxespad=0., frameon=False, handlelength=1.5, )
plot_utils.plot_scale_bar(ax=ax, x0=np.mean(mc_tt[:len(mc_tt) // 2]), y0=mc_ylim[0] + 3, size=60, text=True, unit='s',
                    tdist=1, fontsize=8)
ax.set_xlim(mc_time[0], mc_time[-1] + (mc_time[-1] - mc_time[0]) * 0.35)

ax = axs['C']
tlim_test2 = (mc_tt[59] - mc_tt[59 + 5])
ilim = (mc_time >= mc_tt[59] - 10) & (mc_time <= mc_tt[59 + 5] + 10)
ax.axis('off')
ax.plot(mc_time[ilim], mc_trace[ilim], c='dimgray', lw=1)
for test_i in [59]:
    ax.fill_between(x=[mc_tt[test_i], mc_tt[test_i + 5]], y1=mc_ylim[0], y2=mc_ylim[1], color='C0', alpha=0.3, lw=0,
                    label='test' if test_i == 0 else '_')

ax.vlines(mc_tt[59 - 2:59 + 5 + 2], ymin=mc_ylim[1] - np.diff(mc_ylim) * 0.1, ymax=mc_ylim[1], colors='k',
          linestyles='-', lw=1)
plot_utils.plot_scale_bar(ax=ax, x0=mc_tt[59] + 2.5, y0=mc_ylim[0] + 2, size=5, text=True, unit='s', tdist=1, fontsize=8)

for i in range(5):
    seq = 59 + i
    frame = seq * 30 * 5
    ax = axs[['E', 'F', 'G', 'H', 'I'][i]]
    ax.set(xticks=[], yticks=[])
    ax.imshow(mc15[frame, :, :])
    ax.axis('off')

plt.savefig(os.path.join(FIG_DIR, 'mc_trace_overview.pdf'))

# %%
