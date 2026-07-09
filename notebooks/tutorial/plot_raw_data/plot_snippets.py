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
# Load functional data
from eyewire2_functional_analysis import data_loader

df_rois, df_fields, df_outline = data_loader.load_all_dfs()
print(df_rois.shape)
print(list(df_rois.columns))
df_rois.head()
print(df_fields.shape)
print(list(df_fields.columns))
df_fields.head()

# %%
from matplotlib import pyplot as plt
import numpy as np

from eyewire2_functional_analysis import plot_traces

row = df_rois[(df_rois.bar_qidx > 0.6) & (df_rois.chirp_qidx > 0.6)].iloc[0]
for stim in ['chirp', 'bar']:
    snippets_t0 = row[f'{stim}_snippets_t0']
    snippets_dt = row[f'{stim}_snippets_dt']
    snippets = row[f'{stim}_snippets']
    triggertimes_snippets = row[f'{stim}_triggertimes_snippets']

    snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                      + snippets_t0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    plot_traces.plot_trace_and_trigger(
        ax=axs[0], time=snippets_times, trace=snippets, triggertimes=triggertimes_snippets)

    axs[1].plot(snippets_times - triggertimes_snippets[0], snippets, alpha=0.5)
    axs[1].set(ylabel='trace', xlabel='rel. to trigger')

    aligned_times = plot_traces.get_aligned_snippets_times(snippets_times=snippets_times)
    plot_traces.plot_traces(
        ax=axs[2], time=aligned_times, traces=snippets.T)
    axs[2].set(ylabel='trace', xlabel='aligned time')

    plt.tight_layout()

# %%
