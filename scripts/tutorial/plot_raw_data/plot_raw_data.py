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
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from eyewire2_functional_analysis.plot_traces import plot_chirp

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# %% [markdown]
# # Load data

# %%
from eyewire2_functional_analysis import data_loader

# %%
df_rois, df_fields, df_outline = data_loader.load_all_dfs()
df = data_loader.load_df_rois_morph(verbose=True)

# %% [markdown]
# ### Inspect dataframe

# %%
### each row is a cell
df

# %%
### raw and preprocessed ("pp") traces for the chirp
df[['chirp_trace', 'chirp_trace_t0', 'chirp_trace_dt',
    'chirp_pp_trace', 'chirp_pp_trace_t0', 'chirp_pp_trace_dt', 
    'chirp_qidx',
    'chirp_pres_qidx', 
    'chirp_average_norm', 'chirp_average_dt',
    'chirp_snippets', 'chirp_snippets_t0', 'chirp_snippets_dt',
    'chirp_triggertimes_snippets']]

# %%
#raw trace for first cell
trace = df['chirp_trace'][0]
trace

# %%
#time delta between samples
dt = df['chirp_trace_dt'][0]
dt

# %%
# duration of all chirp recordings (seconds)
dt * len(trace)

# %%
raw_time = np.arange(0,stop=dt*len(trace),step=dt)

# %%
trace

# %%
plt.figure(figsize=(15,3))
plt.plot(raw_time, trace)
plt.title('Raw chirp response')
plt.ylabel('Response')
plt.xlabel('Time')
sns.despine()
plt.savefig(os.path.join(FIG_DIR, 'raw_chirp_response.pdf'))

# %% [markdown]
# #### Looking at single trials

# %%
#raw trace cut into its five trials
trials = df['chirp_snippets'][0]
trials

# %%
#time delta between samples
dt_trial = df['chirp_snippets_dt'][0]
dt_trial

# %%
### duration of a single chirp trial
dt_trial * len(trials)

# %%
time = np.arange(0,stop=dt_trial*len(trials),step=dt_trial)

# %%
plt.figure(figsize=(15,3))
plt.plot(time, trials)
sns.despine()
plt.title('Single-trial responses')
plt.ylabel('Response')
plt.xlabel('Time after trigger')
plt.savefig(os.path.join(FIG_DIR, 'single_trial_responses.pdf'))
plt.show()

# %% [markdown]
# ### Averaged & normalized response

# %%
average = df['chirp_average_norm'][0]

# %%
avg_dt = df['chirp_average_dt'][0]

# %%
time_avg = np.arange(0,stop=avg_dt*len(average),step=avg_dt)

# %%
plt.plot(time_avg,average)
plt.title('Average response')
plt.ylabel('Response')
plt.xlabel('Time after trigger')
sns.despine()
plt.savefig(os.path.join(FIG_DIR, 'average_response.pdf'))

# %% [markdown]
# # Plot data

# %%
row = df.iloc[0]

# %%
chirp_time = np.arange(row.chirp_trace.size) * row.chirp_trace_dt + row.chirp_trace_t0
bar_time = np.arange(row.bar_trace.size) * row.bar_trace_dt + row.bar_trace_t0
mc_time = np.arange(row.mc_trace.size) * row.mc_trace_dt + row.mc_trace_t0

# %%
from eyewire2_functional_analysis.plot_traces import plot_bar

# %%
fig, axs = plt.subplots(3, 3, figsize=(15, 7))

# Raw
ax = axs[0, 0]
ax.plot(chirp_time, row.chirp_trace)
ax.vlines(row.chirp_triggertimes_snippets.flatten(), ymin=row.chirp_trace.min(), ymax=row.chirp_trace.max(), color='r',
          label='trigger')
ax.set(xlabel='Time (s)', ylabel='OGB-1\nraw', title='Chirp')
ax.legend(loc='upper right')

ax = axs[0, 1]
ax.plot(bar_time, row.bar_trace)
ax.vlines(row.bar_triggertimes_snippets.flatten(), ymin=row.bar_trace.min(), ymax=row.bar_trace.max(), color='r')
ax.set(xlabel='Time (s)', title='Moving Bar')

ax = axs[0, 2]
ax.plot(mc_time, row.mc_trace)
ax.vlines(row.mc_triggertimes.flatten(), ymin=row.mc_trace.min(), ymax=row.mc_trace.max(), color='r')
ax.set(xlabel='Time (s)', title='Mouse Cam Movies')

# Preprocessed
ax = axs[1, 0]
ax.plot(chirp_time, row.chirp_pp_trace)
ax.vlines(row.chirp_triggertimes_snippets.flatten(), ymin=row.chirp_pp_trace.min(), ymax=row.chirp_pp_trace.max(),
          color='r')
ax.set(xlabel='Time (s)', ylabel='OGB-1\npreprocessed')

ax = axs[1, 1]
ax.plot(bar_time, row.bar_pp_trace)
ax.vlines(row.bar_triggertimes_snippets.flatten(), ymin=row.bar_pp_trace.min(), ymax=row.bar_pp_trace.max(), color='r')
ax.set(xlabel='Time (s)')

ax = axs[1, 2]
ax.plot(mc_time, row.mc_pp_trace)
ax.vlines(row.mc_triggertimes.flatten(), ymin=row.mc_pp_trace.min(), ymax=row.mc_pp_trace.max(), color='r')
ax.set(xlabel='Time (s)')

# Average and repeats
ax = axs[2, 0]
plot_chirp(ax=ax, row=row, stimulus_ms=None, plot_hline=True, plot_vlines=True)
ax.set(xlabel='Relative time (s)', ylabel='Snippets &\nAverage')

ax = axs[2, 1]
plot_bar(ax=ax, row=row, annotate_dirs=True, annotate_symbols=False, ventral_up=True)
ax.set(xlabel='Relative time (s)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'raw_data_overview.pdf'))
