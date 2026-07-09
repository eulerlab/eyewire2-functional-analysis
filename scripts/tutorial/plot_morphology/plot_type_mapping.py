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

# %%
from eyewire2_functional_analysis import data_loader

df_rois, df_fields, df_outline = data_loader.load_all_dfs()
df = data_loader.load_df_rois_morph(df_rois=df_rois)

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# %%
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 12))
sns.countplot(ax=ax, data=df, y='Cell Type')
plt.savefig(os.path.join(FIG_DIR, 'cell_type_counts.pdf'))
plt.show()

# %%
fig, ax = plt.subplots(3, 1, figsize=(15, 6))
sns.countplot(ax=ax[0], data=df, x='cluster_id', hue='Cell Class', hue_order=['RGC', 'AC'])
sns.countplot(ax=ax[1], data=df, x='group_id', hue='Cell Class', hue_order=['RGC', 'AC'])
sns.countplot(ax=ax[2], data=df, x='supergroup', hue='Cell Class', hue_order=['RGC', 'AC'],
              order=['OFF', 'ON-OFF', 'Fast ON', 'Slow ON', 'Unc. ON', 'Unc. SbC', 'dAC'])
plt.savefig(os.path.join(FIG_DIR, 'cluster_group_supergroup_counts.pdf'))
plt.show()

# %%
