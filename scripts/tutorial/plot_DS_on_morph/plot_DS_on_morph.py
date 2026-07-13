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
# # Direction selectivity on morphology
#
# Plot a single direction-selective cell's moving-bar direction tuning on its
# EM skeleton, with the skeleton rotated/flipped into the 2p/retinal
# reference frame using the fitted 2p<->EM registration (see
# `scripts/preprocessing/em-2p-mapping.py`, which fits and saves it).

# %%
import os

import matplotlib.pyplot as plt

from eyewire2_functional_analysis import data_loader, registration
from eyewire2_functional_analysis.plot_morph import plot_ds_on_morph

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CELL_TYPE = "ON-OFF DS - ventral"

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)

# %% [markdown]
# ## Load data
#
# Pick a random cell of the target type that has a matched EM skeleton, and
# load the fitted 2p<->EM registration (run `scripts/preprocessing/em-2p-mapping.py`
# first if `REG_FILE` doesn't exist yet).

# %%
df_rois = data_loader.load_df_rois()
df = data_loader.load_df_rois_morph(df_rois=df_rois)
df = df[df['Cell Type'] == CELL_TYPE].copy()
df = data_loader.add_skels(df=df, inplace=True)
df = df[df['skel'].notnull()].copy()
row = df.sample(n=1, random_state=0).iloc[0]

reg = registration.load_registration(REG_FILE)

# %% [markdown]
# ## Plot

# %%
fig = plot_ds_on_morph(row, reg=reg)
fig.savefig(os.path.join(FIG_DIR, "plot_DS_on_morph.pdf"))
plt.show()
