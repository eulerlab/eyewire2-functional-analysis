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
# # Precompute IPL stratification profiles
#
# `plot_morph.plot_ipl_profile` computes a cell's IPL depth-density profile on
# the fly via `pywarper` (SAC-surface flattening + z-profile extraction),
# which is somewhat slow, especially without the optional `scikit-sparse`
# dependency. For interactive use (`scripts/tools/interactive_explorer`),
# precompute the profile for every EM-matched cell with a skeleton once here
# and cache it as a small 2-column CSV (`z`, `dens`) per cell, keyed by
# `Latest SegID` (matching `data/swc/{Latest SegID}.swc`), under
# `data_loader.DATA_IPL_PROFILES`. Loaded back via `data_loader.load_ipl_profile`.
#
# Re-run this whenever the SWC skeletons or the EM-2p mapping change; it's
# idempotent (just overwrites existing CSVs).

# %%
import os
import sys

# `pywarper.surface` (imported lazily by `plot_morph.compute_ipl_z_profile`) prints a notice
# containing '≈' the first time it's imported if the optional `scikit-sparse` dependency is
# missing (harmless -- it just means a slower fallback solver). On Windows, the default console
# codepage (cp1252) can't encode that character, so `print()` itself would raise instead of just
# showing the notice; reconfigure stdout/stderr to replace unencodable characters instead.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(errors='replace')

import pandas as pd

from eyewire2_functional_analysis import data_loader, plot_morph

# %%
OUT_DIR = data_loader.DATA_IPL_PROFILES
os.makedirs(OUT_DIR, exist_ok=True)

# %%
df_rois = data_loader.load_df_rois()
df = data_loader.load_df_rois_morph(df_rois=df_rois, verbose=True)
df = data_loader.add_skels(df, inplace=True)
df = df.drop_duplicates(subset='Latest SegID')  # one profile per cell, not per (repeated) ROI

# %%
n_ok, n_failed, n_skipped = 0, 0, 0
for _, row in df.iterrows():
    seg_id = row['Latest SegID']

    if row['skel'] is None:
        n_skipped += 1
        continue

    try:
        ipl, dens = plot_morph.compute_ipl_z_profile(row['skel'])
    except Exception as e:
        print(f"Failed for {seg_id}: {e}")
        n_failed += 1
        continue

    out_path = os.path.join(OUT_DIR, f'{seg_id}.csv')
    pd.DataFrame({'z': ipl, 'dens': dens}).to_csv(out_path, index=False)
    n_ok += 1

print(f"{n_ok} profile(s) written, {n_failed} failed, {n_skipped} skipped (no skeleton) / {len(df)} cell(s)")
