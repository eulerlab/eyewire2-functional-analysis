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
from matplotlib import pyplot as plt
from IPython.core.display_functions import clear_output

# %% [markdown]
# # Load data

# %%
from eyewire2_functional_analysis import data_loader

df_rois_morph = data_loader.load_df_rois_morph()

# %% [markdown]
# # Plot data

# %%
from eyewire2_functional_analysis.plot_dataframe import plot_df_chirp_and_bar

# %%
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')

fig_dir = os.path.join(FIG_DIR, 'responses_by_field')
os.makedirs(fig_dir, exist_ok=True)

for (field, cell_class), df_plot in df_rois_morph.groupby(['field', 'Cell Class']):
    clear_output()
    fig, axs = plot_df_chirp_and_bar(
        df=df_plot, id_col='Latest NucID', type_col='Cell Type',
        title=f"{field}: All {cell_class}s", chirp_q_thresh=0.45, bar_q_thresh=0.6)
    plt.savefig(os.path.join(fig_dir, f'response_overview_{field}_{cell_class}.pdf'))
    plt.close(fig)

# %%
fig_dir = os.path.join(FIG_DIR, 'responses_by_type')
os.makedirs(fig_dir, exist_ok=True)

for cell_type, df_plot in df_rois_morph.groupby('Cell Type'):
    clear_output()
    fig, axs = plot_df_chirp_and_bar(
        df=df_plot.sort_values(['field', 'roi_id'], inplace=False),
        id_col='Latest NucID', type_col='Cell Type',
        title=str(cell_type), chirp_q_thresh=0.45, bar_q_thresh=0.6)
    
    # remove all ".", ",", ":" etc from cell_type for filename
    cell_type_file = str(cell_type).replace(" ", "_").replace(".", "_").replace(",", "_").replace(":", "_")

    plt.savefig(os.path.join(fig_dir, f'response_overview_{cell_type_file}.pdf'))
    plt.close(fig)

# %%

# %%
