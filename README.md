## Principles of Collaboration

**All data and code shared here is subject to the [Eyewire II Principles of Collaboration](https://eyewire.ai/principles).** By the use of this repository, you agree to be bound by these Principles.

## License note

The code in this repository is under MIT license. All data in the repository, as stated in the principles of collaboration, is shared under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/). For the data, the license holders are H. Sebastian Seung, Thomas Euler, Philipp Berens, and Greg Schwartz.

# Eyewire II: Functional data

This repository hosts tools to analyse the functional data from OGB-1 recordings in the Eyewire II dataset.

The following files are included:
- tutorial scripts to facilitate data exploration [scripts/tutorial/*.py](scripts/tutorial/)
- an interactive EM ↔ function explorer tool [scripts/tools/interactive_explorer/](scripts/tools/interactive_explorer/)
- analysis scripts [scripts/analysis/*.py](scripts/analysis/)
- preprocessing scripts, e.g. for the 2P-to-EM coordinate registration [scripts/preprocessing/*.py](scripts/preprocessing/)
- spreadsheet data that are needed to map 2p to EM data (proofread cell main list, 2P-to-EM ROI mapping) in [data/spreadsheets/*](data/spreadsheets/)
- pre-processed calcium traces from recordings over five recording fields, stored as parquet files and downloadable from [Hugging Face](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main/data-2p) (see [data/data-2p/README.md](data/data-2p/README.md))

Scripts are plain `.py` files in [jupytext](https://jupytext.readthedocs.io/) "percent" format (`# %%` cell markers) rather than `.ipynb` notebooks — open them in Jupyter Lab to run them cell-by-cell like a notebook, or run them directly with `uv run python <script>.py`.

Documentation is still incomplete:
- a description of the 2P data can be found [here](data/data-2p/README.md).
- a description of the stimuli can be found [here](data/stimuli/README.md).

Feel free to open issues to ask questions and request features!

### Setup

To use the code in this repository out of the box, you can use [uv](https://docs.astral.sh/uv/) to reproduce our python environment. Follow these steps:

- Install `uv`
- Clone this repository and navigate to its root folder
- Run `uv run jupyter lab` to start jupyter lab  - it should open in your browser, and allows you to run our scripts as notebooks.

On the first call, `uv run` will install all dependencies into a `uv` virtual environment (placed in the `.venv` folder), which is then invoked on all further calls of `uv run`.

### Downloading the data

The pre-processed 2P data is not included in this repository — download it from the [eyewire2-data Hugging Face dataset](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main/data-2p) and place it in `data/data-2p/`. See [data/data-2p/README.md](data/data-2p/README.md) for details on the contents.

### Loading the data

All data loading is handled by `eyewire2_functional_analysis.data_loader`. The easiest way to load all three DataFrames at once is:

```python
from eyewire2_functional_analysis import data_loader

data_folder = "data/data-2p"

df_rois, df_fields, df_outline = data_loader.load_all_dfs(data_folder)
```

You can also load each DataFrame individually using `load_df_rois()`, `load_df_fields()`, or `load_df_outline()`.

See the tutorial scripts for full usage examples:
- [plot_raw_data.py](scripts/tutorial/plot_raw_data/plot_raw_data.py) — load data and plot raw + preprocessed traces for individual ROIs
- [plot_data_overview.py](scripts/tutorial/plot_data_overview/plot_data_overview.py) — plot chirp and bar response overviews grouped by cell type and recording field
- [plot_morph_and_func.py](scripts/tutorial/plot_morphology/plot_morph_and_func.py) — plot morphology and functional data for individual cells
- [plot_retinal_outline.py](scripts/tutorial/plot_retinal_outline/plot_retinal_outline.py) — plot the retinal outline and recording field locations
- [plot_DS_on_morph.py](scripts/tutorial/plot_DS_on_morph/plot_DS_on_morph.py) — plot direction selectivity on cell morphology
- [plot_stimulus_overlay.py](scripts/tutorial/plot_stimulus_overlay/plot_stimulus_overlay.py) — for each recording field and stimulus type (chirp, moving bar, mouse cam), draw that field's stimulus footprint over the retinal outline and ROI positions, at its true recorded position and physical scale

### Interactive explorer

[scripts/tools/interactive_explorer/interactive_explorer.py](scripts/tools/interactive_explorer/interactive_explorer.py) lets you pick a 2P ROI — via the field/ROI dropdowns, or by clicking a point in the retinal-position scatter — and see its EM skeleton next to its chirp, moving-bar, and mouse-cam responses. Built with [Panel](https://panel.holoviz.org/); run it either:
- as a standalone app, no Jupyter needed: `uv run panel serve scripts/tools/interactive_explorer/interactive_explorer.py --show`
- or opened cell-by-cell in Jupyter/VS Code, like the other scripts in this repo.

In addition to `data/data-2p/`, this tool needs the EM skeletons (`data/swc/`) and the master spreadsheets (`data/spreadsheets/`) to link 2P ROIs to their EM reconstructions.

### 2P ↔ EM coordinate registration

[scripts/preprocessing/em-2p-mapping.py](scripts/preprocessing/em-2p-mapping.py) fits the rotation + isotropic scale (a global fit, plus a per-field refinement) that maps 2P ROI coordinates onto EM soma coordinates, using the known EM-cell ↔ 2P-ROI correspondences in the EM-2P mapping spreadsheet — fit independently in both directions (`2p_to_em` and `em_to_2p`), rather than guessing an angle or algebraically inverting one fit to get the other.

The fitted registration is what most other scripts/tools in this repo (`plot_DS_on_morph.py`, `plot_retinal_outline.py`, the interactive explorer, ...) load via `registration.load_registration(...)` to rotate EM skeletons into the 2P/retinal reference frame. It's already committed at `data/registration/em_2p_registration.yaml`, so you don't need to run this script yourself unless you want to re-fit it (e.g. after the spreadsheets are updated) or inspect the fit diagnostics, saved as figures under `scripts/preprocessing/figures/`. Set `REFIT = True` in the script to force a re-fit even if the file already exists.

It also writes `2p_roi_estimated_em_coordinates.csv` (an estimated EM-space coordinate for every 2P ROI, including still-unmatched ones) and a Neuroglancer link to help locate those unmatched cells.
