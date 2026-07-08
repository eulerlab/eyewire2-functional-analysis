## Principles of Collaboration

**All data and code shared here is subject to the [Eyewire II Principles of Collaboration](https://eyewire.ai/principles).** By the use of this repository, you agree to be bound by these Principles.

## License note

The code in this repository is under MIT license. All data in the repository, as stated in the principles of collaboration, is shared under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/). For the data, the license holders are H. Sebastian Seung, Thomas Euler, Philipp Berens, and Greg Schwartz.

# Eyewire II: Functional data

This repository hosts tools to analyse the functional data from OGB-1 recordings in the Eyewire II dataset.

The following files are included:
- tutorial notebooks to facilitate data exploration [notebooks/tutorial/*.ipynb](notebooks/tutorial/)
- analysis notebooks [notebooks/analysis/*.ipynb](notebooks/analysis/)
- spreadsheet data that are needed to map 2p to EM data (proofread cell main list, 2P-to-EM ROI mapping) in [data/spreadsheets/*](data/spreadsheets/)
- pre-processed calcium traces from recordings over five recording fields, stored as parquet files and downloadable from [Hugging Face](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main/data-2p) (see [data/data-2p/README.md](data/data-2p/README.md))


Documentation is still incomplete:
- a description of the 2P data can be found [here](data/data-2p/README.md).
- a description of the stimuli can be found [here](data/stimuli/README.md).

Feel free to open issues to ask questions and request features!

### Setup

To use the code in this repository out of the box, you can use [uv](https://docs.astral.sh/uv/) to reproduce our python environment. Follow these steps:

- Install `uv`
- Clone this repository and navigate to its root folder
- Run `uv run jupyter lab` to start jupyter lab  - it should open in your browser, and allows you to run our notebooks.

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

See the tutorial notebooks for full usage examples:
- [plot_raw_data.ipynb](notebooks/tutorial/plot_raw_data/plot_raw_data.ipynb) — load data and plot raw + preprocessed traces for individual ROIs
- [plot_data_overview.ipynb](notebooks/tutorial/plot_data_overview/plot_data_overview.ipynb) — plot chirp and bar response overviews grouped by cell type and recording field
- [plot_morph_and_func.ipynb](notebooks/tutorial/plot_morphology/plot_morph_and_func.ipynb) — plot morphology and functional data for individual cells
