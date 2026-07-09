# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Analysis tools for the functional (2P calcium imaging, OGB-1) data of the Eyewire II dataset, plus the code that links that functional data to EM reconstructions/proofreading. It's a research repo (`eulerlab`), not a package with a public API contract — scripts under `scripts/` are the primary "consumers" of the `eyewire2_functional_analysis` library in `src/`.

All data and code is subject to the [Eyewire II Principles of Collaboration](https://eyewire.ai/principles). Code is MIT-licensed; data is CC-BY-NC-4.0 (license holders: H. Sebastian Seung, Thomas Euler, Philipp Berens, Greg Schwartz).

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for environment/dependency management. There is no test suite, linter, or CI config in this repo.

- Install deps / create venv: `uv sync`
- Run any script or tool: `uv run <command>` (installs deps into `.venv` on first call)
- Start Jupyter Lab (to run scripts as notebooks, see below): `uv run jupyter lab`
- Run a single analysis/tutorial script directly: `uv run python scripts/<path>/<script>.py`

## Architecture

### `src/eyewire2_functional_analysis/` — the library

- `paths.py` — `PROJECT_ROOT` constant, resolved from file location.
- `data_loader.py` — the main entry point for loading data. `DATA_2P`, `DATA_SS` (spreadsheets), `DATA_SWC` are default data locations under `data/`. Key functions:
  - `load_df_rois` / `load_df_fields` / `load_df_outline` / `load_all_dfs` — load the pre-processed parquet DataFrames (ROI-level traces+QC, field-level metadata, retinal outline). ROI-level data is split across 5 files (`GCL0`..`GCL4`, one per recording field) and concatenated; a `qfilt` quality-filter boolean column is added based on `bar_qidx`/`chirp_qidx` thresholds.
  - `load_df_rois_morph` — joins ROI-level functional data to the EM/morphology side: 2P ROI (`field`/`roi_id`) → EM-2p mapping sheet → nucleus ID → master proofread-cells spreadsheet. This is the bridge between functional (2P) and structural (EM) data; the underlying registration is fit in `scripts/preprocessing/em-2p-mapping.py`.
  - `add_skels` — attaches SWC skeletons (via `skeliner`) to a DataFrame keyed by `Latest SegID`.
  - Master spreadsheet filenames are hardcoded as constants (`MAIN_ALL_CELLS_SHEET`, `MAP_SHEET`) and are dated/versioned — check `data/spreadsheets/` for the current filenames before changing these.
- `io.py` — `serialize_numpy_arrays`/`restore_numpy_arrays`: parquet can't store nested/multi-dim numpy arrays directly, so array columns are round-tripped through nested lists. `load_parquet_df` in `data_loader.py` calls `restore_numpy_arrays` automatically on every load.
- `plot.py` — the bulk of plotting code: chirp/moving-bar stimulus response plots, direction-selectivity plots on morphology, mosaic/outline plots, scale bars, IPL profile plots, snippet/kernel helpers for moving-bar direction tuning (`get_dir_idx`, `sort_response_matrix`, `preprocess_mb_snippets`, `get_time_dir_kernels`).
- `plot_dataframe.py` / `plot_traces.py` — higher-level plotting over whole DataFrames (grouped chirp+bar overview grids) and raw trace/trigger plotting.
- `skeleton.py` — skeleton (SWC) transforms, e.g. `rotate_skel`.
- `neuroglancer.py` — builds Neuroglancer state/links (layers, annotations) for viewing EM segmentation alongside 2P field mappings; used together with `caveclient`.
- `baden16_utils.py` — mapping between Baden et al. 2016 RGC/AC functional cluster IDs, group IDs, and human-readable names/supergroups.
- `style.py` — `set_rc_params()` applies the project's shared Matplotlib/Seaborn style (`paper.mplstyle`, seaborn `'paper'`/`'ticks'`); call this at the top of plotting scripts for consistent figures.
- `stimulus/` — stimulus-side tools, separate from response analysis: `stim_movies.py` (QDSpy movie loading, intensity trace extraction), `stim_outlines.py` (moving bar / spot / box stimulus geometry), `stimulus_tools.py` (displayed-movie-sequence reconstruction).
- `scanm/` — parsers for ScanM raw acquisition file formats (`.smh`/`.smp` headers and stim buffers, `scanm_global.py` for shared constants/helpers). Used for lower-level access to raw recordings, below the level of the pre-processed parquet data.

### `scripts/` — jupytext "percent format" notebooks-as-scripts

Every script under `scripts/` is a `.py` file with `# %%` cell markers and jupytext front-matter (`format_name: percent`). Open them in Jupyter/JupyterLab (via `uv run jupyter lab`) to run cell-by-cell, or run directly with `uv run python <script>.py`. Figures they produce are typically saved to a `figures/` subfolder next to the script.

- `scripts/tutorial/` — worked examples for loading and plotting the released data (raw traces, chirp/bar response overviews by field/type, morphology+function, retinal outline, stimuli).
- `scripts/analysis/` — actual analysis notebooks (e.g. light adaptation, light-exposure/stimulus-history).
- `scripts/preprocessing/` — one-off/pipeline scripts that produced derived data checked into the repo, notably `em-2p-mapping.py` (fits the 2P↔EM coordinate registration referenced by `data_loader.load_df_rois_morph`).

### `data/` — inputs and reference data (mostly gitignored)

- `data/data-2p/` — the pre-processed parquet files (gitignored; download separately from the [eyewire2-data Hugging Face dataset](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main/data-2p) and place here). See `data/data-2p/README.md`.
- `data/spreadsheets/` — master proofread-cell list and EM-2p mapping CSVs consumed by `data_loader.load_df_rois_morph` (filenames are dated; keep `MAIN_ALL_CELLS_SHEET`/`MAP_SHEET` in `paths.py`/`data_loader.py` in sync when these are updated).
- `data/stimuli/` — stimulus assets (QDSpy movies, moving-bar/global-chirp definitions); gitignored.
- `data/smh/`, `data/swc/` — raw ScanM headers and SWC skeleton files.

## Working conventions

- Data loading always goes through `eyewire2_functional_analysis.data_loader`, not ad-hoc `pd.read_parquet`/`pd.read_csv`, because of the numpy-array round-tripping in `io.py` and the quality-filter/merge logic baked into the loader functions.
- When adding a new analysis, prefer following the existing `scripts/<category>/<name>/<name>.py` + `figures/` layout and jupytext percent-cell format rather than a plain script or a real `.ipynb`.
