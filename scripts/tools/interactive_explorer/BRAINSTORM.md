# Interactive EM ↔ Function ↔ Stimulus Explorer — Brainstorm

Goal: an interactive tool where you scrub a session timeline, pick a cell (click on
an XY mosaic or a dropdown), and see its EM skeleton, its chirp/moving-bar functional
responses, and how the light-exposure/stimulus history around its recording looked —
shareable with collaborators without a local dev setup (e.g. via Colab).

This doc is a brainstorm, not a spec. Nothing here is committed to.

**Status:** a quick, deliberately un-fancy v0 exists at `interactive_explorer.py` in this folder — dropdown + clickable-scatter cell picker, EM skeleton + chirp + moving-bar + DS-polar panel. No timeline slider or stimulus overlay yet.

## 1. What already exists in this repo (so we don't reinvent it)

| Need | Existing building block |
|---|---|
| ROI-level data (traces, snippets, quality, DS/OS, xy position) | `data_loader.load_df_rois` → `df_rois` (380 ROIs × 5 fields, columns incl. `temporal_nasal_pos_um`/`ventral_dorsal_pos_um`, `chirp_*`, `bar_*`, `mc_*`) |
| Field-level images / ROI masks | `data_loader.load_df_fields` |
| Retinal outline/mosaic backdrop | `data_loader.load_df_outline`, see `scripts/tutorial/plot_retinal_outline/` |
| 2P ROI → EM segment/nucleus/cell-type join | `data_loader.load_df_rois_morph` (→ `Latest SegID`, cell type, etc.) |
| Skeleton loading | `data_loader.add_skels` → `skeliner.core.Skeleton` (`.nodes`, `.edges`, `.radii`, `.soma`, `.ntype`) |
| Chirp / bar / DS-polar plots | `plot.plot_chirp`, `plot.plot_bar`, `plot.plot_bar_dir_grid`, `plot.plot_ds_on_morph` — all draw onto a caller-supplied `ax`/`fig`, no forced `plt.show()`, so they're embeddable as-is |
| EM skeleton 2D projection | `plot.plot_morph` (via `skeliner.plot.projection`), with retinal-frame rotation via `registration.align_skel` |
| Raw trace + trigger scrubbing | `plot_traces.plot_trace_and_trigger` |
| Cohort overview grid | `plot_dataframe.plot_df_chirp_and_bar` |
| External full 3D EM view | `neuroglancer.py` (builds a shareable Neuroglancer link — opens in browser, not embedded) |
| Stimulus movie reconstruction | `stimulus/stim_movies.py`, `stimulus/stimulus_tools.create_displayed_movie_sequence` |
| Stimulus geometry (moving-bar/spot/box outlines, positioned + scaled to a field of view) | `stimulus/stim_outlines.py` (`movingBar`, `spot`, `box` — return shapely polygons at given `x0,y0,FOV_diam`; already used in `EW2_stim_history.py` to draw stimulus footprints at the correct real-world location) |
| Session-level light-exposure history (spatial + temporal, incl. an adaptation model) | `scripts/analysis/light_exposure/EW2_stim_history.py` (`calc_spatial_exposure`, `calc_temporal_exposure`) + `load_recording_log.py` (parses QDSpy `.log` + `.smh` headers) |

**Nothing here does in-repo 3D skeleton rendering.** `skeliner.plot.vis3d` exists upstream but needs `vtk`, which isn't installed. `plotly` *is* already a declared dependency but currently unused anywhere in `src/` — a natural fit for a lightweight interactive 3D skeleton view (3D line/scatter trace of nodes+edges) without adding a VTK dependency.

**The "stimulus positioned correctly at real-world scale, clipped at the display borders" problem is already solved once**, inside `calc_spatial_exposure` (`EW2_stim_history.py:297-475`): it places a flattened stimulus movie into a fixed-bounds accumulator image using `pix_size_um`-based pixel conversion and explicit clipping math (`acc_x0/acc_x1/acc_y0/acc_y1` vs. array bounds) so stimuli that extend past the display area get correctly cropped rather than erroring or wrapping. That placement/clipping logic is the direct precedent for a "stimulus overlay behind the cell mosaic, live at time *t*" feature — it would just need to run per-timestep instead of accumulated over the whole session, and to become zoom-aware (see below).

## 2. Session scope (resolved)

There is only **one** recording session in this dataset (it's an expensive one-off acquisition, not one of several pilot sessions) — so the light-exposure/timeline feature doesn't need to generalize across sessions; it applies to the whole released dataset already. The "light-exposure/stimulus-history" pipeline (`load_recording_log.py` → `EW2_stim_history.py`) is driven by that session's log file (`20181011_182540.log`), its 5 field positions, and a manually consolidated `data/experiment-overview_consolidated.csv`; it also depends on stimulus-movie pickle files that aren't currently in the repo. Reusing that pipeline as-is (rather than generalizing it) is the right amount of engineering here.

## 3. Content brainstorm — what the explorer actually shows

**Layout sketch:**
```
┌─────────────────────────────────────────────────────────┐
│  session timeline slider  [=========|-------------]      │  ← scrub time
│  (ticks: field switches, stimulus type, light-exposure)   │
├───────────────────────┬───────────────────────────────────┤
│  XY cell picker        │  Selected cell detail:            │
│  (mosaic/outline back- │   - EM skeleton (2D, maybe 3D)     │
│   drop + live stimulus │   - Chirp response                 │
│   footprint at t as    │   - Moving-bar response + DS/OS    │
│   background layer,    │   - Cell type / cluster / soma pos │
│   dots = ROIs on top,  │   - "light history at record time":│
│   click to select;     │     spatial exposure map @ t,       │
│   dropdown as alt.     │     adapted-activity trace @ t,     │
│   input; zoom/pan)     │     marker for when THIS cell was   │
│                        │     actually recorded               │
└───────────────────────┴───────────────────────────────────┘
```

Ideas for what the timeline scrub could drive:
- Highlight which field/recording was active on screen at time *t*.
- Draw the stimulus **currently on screen at time *t*** as a background layer under the cell dots, at its true real-world position and scale (µm), reusing the positioning/scaling/clipping approach from `calc_spatial_exposure` but evaluated live per-frame instead of accumulated. Two fidelity levels worth considering:
  - **Vector outline** (fast, zoom-friendly): the moving-bar/spot/box polygon from `stim_outlines.py` at its current position — scales trivially with zoom since it's geometry in data coordinates, not pixels.
  - **Actual movie raster** (higher fidelity, heavier): the real stimulus frame from `stim_movies.load_qdspy_movie`/`flatten_movie`, resampled into the display's pixel grid — more true-to-life but needs re-rendering (not just re-scaling) whenever the zoom level or view window changes, since it's a raster image at whatever resolution the current view demands. This only exists if the stimulus-movie pickle files are available locally (see §2) -- they aren't currently in the repo.
- A **zoom/pan** control on the cell-picker panel (e.g. Bokeh/Plotly native box-zoom + scroll-zoom) so the stimulus-relative-to-cells relationship can be inspected close up; this pushes toward Bokeh/Plotly-based rendering rather than static matplotlib images, since those already give zoom/pan for free and can redraw a raster layer's extent on zoom callbacks if needed.
- Redraw the spatial exposure accumulation (`calc_spatial_exposure`-style heatmap) truncated to `[0, t]` as an alternate "history so far" view, separate from the "what's on screen right now" live view above.
- Show each field's adapted-activity curve (`calc_temporal_exposure`) with a moving cursor at *t*.
- When a cell is selected, mark on the timeline exactly when its recording happened, so you can visually relate "how much/what light had this retina seen right before this response was recorded."

Optional stretch ideas (not required for v1):
- Link-out button to a live Neuroglancer view of the selected cell (`neuroglancer.spawn_example_cells_link`).
- Overlay cluster/group color coding (via `baden16_utils`) on the XY picker.
- Small multiples: click several cells and compare chirp/bar side by side.

## 4. Gaps to fill

- Promote `calc_spatial_exposure`/`calc_temporal_exposure` (and the log/smh parsing) from the one-off analysis scripts into `src/eyewire2_functional_analysis/` if the timeline feature is meant to be reusable/library-quality rather than a single demo notebook.
- A lightweight "nearest ROI to clicked xy" helper (trivial — `scipy.spatial.cKDTree` or just argmin over `df_rois` xy columns).
- A lightweight 3D skeleton renderer using `plotly` (3D line trace over `skel.edges`/`skel.nodes`) as a VTK-free alternative to `skeliner.plot.vis3d`, if 3D is wanted.
- A "stimulus footprint at time *t*" function: given the parsed stimulus log/timeline, return which stimulus (if any) is on screen at *t* and at which field, then place its outline (`stim_outlines`) or rendered frame (`stim_movies`) at the right µm position — factoring out the placement/clipping math currently embedded inline in `calc_spatial_exposure` so it can be called per-frame rather than only as a whole-session accumulator.
- If the raster-movie-frame fidelity level is wanted: a resampling step that re-renders the background image to match the current zoom/pan extent (naive "one static image" won't stay pixel-correct once you zoom), plus a plan for keeping this responsive (e.g. debounce redraws on zoom, cap raster resolution).

## 5. Tech stack options (all considered against "share easily, ideally via Colab")

| Option | Colab story | Notes |
|---|---|---|
| **Panel** (bokeh-based) | Renders inline in Jupyter/Colab via `pn.extension()`; can *also* be exported with `panel convert` into a **fully static, serverless WASM app** (single HTML file, hostable on GitHub Pages, no server needed at all) | Not currently a dependency. Best long-term "share a link" story. Bokeh plots are interactive (pan/zoom/click) out of the box. |
| **Plotly `FigureWidget` + `ipywidgets`** | Renders natively in Colab notebook output; `ipywidgets` already resolves transitively (via `jupyter`/`notebook`), `plotly` already declared | Lightest lift dependency-wise. Click-to-select needs manual wiring via `on_click` callbacks. No static-export path — always needs a live kernel (Colab is fine for this). |
| **jupyter-dash / Dash** | Can render inline in Colab via `jupyter-dash`, or standalone via a tunnel (ngrok) | More powerful callback model, but heavier, and "standalone app from Colab" needs a tunnel — extra moving part. |
| **Streamlit** | Doesn't run natively inside a Colab *cell*; realistic sharing path is deploying to Streamlit Community Cloud (separate from "open in Colab") | Great DX, but doesn't match the "share via Colab" ask directly. |
| **marimo** | Different notebook paradigm (reactive, not jupytext); has a WASM export → fully static shareable link, no server | Interesting if we're open to stepping outside the jupytext-percent-script convention this repo already uses everywhere else. Probably a bigger convention break than warranted here. |

**Leaning recommendation:** Panel — reuses the already-declared `plotly`/matplotlib pieces, embeds fine in a notebook for local/Colab use during development, and has a real path to a static, serverless shareable build later if that becomes a goal. Plotly+ipywidgets is the fallback if we want to minimize new dependencies for a first cut.

## 6. Data packaging for a shareable demo

- `data/data-2p/*.parquet` (functional data): ~87 MB total — small enough to bundle directly or fetch once at notebook start (matches the Hugging Face dataset source already referenced in `data/data-2p/README.md`).
- `data/swc/*.swc` (EM skeletons): 368 files, ~364 MB total — too heavy to bundle wholesale for a lightweight demo. Options: fetch a skeleton on-demand per selected cell (`huggingface_hub.hf_hub_download`) instead of bundling all of them, or ship a curated demo subset (e.g. a handful of cells spanning a few cell types/fields).
- Everything is CC-BY-NC-4.0 — needs attribution surfaced in whatever notebook/app gets shared.

## 7. Open questions

- Is an embedded 3D EM skeleton view a hard requirement for v1, or is the existing 2D XY projection (`plot.plot_morph`, used in the v0 script) enough to start?
- Preferred sharing mechanism: a notebook people open and run themselves in Colab, a hosted static app link (Panel WASM export), or both?
- Is this meant to become a real feature of `src/eyewire2_functional_analysis/` (reusable, tested-ish), or a self-contained demo script/notebook under `scripts/` (where the v0 currently lives)?
- For the stimulus background overlay: is a vector outline (fast, always zoom-correct, but schematic) good enough, or is showing the actual stimulus movie raster (higher fidelity, heavier, needs the movie pickle files) worth the extra cost?
- Should the overlay default to "what's on screen live at time *t*" (a single instant), "accumulated exposure up to *t*" (a heatmap of history), or a toggle between both?
