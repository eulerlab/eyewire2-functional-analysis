# `df_eyewire2_outline.parquet`

Loaded via `data_loader.load_df_outline()`. One row per retinal-outline scan point. 24 rows x 16 columns.

Per `scripts/tutorial/plot_retinal_outline/plot_retinal_outline.py`, these are edge-of-retina scans taken around the perimeter of the eye-cup (plus one center reference point) so that, connected in order, they trace the outline of the retina; the first row is actually the optic disc, used as the coordinate-system origin.

- **experimenter**: Name of the experimenter who conducted the recording.
- **date**: Date of the experiment.
- **exp_num**: Experiment number for the given date. Typically corresponds to the eye: 1 = left eye, 2 = right eye.
- **field**: Outline scan-point identifier (e.g. `Wings00`), sequential around the retinal perimeter.
- **raw_id**: Internal recording/table index from the acquisition pipeline; not meaningful for analysis on its own.
- **field_data_file**: Path to the raw ScanM acquisition file (`.smp`) for this scan point on the original acquisition server (provenance record; the file itself is not part of this repo/dataset).
- **pixel_size_um**: Physical size of one pixel in micrometers.
- **absx**, **absy**, **absz**: Absolute microscope/stage position of this scan point in micrometers (raw acquisition coordinate frame, before referencing to the optic disc).
- **relx**, **rely**, **relz**: Position of this scan point in micrometers relative to the optic disc (first row), in the raw (un-rotated) stage coordinate frame.
- **ventral_dorsal_pos_um**: Ventral-dorsal position relative to the optic nerve in micrometers (`relx`/`rely` rotated into the anatomical axis via `prepwmorient`; negative = ventral, positive = dorsal).
- **temporal_nasal_pos_um**: Temporal-nasal position relative to the optic nerve in micrometers (negative = temporal, positive = nasal).
- **stack_averages**: Average-intensity structural image stack at this scan point, shape `(128, 128, 2)` — the two acquisition channels (OGB-1, SR-101), analogous to `ch0_average`/`ch1_average` in the field-/ROI-level DataFrames but stacked into a single array.
