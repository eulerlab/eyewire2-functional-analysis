# `df_eyewire2_field_level.parquet`

Loaded via `data_loader.load_df_fields()`. One row per recording field. 5 rows x 22 columns.

- **experimenter**: Name of the experimenter who conducted the recording.
- **date**: Date of the experiment.
- **exp_num**: Experiment number for the given date. Typically corresponds to the eye: 1 = left eye, 2 = right eye.
- **raw_id**: Internal recording/table index from the acquisition pipeline; not meaningful for analysis on its own.
- **field**: Field identifier ("GCL" + a number indicating the sequential order of the field within the recording session).
- **region**: Anatomical region of the recording field. Values seen: "RR" (Right Retina).
- **cond1**: Experimental condition identifier for the recording (e.g. "control").
- **ch0_name**, **ch1_name**: Names of the two acquisition channels (`wDataCh0` = OGB-1/calcium indicator channel, `wDataCh1` = SR-101/structural astrocyte-marker channel).
- **genline**: Genetic line of the animal (typically "Bl6").
- **animgender**: Sex of the animal.
- **eye**: Eye used for recording ("left" or "right").
- **setupid**: Identifier of the physical 2P rig/setup used for acquisition.
- **prepwmorient**: Orientation of the eye-cup preparation in degrees. 0 deg indicates dorsal is toward the back of the chamber; -1 indicates unknown orientation.
- **nxpix**, **nypix**: Width/height of the recorded field in pixels.
- **pixel_size_um**: Physical size of one pixel in micrometers.
- **field_ventral_dorsal_pos_um**: Ventral-dorsal position of the field's center relative to the optic nerve, in micrometers (negative = ventral, positive = dorsal).
- **field_temporal_nasal_pos_um**: Temporal-nasal position of the field's center relative to the optic nerve, in micrometers (negative = temporal, positive = nasal).
- **ch0_average**, **ch1_average**: Average-intensity images (2D, `nxpix` x `nypix`) of the two acquisition channels for the field.
- **roi_mask**: 2D integer array (`nxpix` x `nypix`) segmenting the field into ROIs. Each unique positive value identifies one ROI (matching `roi_id` in the ROI-level DataFrame); zero/negative values are background (see `plot_morph.plot_roi_mask`).
