# `df_eyewire2_roi_level_GCL{0..4}.parquet`

Loaded via `data_loader.load_df_rois()` (concatenates all 5 files and adds `qfilt`). One row per ROI (cell). 380 rows x 84 columns across the 5 files combined.

## General / recording metadata

- **experimenter**: Name of the experimenter who conducted the recording.
- **date**: Date of the experiment.
- **exp_num**: Experiment number for the given date. Typically corresponds to the eye: 1 = left eye, 2 = right eye.
- **raw_id**: Internal recording/table index from the acquisition pipeline; not meaningful for analysis on its own.
- **field**: Field identifier in the experiment ("GCL" + a number indicating the sequential order of the field within the recording session).
- **region**: Anatomical region of the recording field. Values seen: "RR" (Right Retina).
- **cond1**, **cond2**: Experimental condition identifiers for the recording (e.g. "control"); distinguish e.g. drug/control conditions when an experiment has more than one.
- **ch0_name**, **ch1_name**: Names of the two acquisition channels (e.g. `wDataCh0` = OGB-1/calcium indicator channel, `wDataCh1` = SR-101/structural astrocyte-marker channel, based on `ch0_average`/`ch1_average` usage elsewhere in the codebase).
- **roi_id**: ROI identifier within the field.
- **chirp**, **bar**: Stimulus identifiers for the chirp (`gChirp`) and moving-bar (`movingbar`) stimulus protocols.
- **preprocess_id**, **classifier_id**: Internal pipeline version/step identifiers (which preprocessing/classifier run produced this row); not meaningful for analysis on their own.
- **chirp_stim_name**, **bar_stim_name**: Stimulus protocol names, redundant with `chirp`/`bar`.
- **genline**: Genetic line of the animal (typically "Bl6").
- **animgender**: Sex of the animal.
- **eye**: Eye used for recording ("left" or "right").
- **setupid**: Identifier of the physical 2P rig/setup used for acquisition.
- **prepwmorient**: Orientation of the eye-cup preparation in degrees. 0 deg indicates dorsal is toward the back of the chamber; -1 indicates unknown orientation.
- **nxpix**, **nypix**: Width/height of the recorded field in pixels.
- **pixel_size_um**: Physical size of one pixel in micrometers.
- **field_ventral_dorsal_pos_um**, **field_temporal_nasal_pos_um**: Position of the recording field's center relative to the optic nerve, in micrometers (negative ventral/temporal, positive dorsal/nasal) — same value repeated for all ROIs in a field, see the corresponding columns in `df_eyewire2_field_level.parquet`.
- **ch0_average**, **ch1_average**: Average-intensity images (2D, `nxpix` x `nypix`) of the two acquisition channels for the ROI's field.

## ROI geometry

- **roi_size_um2**: Area of the ROI mask in square micrometers.
- **roi_dia_um**: Circle-equivalent diameter of the ROI mask in micrometers.
- **ventral_dorsal_pos_um**: Ventral-dorsal position of this ROI relative to the optic nerve, in micrometers (negative = ventral, positive = dorsal).
- **temporal_nasal_pos_um**: Temporal-nasal position of this ROI relative to the optic nerve, in micrometers (negative = temporal, positive = nasal).

## Chirp stimulus responses

- **chirp_trace**: Raw fluorescence trace during chirp stimulus presentation (1D array, one value per recording sample, all repeats concatenated).
- **chirp_trace_t0**, **chirp_trace_dt**: Start time and sampling interval (s) of `chirp_trace`.
- **chirp_pp_trace**: Preprocessed (detrended/normalized) version of `chirp_trace`.
- **chirp_pp_trace_t0**, **chirp_pp_trace_dt**: Start time and sampling interval (s) of `chirp_pp_trace`.
- **chirp_qidx**: Quality index (signal-to-noise ratio) of the chirp response for this ROI.
- **chirp_pres_qidx**: Fraction of ROIs in this ROI's field that pass the chirp quality threshold.
- **chirp_average_norm**: Normalized chirp response, averaged across repeats (1D array).
- **chirp_average_dt**: Sampling interval (s) of `chirp_average_norm`.
- **chirp_snippets**: Per-repeat chirp response snippets, shape `(time_points, n_repeats)`.
- **chirp_snippets_t0**: Start time of each repeat in `chirp_snippets` (1D array, one per repeat).
- **chirp_snippets_dt**: Sampling interval (s) of `chirp_snippets`.
- **chirp_triggertimes_snippets**: Stimulus trigger times per repeat, used to align `chirp_snippets` (2D array, `n_triggers_per_repeat` x `n_repeats`).

## Moving-bar stimulus responses

- **bar_trace**: Raw fluorescence trace during moving-bar stimulus presentation (1D array).
- **bar_trace_t0**, **bar_trace_dt**: Start time and sampling interval (s) of `bar_trace`.
- **bar_pp_trace**: Preprocessed (detrended/normalized) version of `bar_trace`.
- **bar_pp_trace_t0**, **bar_pp_trace_dt**: Start time and sampling interval (s) of `bar_pp_trace`.
- **bar_qidx**: Quality index for the moving-bar response, calculated using only the preferred direction.
- **bar_pres_qidx**: Fraction of ROIs in this ROI's field that pass the moving-bar quality threshold.
- **bar_ds_index**: Direction-selectivity index (DSi).
- **bar_ds_pvalue**: Statistical p-value for direction selectivity.
- **bar_pref_dir**: Preferred direction in radians.
- **bar_os_index**: Orientation-selectivity index (OSi).
- **bar_os_pvalue**: Statistical p-value for orientation selectivity.
- **bar_pref_or**: Preferred orientation in radians. Not corrected for eye orientation (left vs. right) or experimental setup geometry.
- **bar_dir_component**: Direction-tuning component from a rank-1 SVD decomposition of the direction-sorted moving-bar snippet matrix (one value per stimulus direction, see `ds.get_time_dir_kernels`).
- **bar_time_component**: Time-kernel component from the same SVD decomposition (one value per time point within a snippet).
- **bar_snippets**: Per-direction moving-bar response snippets, shape `(time_points, n_directions)`.
- **bar_snippets_t0**: Start time of each direction snippet in `bar_snippets` (1D array).
- **bar_snippets_dt**: Sampling interval (s) of `bar_snippets`.
- **bar_triggertimes_snippets**: Stimulus trigger times per direction, used to align `bar_snippets`.

## Mouse-cam (natural movie) stimulus responses

- **mc**: Identifier of the mouse-cam movie clip shown (e.g. `mc16`).
- **mc_trace**: Raw fluorescence trace during mouse-cam movie presentation (1D array).
- **mc_trace_t0**, **mc_trace_dt**: Start time and sampling interval (s) of `mc_trace`.
- **mc_pp_trace**: Preprocessed (detrended/normalized) version of `mc_trace`.
- **mc_pp_trace_t0**, **mc_pp_trace_dt**: Start time and sampling interval (s) of `mc_pp_trace`.
- **mc_triggertimes**: Stimulus trigger times marking movie-clip boundaries/repeats during the mouse-cam presentation.

## Cell type classification (Baden et al. 2016)

Computed via `baden16_utils.baden16_cluster_probs_to_info` from `probs_per_cluster`.

- **cluster_id**: Predicted Baden et al. 2016 functional cluster ID (1-75).
- **group_id**: Predicted Baden et al. 2016 functional group ID (1-46), derived from `cluster_id`.
- **supergroup**: Predicted broad functional supergroup label (e.g. "OFF", "ON-OFF", "Fast ON", "dAC"), derived from `group_id`.
- **prob_cluster**: Predicted probability of the assigned `cluster_id`.
- **prob_group**: Summed probability across all clusters belonging to the assigned `group_id`.
- **prob_supergroup**: Summed probability across all clusters belonging to the assigned `supergroup`.
- **prob_class**: Predicted probability of the broad RGC-vs-dAC class implied by `supergroup`.
- **probs_per_cluster**: Full probability vector (75 values) across all Baden et al. 2016 clusters.

## Derived

- **qfilt**: Boolean quality-filter flag, added by `data_loader.load_df_rois`. `True` if `bar_qidx > 0.6` or `chirp_qidx > 0.45`.
