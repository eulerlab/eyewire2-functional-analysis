# `df_eyewire2_roi_level_mc_rf.parquet`

Mouse-cam (MC) movie receptive fields (RFs) of all ROIs in `df_eyewire2_roi_level_GCL{0..4}.parquet`. One row per ROI.
Join to the ROI tables on `experimenter, date, exp_num, field, roi_id`. Array columns are stored as JSON lists. They
are restored to numpy arrays by `data_loader.load_parquet_df`.

Produced by `scripts/preprocessing/mc_rfs/fit_mc_rfs.py` in eyewire2-functional-analysis.

## Method

- **Stimulus**: the MC movie of the field (`mc` column), green and UV channels, binned 2 x 2 to 28 x 28 pixels of 25 um,
  z-scored. The movie is centered on the recording field.
- **Response**: the positive part of the derivative of `mc_pp_trace`, sampled at the movie frame times (30 Hz).
- **Model**: lasso GLM `y ~ relu(X w + b)`. `X` is the stimulus projected onto a separable cubic B-spline basis: 15
  functions over time lags from -1.2 s to +0.3 s, 14 x 14 over space, and the 2 channels. The fit minimizes the MSE plus
  an L1 penalty on `w`, with the response divided by its std, using FISTA (accelerated proximal gradient with adaptive
  restart, JAX), which gives exact zeros. It is trained on both training sequences (`train1` + `train2`, 2 x 54
  five-second clips) and has no cross-validation. The model and cost are those of the GLM of RFEst
  (https://github.com/berenslab/RFEst).
- **Penalty sweep**: 8 penalties, log-spaced from 1e-4 to 1 relative to `alpha_max`. `alpha_max` is the smallest
  penalty that gives all-zero weights in the linear model, so with the ReLU it is only approximate. The fits run from
  the largest to the smallest penalty, each warm-started from the previous one.
- **Test**: the three repeats of the test sequence (`test1`-`test3`, 25 s each). The first 1.2 s of each repeat are
  skipped. The prediction includes the intercept and the ReLU and is in the units of the response.
- **Selected RF**: the penalty with the best mean test correlation is chosen per ROI. Because it is chosen on the test
  data, `mc_rf_best_cc` is optimistic. Its STRF is split per channel into an sRF and a tRF by SVD (rank 1). The sRFs of
  both channels are then fitted jointly with a difference of Gaussians (DoG). The fit has a soft prior pulling the
  centers towards the window center (SD 100 um) and a soft penalty on the distance between the green and UV centers.

## Per-penalty scores (arrays with one entry per penalty)

- **mc_rf_alphas**: Penalties relative to `alpha_max`, shape `(8,)`.
- **mc_rf_cc**: Test correlation between prediction and response, averaged over the 3 repeats, shape `(8,)`. NaN if the
  prediction is constant, e.g. a ReLU that is always off.
- **mc_rf_mse**: Test MSE, averaged over the 3 repeats, shape `(8,)`. In the raw units of the response, so it is not
  comparable across ROIs.
- **mc_rf_cc_repeats**, **mc_rf_mse_repeats**: Same per repeat, shape `(8, 3)`.
- **mc_rf_frac_nonzero**: Fraction of basis weights that are not exactly zero, shape `(8,)`. (With the Adam solver,
  which gives no exact zeros, the fraction with `|w| > 0.01 max|w|`.)
- **mc_rf_n_iter**: Number of FISTA iterations, shape `(8,)`.
- **mc_rf_converged**: False if the fit ran to the maximum number of iterations, shape `(8,)`.
- **mc_rf_opt_res**: Final optimality residual of the FISTA fit (largest violation of the lasso optimality conditions,
  relative to `alpha_max`; the fit stops once it is small), shape `(8,)`. NaN for Adam.

## Selected penalty

- **mc_rf_best_alpha_idx**: Index of the selected penalty, or -1 if no penalty gave a valid correlation. In that case
  all RF columns are NaN.
- **mc_rf_best_alpha**, **mc_rf_best_cc**, **mc_rf_best_mse**: Penalty, mean test correlation and mean test MSE of the
  selected penalty.

## RF of the selected penalty (per channel `{ch}` = `green`, `uv`)

sRF, tRF and DoG amplitudes are flipped so that the fitted DoG center is positive, i.e. the tRF carries the center
polarity. For example, an OFF-center RF has a positive sRF center and a tRF that is negative before the response.
The sRF scaled by the tRF approximates the STRF in units of the response per unit of the z-scored stimulus.

- **mc_rf_{ch}_srf**: sRF, shape `(28, 28)` `(y, x)` in stimulus (movie) pixel coordinates, not rotated or mirrored
  into retinal coordinates.
- **mc_rf_{ch}_trf**: tRF, normalized to a maximum absolute value of 1, shape `(46,)`.
- **mc_rf_trf_t0**, **mc_rf_trf_dt**: Time of the first tRF sample (-1.2 s) and the sampling interval (1/30 s).
  Negative times are stimulus frames before the response.
- **mc_rf_pixel_size_um**: sRF pixel size (25 um).
- **mc_rf_{ch}_svd_ev**: Fraction of the STRF variance explained by the rank-1 (sRF x tRF) approximation.
- **mc_rf_{ch}_dog_polarity**: Sign of the DoG center in the unflipped sRF, where the SVD makes the sRF peak positive.
  -1 means the sRF was flipped.
- **mc_rf_{ch}_dog_amp**: Center peak amplitude (positive after the flip).
- **mc_rf_{ch}_dog_x0**, **mc_rf_{ch}_dog_y0**: Center position in pixels. (0, 0) is the center of the top-left pixel;
  the window center is (13.5, 13.5).
- **mc_rf_{ch}_dog_sx**, **mc_rf_{ch}_dog_sy**, **mc_rf_{ch}_dog_theta**: Center Gaussian sigmas (pixels) and angle
  (radians).
- **mc_rf_{ch}_dog_amp_s**: Surround peak amplitude (opposite sign to the center).
- **mc_rf_{ch}_dog_sx_s**, **mc_rf_{ch}_dog_sy_s**: Surround sigmas (pixels). The surround is concentric with the center
  and has the same angle and aspect ratio.
- **mc_rf_{ch}_dog_surround_ratio**: Surround / center sigma ratio `k` (between 2 and 10).
- **mc_rf_{ch}_dog_surround_rel_amp**: Surround / center peak ratio `w` (between 0 and 1). The relative surround volume
  is `w k^2`.
- **mc_rf_{ch}_dog_r2**: Explained variance of the DoG fit of this channel's sRF.
