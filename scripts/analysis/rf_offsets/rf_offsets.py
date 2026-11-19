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
# # Offsets between 2p ROI centers, EM arbor centers and mouse-cam RF centers
#
# How well do the mouse-cam receptive fields (RFs) line up with the cells? Per ROI, this script computes the offset from
# the 2p ROI center (`temporal_nasal_pos_um`, `ventral_dorsal_pos_um`) to the RF center, and, for ROIs matched to an EM
# cell with a skeleton, via the center of the EM dendritic arbor.
#
# - **RF center**: center of the DoG fit of the better-fitted channel (higher DoG r²) of
#   `df_eyewire2_roi_level_mc_rf.parquet`, mapped from movie pixels to the retinal frame. The movie is centered on the
#   field center and oriented by `receptive_fields.movie_to_retina` (an unverified assumption, tested below).
# - **Hull center**: area centroid of the convex hull of the dendritic (non-axon) skeleton nodes (xy), with the skeleton
#   rotated into the 2p frame by the fitted registration and its soma placed on the ROI center (as in
#   `plot_rf_on_morph`). The ROI -> hull leg is therefore the soma -> arbor-center offset, without any registration
#   translation error.
# - **Center prior**: the stored DoG fits have a soft prior pulling the centers towards the window center, i.e. the
#   field center (SD 100 um), which biases offsets towards the field center. The DoG fits are therefore also redone
#   without that prior (same model otherwise, incl. the green-UV coupling), cached in `results/`.
# - **Movie orientation**: offsets are computed for all 8 flips / rotations of the movie in the retinal frame. With the
#   right orientation, a (registration) shift is the same for all ROIs of a field. With a wrong one, the RF positions
#   are mirrored or rotated about the field center, so the offsets depend on where the ROI is in the field. The
#   orientations are ranked by the pooled within-field RMS of the ROI -> RF offsets (after subtracting the field means).
#
# ROIs: `qfilt`, a valid RF, DoG r² >= `R2_MIN` in the chosen channel and its center at least 2 pixels inside the
# stimulus window. All positions are in um in the retinal frame (x = nasal, y = dorsal).
#
# Outputs (`results/`):
# - `rf_offsets_per_roi.csv`: per ROI, variant and orientation: positions and offsets.
# - `rf_offsets_summary.csv`: mean / median / SD / SEM / n of every offset, per field and over all fields, per variant
#   and orientation, and the within-field RMS.
# - `rf_offsets_mean_shift.json`: the mean shifts of the assumed orientation, per variant.
#
# Figures (`figures/`): `rf_offsets_fields` (field averages with the offsets), `rf_offsets_orientations`.

# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import ConvexHull, QhullError

from eyewire2_functional_analysis import data_loader, plot_rf, receptive_fields as rf, registration, space_mapping, style
from eyewire2_functional_analysis.stimulus import mouse_cam as mc

style.set_rc_params()

# %%
HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
FIG_DIR = os.path.join(HERE, 'figures')
RES_DIR = os.path.join(HERE, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

REG_FILE = os.path.join(data_loader.DATA_REGISTRATION, data_loader.EM_2P_REGISTRATION_FILE)
KEY_COLS = ['experimenter', 'date', 'exp_num', 'field', 'roi_id']
R2_MIN = 0.5  # minimum DoG r² of the chosen channel
EDGE_PX = 2  # the RF center must be at least this many pixels from the window border

# DoG refit without the center prior; the other settings as in scripts/preprocessing/mc_rfs/fit_mc_rfs.py
SRF_FIT_KIND = 'dog'
COUPLING_WEIGHT = 0.05
NOPRIOR_CACHE = os.path.join(RES_DIR, 'dog_fits_noprior.pkl')

VARIANTS = {'prior': 'with center prior (stored fit)', 'noprior': 'without center prior (refit)'}

# Orientations of the movie in the retinal frame. (u, v) = (column, row) offset of a pixel from the window center; the
# retinal offset from the field center is (sx * a, sy * b) * pixel size, with (a, b) = (v, u) if swapped else (u, v).
# `receptive_fields.movie_to_retina` assumes nasal = -row, dorsal = -column.
ORIENTATIONS = {
    f"{'x' if not sw else 'x'}={'+-'[sx < 0]}{'row' if sw else 'col'}, y={'+-'[sy < 0]}{'col' if sw else 'row'}":
        (sw, sx, sy)
    for sw in (False, True) for sx in (1, -1) for sy in (1, -1)}
ASSUMED = 'x=-row, y=-col'
assert ASSUMED in ORIENTATIONS


def movie_px_to_retina_offset(x, y, n_px, pixel_size, orientation):
    """Offset (um, retinal frame) of movie pixel coordinates ``x`` (column), ``y`` (row) from the window center."""
    sw, sx, sy = ORIENTATIONS[orientation]
    c = (n_px - 1) / 2
    u, v = np.asarray(x) - c, np.asarray(y) - c
    a, b = (v, u) if sw else (u, v)
    return sx * a * pixel_size, sy * b * pixel_size


# The assumed orientation must agree with plot_rf (i.e. receptive_fields.movie_to_retina)
_img = np.zeros((28, 28))
_img[3, 20] = 1.  # row 3, column 20
_r, _c = np.unravel_index(np.argmax(rf.movie_to_retina(_img)), _img.shape)  # retinal image: row = y, column = x
assert np.allclose(movie_px_to_retina_offset(20, 3, 28, 1., ASSUMED), (_c - 13.5, _r - 13.5))


def hull_centroid(xy):
    """Area centroid of the convex hull of the points ``xy`` (n, 2), or NaNs if degenerate."""
    try:
        p = xy[ConvexHull(xy).vertices]
    except (QhullError, ValueError):
        return np.array([np.nan, np.nan])
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = cross.sum() / 2
    return np.array([np.sum((p[:, 0] + q[:, 0]) * cross), np.sum((p[:, 1] + q[:, 1]) * cross)]) / (6 * area)


# %% [markdown]
# ## Load data

# %%
df_rois = data_loader.load_df_rois()
df_fields = data_loader.load_df_fields().set_index('field')
df = df_rois.merge(data_loader.load_df_mc_rfs(), on=KEY_COLS, validate='one_to_one')

df_morph = data_loader.load_df_rois_morph(df_rois=df_rois)
df_morph = data_loader.add_skels(df=df_morph, inplace=True)
df_morph = df_morph[df_morph.skel.notnull()]
reg = registration.load_registration(REG_FILE)

# Hull centers (soma placed on the ROI center)
hull = {}
for _, row in df_morph.iterrows():
    roi_xy = (row.temporal_nasal_pos_um, row.ventral_dorsal_pos_um)
    skel = space_mapping.align_and_place_skel(row.skel, reg, field=row.field, target_xy=roi_xy)
    pts = skel.nodes[:, :2]
    if skel.ntype is not None and np.any(skel.ntype != 2):
        pts = pts[skel.ntype != 2]
    hull[tuple(row[KEY_COLS])] = hull_centroid(pts)
df['hull_x'] = [hull.get(tuple(k), (np.nan, np.nan))[0] for k in df[KEY_COLS].itertuples(index=False)]
df['hull_y'] = [hull.get(tuple(k), (np.nan, np.nan))[1] for k in df[KEY_COLS].itertuples(index=False)]
print(f'{len(df)} ROIs with RFs, {np.isfinite(df.hull_x).sum()} with an EM hull')

# %% [markdown]
# ## DoG refit without the center prior

# %%
if os.path.exists(NOPRIOR_CACHE):
    df_np = pd.read_pickle(NOPRIOR_CACHE)
else:
    recs = []
    for _, row in df.iterrows():
        srfs = np.stack([row[f'mc_rf_{ch}_srf'] for ch in mc.MC_CHANNELS])
        rec = dict(zip(KEY_COLS, row[KEY_COLS]))
        if np.all(np.isfinite(srfs)):
            _, params = rf.fit_srfs(srfs, kind=SRF_FIT_KIND, center_prior_weight=0., coupling_weight=COUPLING_WEIGHT)
        else:
            params = [dict(x0=np.nan, y0=np.nan, r2=np.nan) for _ in mc.MC_CHANNELS]
        for ch, p in zip(mc.MC_CHANNELS, params):
            rec.update({f'mc_rf_{ch}_dog_{k}': p[k] for k in ('x0', 'y0', 'r2')})
        recs.append(rec)
    df_np = pd.DataFrame(recs)
    df_np.to_pickle(NOPRIOR_CACHE)
print(f'{len(df_np)} refits without the prior')

# %% [markdown]
# ## Offsets per ROI, variant and orientation

# %%
def rf_centers(df_var):
    """Per ROI: chosen (better-fitted) channel, its DoG center (movie px) and r²."""
    r2 = np.stack([df_var[f'mc_rf_{ch}_dog_r2'].to_numpy(float) for ch in mc.MC_CHANNELS], axis=1)
    i_ch = np.argmax(np.nan_to_num(r2, nan=-np.inf), axis=1)
    rows = np.arange(len(df_var))
    pick = lambda k: np.stack([df_var[f'mc_rf_{ch}_dog_{k}'].to_numpy(float) for ch in mc.MC_CHANNELS], 1)[rows, i_ch]
    return pd.DataFrame(dict(rf_channel=np.array(mc.MC_CHANNELS)[i_ch], rf_r2=r2[rows, i_ch],
                             rf_x0_px=pick('x0'), rf_y0_px=pick('y0')), index=df_var.index)


n_px = df.mc_rf_green_srf.iloc[0].shape[0]
base_cols = KEY_COLS + ['qfilt', 'mc_rf_best_alpha_idx', 'mc_rf_best_cc', 'mc_rf_pixel_size_um', 'temporal_nasal_pos_um',
                        'ventral_dorsal_pos_um', 'field_temporal_nasal_pos_um', 'field_ventral_dorsal_pos_um',
                        'hull_x', 'hull_y']
fits = {'prior': df, 'noprior': df[KEY_COLS].merge(df_np, on=KEY_COLS, how='left', validate='one_to_one')}

out = []
for var, df_var in fits.items():
    d0 = pd.concat([df[base_cols].reset_index(drop=True), rf_centers(df_var.reset_index(drop=True))], axis=1)
    inside = (d0.rf_x0_px.between(EDGE_PX, n_px - 1 - EDGE_PX) & d0.rf_y0_px.between(EDGE_PX, n_px - 1 - EDGE_PX))
    d0 = d0[d0.qfilt & (d0.mc_rf_best_alpha_idx >= 0) & (d0.rf_r2 >= R2_MIN) & inside]
    for ori in ORIENTATIONS:
        d = d0.copy()
        dx, dy = movie_px_to_retina_offset(d.rf_x0_px, d.rf_y0_px, n_px, d.mc_rf_pixel_size_um, ori)
        d['rf_x'], d['rf_y'] = d.field_temporal_nasal_pos_um + dx, d.field_ventral_dorsal_pos_um + dy
        d['roi_x'], d['roi_y'] = d.temporal_nasal_pos_um, d.ventral_dorsal_pos_um
        for a, b in [('roi', 'rf'), ('roi', 'hull'), ('hull', 'rf')]:
            d[f'{a}_to_{b}_dx'], d[f'{a}_to_{b}_dy'] = d[f'{b}_x'] - d[f'{a}_x'], d[f'{b}_y'] - d[f'{a}_y']
        d.insert(0, 'orientation', ori)
        d.insert(0, 'variant', var)
        out.append(d)
df_off = pd.concat(out, ignore_index=True).drop(columns=['qfilt', 'mc_rf_best_alpha_idx', 'temporal_nasal_pos_um',
                                                         'ventral_dorsal_pos_um'])
df_off.to_csv(os.path.join(RES_DIR, 'rf_offsets_per_roi.csv'), index=False)
for var in VARIANTS:
    d = df_off[(df_off.variant == var) & (df_off.orientation == ASSUMED)]
    print(f'{var}: {len(d)} ROIs, {d.hull_x.notna().sum()} with an EM hull; channels {d.rf_channel.value_counts().to_dict()}')

# %% [markdown]
# ## Summary: mean shifts per field and over all fields

# %%
LEGS = ['roi_to_rf', 'roi_to_hull', 'hull_to_rf']


def summarize(d):
    rec = {}
    for leg in LEGS:
        v = d[[f'{leg}_dx', f'{leg}_dy']].dropna().to_numpy()
        n = len(v)
        rec[f'{leg}_n'] = n
        for i, ax in enumerate(('dx', 'dy')):
            rec[f'{leg}_{ax}_mean'] = v[:, i].mean() if n else np.nan
            rec[f'{leg}_{ax}_median'] = np.median(v[:, i]) if n else np.nan
            rec[f'{leg}_{ax}_sd'] = v[:, i].std(ddof=1) if n > 1 else np.nan
            rec[f'{leg}_{ax}_sem'] = rec[f'{leg}_{ax}_sd'] / np.sqrt(n) if n > 1 else np.nan
        rec[f'{leg}_dist_mean'] = np.linalg.norm(v, axis=1).mean() if n else np.nan
        rec[f'{leg}_mean_shift_norm'] = np.hypot(rec[f'{leg}_dx_mean'], rec[f'{leg}_dy_mean'])
    return pd.Series(rec)


def within_field_rms(d, leg='roi_to_rf'):
    """RMS of the offsets after subtracting the mean offset of each field (pooled over fields)."""
    v = d[['field', f'{leg}_dx', f'{leg}_dy']].dropna()
    res = v[[f'{leg}_dx', f'{leg}_dy']] - v.groupby('field')[[f'{leg}_dx', f'{leg}_dy']].transform('mean')
    return float(np.sqrt(np.mean(np.sum(res.to_numpy() ** 2, axis=1))))


rows = []
for (var, ori), d in df_off.groupby(['variant', 'orientation'], sort=False):
    for field, dd in [('all', d)] + list(d.groupby('field')):
        s = summarize(dd)
        s['roi_to_rf_within_field_rms'] = within_field_rms(dd)
        rows.append(pd.concat([pd.Series(dict(variant=var, orientation=ori, assumed=ori == ASSUMED, field=field)), s]))
df_sum = pd.DataFrame(rows)
df_sum.to_csv(os.path.join(RES_DIR, 'rf_offsets_summary.csv'), index=False)

mean_shift = {}
for var in VARIANTS:
    s = df_sum[(df_sum.variant == var) & df_sum.assumed].set_index('field')
    mean_shift[var] = {
        field: {leg: {'dx_um': round(float(r[f'{leg}_dx_mean']), 2), 'dy_um': round(float(r[f'{leg}_dy_mean']), 2),
                      'sem_dx_um': round(float(r[f'{leg}_dx_sem']), 2), 'sem_dy_um': round(float(r[f'{leg}_dy_sem']), 2),
                      'n': int(r[f'{leg}_n'])} for leg in LEGS}
        for field, r in s.iterrows()}
with open(os.path.join(RES_DIR, 'rf_offsets_mean_shift.json'), 'w') as f:
    json.dump(dict(description='Mean offsets in um in the retinal frame (dx = nasal, dy = dorsal), assumed movie '
                               f'orientation ({ASSUMED}); see rf_offsets.py', **mean_shift), f, indent=2)

cols = ['variant', 'field', 'roi_to_rf_n', 'roi_to_rf_dx_mean', 'roi_to_rf_dy_mean', 'roi_to_rf_dist_mean',
        'roi_to_rf_within_field_rms', 'roi_to_hull_n', 'roi_to_hull_dx_mean', 'roi_to_hull_dy_mean',
        'hull_to_rf_dx_mean', 'hull_to_rf_dy_mean']
df_sum[df_sum.assumed][cols].round(1)

# %% [markdown]
# ## Orientation test
#
# Within-field RMS of the ROI -> RF offsets per orientation. The assumed orientation should have (close to) the lowest.

# %%
df_rms = df_sum[df_sum.field == 'all'].pivot(index='orientation', columns='variant',
                                             values='roi_to_rf_within_field_rms').loc[list(ORIENTATIONS)]
print(df_rms.round(1))

fig, ax = plt.subplots(figsize=(6, 3))
w = 0.38
xs = np.arange(len(df_rms))
for i, (var, c) in enumerate(zip(VARIANTS, ['0.3', '0.7'])):
    ax.bar(xs + (i - 0.5) * w, df_rms[var], w, color=c, label=VARIANTS[var])
ax.set_xticks(xs, [o + ('\n(assumed)' if o == ASSUMED else '') for o in df_rms.index], rotation=45, ha='right',
              fontsize=7)
ax.set(ylabel='within-field RMS of\nROI → RF offset (µm)', title='Movie orientation in the retinal frame')
ax.legend(frameon=False, fontsize=7)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'rf_offsets_orientations.{ext}'), dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Field averages with the offsets
#
# Per field (columns) and variant (rows), for the assumed orientation: the field average (ch0) with every ROI center
# (black dot) and an arrow to its RF center. For ROIs with an EM cell the arrow bends at the EM hull center (orange
# diamond); otherwise it is straight. The thick red arrow from the field center is the mean ROI -> RF shift of the field.
# The last column shows all offsets relative to the ROI center, with the mean shift over all fields (red) and per field
# (gray).

# %%
fields = sorted(df_off.field.unique())
C_ARROW, C_HULL, C_MEAN = '#1f77b4', '#ff7f0e', '#d62728'
arrow_kw = dict(arrowstyle='-|>', mutation_scale=7, lw=0.7, color=C_ARROW, shrinkA=0, shrinkB=0, alpha=0.8)
FIELD_VIEW_UM = 150.  # half-width of the per-field panels around the field center


def arrow(ax, xy_from, xy_to, **kw):
    """Arrow clipped to the axes (unlike ``annotate``)."""
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, **{**arrow_kw, **kw}))

fig, axs = plt.subplots(len(VARIANTS), len(fields) + 1, figsize=(3 * (len(fields) + 1), 3.1 * len(VARIANTS)),
                        squeeze=False)
for i, var in enumerate(VARIANTS):
    d_var = df_off[(df_off.variant == var) & (df_off.orientation == ASSUMED)]
    s_var = df_sum[(df_sum.variant == var) & df_sum.assumed].set_index('field')
    for j, field in enumerate(fields):
        ax = axs[i, j]
        fr = df_fields.loc[field]
        fx, fy = fr.field_temporal_nasal_pos_um, fr.field_ventral_dorsal_pos_um
        avg = fr.ch0_average.astype(float)
        lo, hi = np.percentile(avg, [5, 99])
        half = avg.shape[0] * fr.pixel_size_um / 2
        # Image axis 0 = nasal, axis 1 = -dorsal (checked against the ROI mask centroids), as plot_morph.plot_stack_average
        ax.imshow(np.clip((avg - lo) / (hi - lo), 0, 1).T ** 0.5, cmap='gray', interpolation='none',
                  extent=(fx - half, fx + half, fy - half, fy + half))
        d = d_var[d_var.field == field]
        for _, r in d.iterrows():
            if np.isfinite(r.hull_x):
                ax.plot([r.roi_x, r.hull_x], [r.roi_y, r.hull_y], c=C_ARROW, lw=0.7, alpha=0.8)
                arrow(ax, (r.hull_x, r.hull_y), (r.rf_x, r.rf_y))
            else:
                arrow(ax, (r.roi_x, r.roi_y), (r.rf_x, r.rf_y))
        ax.plot(d.hull_x, d.hull_y, 'D', c=C_HULL, ms=2.5, mew=0)
        ax.plot(d.roi_x, d.roi_y, 'o', c='k', ms=2.5, mew=0)
        if field in s_var.index:
            m = s_var.loc[field]
            arrow(ax, (fx, fy), (fx + m.roi_to_rf_dx_mean, fy + m.roi_to_rf_dy_mean),
                  mutation_scale=12, lw=2, color=C_MEAN, alpha=1)
        # Fixed zoom around the field; longer arrows are clipped (see the last column for all offsets)
        ax.set(xlim=(fx - FIELD_VIEW_UM, fx + FIELD_VIEW_UM), ylim=(fy - FIELD_VIEW_UM, fy + FIELD_VIEW_UM),
               aspect='equal')
        n = int(s_var.loc[field, 'roi_to_rf_n']) if field in s_var.index else 0
        mtxt = (f'mean ({s_var.loc[field, "roi_to_rf_dx_mean"]:.0f}, {s_var.loc[field, "roi_to_rf_dy_mean"]:.0f}) µm'
                if n else 'no ROIs')
        ax.set_title(f'{field} (n = {n})\n{mtxt}', fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel(f'{VARIANTS[var]}\ndorsal (µm)', fontsize=8)
        if i == len(VARIANTS) - 1:
            ax.set_xlabel('nasal (µm)', fontsize=8)

    # Summary: all offsets relative to the ROI center
    ax = axs[i, -1]
    ax.axhline(0, c='0.8', lw=0.5)
    ax.axvline(0, c='0.8', lw=0.5)
    ax.plot(d_var.roi_to_rf_dx, d_var.roi_to_rf_dy, 'o', c=C_ARROW, ms=2, alpha=0.5, mew=0)
    ax.plot(d_var.roi_to_hull_dx, d_var.roi_to_hull_dy, 'D', c=C_HULL, ms=2, alpha=0.6, mew=0)
    for field in fields:
        if field in s_var.index:
            m = s_var.loc[field]
            arrow(ax, (0, 0), (m.roi_to_rf_dx_mean, m.roi_to_rf_dy_mean),
                  mutation_scale=8, lw=1, color='0.4', alpha=1)
    m = s_var.loc['all']
    arrow(ax, (0, 0), (m.roi_to_rf_dx_mean, m.roi_to_rf_dy_mean),
                  mutation_scale=12, lw=2, color=C_MEAN, alpha=1)
    lim = 1.05 * np.nanmax(np.abs(d_var[['roi_to_rf_dx', 'roi_to_rf_dy']].to_numpy()))
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
    ax.set_title(f'all fields (n = {int(m.roi_to_rf_n)})\nmean ({m.roi_to_rf_dx_mean:.0f} ± {m.roi_to_rf_dx_sem:.0f}, '
                 f'{m.roi_to_rf_dy_mean:.0f} ± {m.roi_to_rf_dy_sem:.0f}) µm', fontsize=8)
    ax.tick_params(labelsize=6)
    if i == len(VARIANTS) - 1:
        ax.set_xlabel('offset from ROI center, nasal (µm)', fontsize=8)

handles = [Line2D([], [], marker='o', ls='', c='k', ms=4, label='ROI center'),
           Line2D([], [], marker='D', ls='', c=C_HULL, ms=4, label='EM hull center (soma on ROI)'),
           Line2D([], [], c=C_ARROW, label='→ RF center (better-fitted channel)'),
           Line2D([], [], c=C_MEAN, lw=2, label='mean ROI → RF shift')]
fig.suptitle(f'ROI → (EM hull) → RF center offsets; movie orientation {ASSUMED} (assumed)', y=1.0)
plt.tight_layout(h_pad=2.5)
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'rf_offsets_fields.{ext}'), dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## ROI -> RF only
#
# The same mapping without the EM hull: per field the field average with an arrow from each ROI center to its RF center
# and the mean shift of the field (red, from the field center); the last column shows all offsets relative to the ROI
# center with the mean over all fields (red, ± SEM in the title) and per field (gray).
#
# Two versions: all ROIs that pass the criteria above, and only the `TOP_N` RFs with the best test correlation
# (`mc_rf_best_cc`) over all fields, to see whether the shift holds for RFs that are surely meaningful.

# %%
TOP_N = 10


def plot_roi_to_rf(sel, name, title, view_um=FIELD_VIEW_UM, label_cc=False):
    """``sel``: per variant the offset rows (assumed orientation) to plot."""
    fig, axs = plt.subplots(len(VARIANTS), len(fields) + 1, figsize=(3 * (len(fields) + 1), 3.1 * len(VARIANTS)),
                            squeeze=False)
    for i, var in enumerate(VARIANTS):
        d_var = sel[var]
        for j, field in enumerate(fields):
            ax = axs[i, j]
            fr = df_fields.loc[field]
            fx, fy = fr.field_temporal_nasal_pos_um, fr.field_ventral_dorsal_pos_um
            avg = fr.ch0_average.astype(float)
            lo, hi = np.percentile(avg, [5, 99])
            half = avg.shape[0] * fr.pixel_size_um / 2
            ax.imshow(np.clip((avg - lo) / (hi - lo), 0, 1).T ** 0.5, cmap='gray', interpolation='none',
                      extent=(fx - half, fx + half, fy - half, fy + half))
            d = d_var[d_var.field == field]
            for _, r in d.iterrows():
                arrow(ax, (r.roi_x, r.roi_y), (r.rf_x, r.rf_y), lw=1 if label_cc else 0.7)
                if label_cc:
                    ax.text(r.rf_x, r.rf_y, f' {r.mc_rf_best_cc:.2f}', fontsize=6, color=C_ARROW, va='center')
            ax.plot(d.roi_x, d.roi_y, 'o', c='k' if not label_cc else 'yellow', ms=2.5 if not label_cc else 4, mew=0)
            if len(d):
                arrow(ax, (fx, fy), (fx + d.roi_to_rf_dx.mean(), fy + d.roi_to_rf_dy.mean()),
                      mutation_scale=12, lw=2, color=C_MEAN, alpha=1)
                mtxt = f'mean ({d.roi_to_rf_dx.mean():.0f}, {d.roi_to_rf_dy.mean():.0f}) µm'
            else:
                mtxt = 'no ROIs'
            ax.set(xlim=(fx - view_um, fx + view_um), ylim=(fy - view_um, fy + view_um), aspect='equal')
            ax.set_title(f'{field} (n = {len(d)})\n{mtxt}', fontsize=8)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f'{VARIANTS[var]}\ndorsal (µm)', fontsize=8)
            if i == len(VARIANTS) - 1:
                ax.set_xlabel('nasal (µm)', fontsize=8)

        ax = axs[i, -1]
        ax.axhline(0, c='0.8', lw=0.5)
        ax.axvline(0, c='0.8', lw=0.5)
        ax.plot(d_var.roi_to_rf_dx, d_var.roi_to_rf_dy, 'o', c=C_ARROW, ms=2 if not label_cc else 4,
                alpha=0.5 if not label_cc else 0.9, mew=0)
        for field, d in d_var.groupby('field'):
            arrow(ax, (0, 0), (d.roi_to_rf_dx.mean(), d.roi_to_rf_dy.mean()), mutation_scale=8, lw=1, color='0.4',
                  alpha=1)
        mx, my = d_var.roi_to_rf_dx.mean(), d_var.roi_to_rf_dy.mean()
        sx, sy = (d_var[c].std(ddof=1) / np.sqrt(len(d_var)) for c in ('roi_to_rf_dx', 'roi_to_rf_dy'))
        arrow(ax, (0, 0), (mx, my), mutation_scale=12, lw=2, color=C_MEAN, alpha=1)
        lim = 1.1 * np.nanmax(np.abs(d_var[['roi_to_rf_dx', 'roi_to_rf_dy']].to_numpy()))
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
        ax.set_title(f'all fields (n = {len(d_var)})\nmean ({mx:.0f} ± {sx:.0f}, {my:.0f} ± {sy:.0f}) µm', fontsize=8)
        ax.tick_params(labelsize=6)
        if i == len(VARIANTS) - 1:
            ax.set_xlabel('offset from ROI center, nasal (µm)', fontsize=8)

    handles = [Line2D([], [], marker='o', ls='', c='yellow' if label_cc else 'k', ms=4, label='ROI center'),
               Line2D([], [], c=C_ARROW, label='→ RF center (better-fitted channel)'
                      + (', label = test correlation' if label_cc else '')),
               Line2D([], [], c='0.4', label='mean shift per field'),
               Line2D([], [], c=C_MEAN, lw=2, label='mean ROI → RF shift')]
    fig.suptitle(f'{title}; movie orientation {ASSUMED} (assumed)', y=1.0)
    plt.tight_layout(h_pad=2.5)
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.show()


sel_all = {var: df_off[(df_off.variant == var) & (df_off.orientation == ASSUMED)] for var in VARIANTS}
plot_roi_to_rf(sel_all, 'rf_offsets_roi_to_rf', 'ROI → RF center offsets, all ROIs passing the criteria')

sel_top = {var: d.nlargest(TOP_N, 'mc_rf_best_cc') for var, d in sel_all.items()}
for var, d in sel_top.items():
    print(f'{var}: top {TOP_N} test cc {d.mc_rf_best_cc.min():.2f}-{d.mc_rf_best_cc.max():.2f}, '
          f'fields {d.field.value_counts().to_dict()}, mean shift ({d.roi_to_rf_dx.mean():.1f}, '
          f'{d.roi_to_rf_dy.mean():.1f}) um')
    print(d[['field', 'roi_id', 'mc_rf_best_cc', 'rf_channel', 'rf_r2', 'roi_to_rf_dx', 'roi_to_rf_dy']].round(2)
          .to_string(index=False))
plot_roi_to_rf(sel_top, 'rf_offsets_roi_to_rf_top', f'ROI → RF center offsets, top {TOP_N} RFs by test correlation',
               view_um=90., label_cc=True)

# %% [markdown]
# ## Skeleton orientation: does a flipped / rotated arbor match the RFs better?
#
# If the skeleton were mis-oriented in the 2p frame, a flip or rotation `T` of the skeleton about its soma would bring
# the hull center `T h` (`h` = ROI -> hull offset; the hull area centroid is equivariant under linear maps) in line
# with the RF center (`s` = ROI -> RF offset). Per `T`: the least-squares gain `a = sum(s . T h) / sum(|h|^2)` of
# `s ~ a T h` (1 = the RF center sits on the hull center, 0 = unrelated, < 0 = opposite) and its R². All rotations (by
# `theta`) and mirror flips (about an axis at angle `phi`) are tested.
#
# - **raw**: `h` and `s` as they are. This includes the global shift (the mean `s` of a field), so a mean `h` that
#   happens to point the same way as the global shift scores even without any per-cell relation.
# - **field-centered**: `h` and `s` minus their field means, i.e. only whether a cell's RF follows its own arbor.
#
# Control: the best gain over all rotations / flips with `h` shuffled across the ROIs of each field. Uses the stored fits
# (with the center prior) and the assumed movie orientation, for all ROIs with a hull and the top ROIs by test cc.

# %%
SKEL_SETS = (None, 60, 30, TOP_N)  # None: all ROIs with a hull
N_SHUFFLE = 1000
ANGLES = np.arange(0, 360, 1.)


def rot(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def mirror(phi_deg):
    """Mirror about the axis at angle ``phi`` (deg) from the x (nasal) axis."""
    t = np.deg2rad(2 * phi_deg)
    return np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])


TRANSFORMS = np.concatenate([np.stack([rot(a) for a in ANGLES]), np.stack([mirror(a) for a in ANGLES[:180]])])
NAMED = {'as is': 0, 'rot 90': 90, 'rot 180': 180, 'rot 270': 270, 'flip x (N-T)': 360 + 90,
         'flip y (D-V)': 360 + 0, 'swap x, y': 360 + 45, 'swap x, -y': 360 + 135}  # indices into TRANSFORMS


def gains(h, s):
    """Gain of ``s ~ a T h`` per transform of ``TRANSFORMS``; ``h`` (..., n, 2), ``s`` (n, 2)."""
    return np.einsum('...ni,kij,...nj->...k', s, TRANSFORMS, h) / np.sum(h ** 2, axis=(-2, -1))[..., None]


def field_center(v, fld):
    return v - pd.DataFrame(v).groupby(fld).transform('mean').to_numpy()


d_ref = df_off[(df_off.variant == 'prior') & (df_off.orientation == ASSUMED) & df_off.hull_x.notna()]
rng = np.random.default_rng(0)
skel_res, sweeps = [], {}
for n_top in SKEL_SETS:
    d = d_ref if n_top is None else d_ref.nlargest(n_top, 'mc_rf_best_cc')
    set_name = f'all ({len(d)})' if n_top is None else f'top {n_top}'
    fld = d.field.to_numpy()
    h0 = d[['roi_to_hull_dx', 'roi_to_hull_dy']].to_numpy()
    s0 = d[['roi_to_rf_dx', 'roi_to_rf_dy']].to_numpy()
    perms = np.empty((N_SHUFFLE, len(d)), int)
    for f in np.unique(fld):
        idx = np.where(fld == f)[0]
        perms[:, idx] = np.stack([rng.permutation(idx) for _ in range(N_SHUFFLE)])
    for mode in ('raw', 'field-centered'):
        h, s = (h0, s0) if mode == 'raw' else (field_center(h0, fld), field_center(s0, fld))
        g = gains(h, s)
        g_sh = gains(h[perms], s).max(axis=1)
        k = int(np.argmax(g))
        r2 = lambda kk: 1 - np.sum((s - g[kk] * h @ TRANSFORMS[kk].T) ** 2) / np.sum(s ** 2)
        rec = dict(set=set_name, mode=mode, n=len(d), mean_h_dx=h0[:, 0].mean(), mean_h_dy=h0[:, 1].mean(),
                   best=f'rot {ANGLES[k]:.0f}' if k < 360 else f'mirror {ANGLES[k - 360]:.0f}', best_gain=g[k],
                   best_r2=r2(k), shuffle_gain_95=np.percentile(g_sh, 95),
                   p_shuffle=(np.sum(g_sh >= g[k]) + 1) / (N_SHUFFLE + 1), r2_as_is=r2(0))
        rec.update({f'gain {name}': g[kk] for name, kk in NAMED.items()})
        skel_res.append(rec)
        sweeps[(set_name, mode)] = (g, g_sh)
df_skel = pd.DataFrame(skel_res)
df_skel.to_csv(os.path.join(RES_DIR, 'skel_orientation_test.csv'), index=False)
print(df_skel[['set', 'mode', 'n', 'mean_h_dx', 'mean_h_dy', 'gain as is', 'gain rot 180', 'r2_as_is', 'best',
               'best_gain', 'best_r2', 'shuffle_gain_95', 'p_shuffle']].round(3).to_string(index=False))

# %%
set_names = list(dict.fromkeys(df_skel.set))
colors = dict(zip(set_names, ['0.75', '0.55', '0.35', 'k']))
fig, axs = plt.subplots(2, 3, figsize=(13, 6), width_ratios=(1.2, 1, 1))
for i, mode in enumerate(('raw', 'field-centered')):
    ax = axs[i, 0]
    sub = df_skel[df_skel['mode'] == mode].set_index('set')
    w = 0.8 / len(set_names)
    xs = np.arange(len(NAMED))
    for j, set_name in enumerate(set_names):
        ax.bar(xs + (j - (len(set_names) - 1) / 2) * w, [sub.loc[set_name, f'gain {n}'] for n in NAMED], w,
               color=colors[set_name], label=set_name)
    ax.axhline(0, c='k', lw=0.5)
    ax.set_xticks(xs, list(NAMED), rotation=45, ha='right', fontsize=7)
    ax.set(ylabel=f'gain of RF on hull offset\n({mode})')
    if i == 0:
        ax.set_title('Flips / rotations of the skeleton')
        ax.legend(frameon=False, fontsize=7, ncol=2)
    for ax, sl, xlab in [(axs[i, 1], slice(0, 360), 'rotation of the skeleton (deg)'),
                         (axs[i, 2], slice(360, None), 'mirror axis angle from nasal (deg)')]:
        for set_name in set_names:
            g, g_sh = sweeps[(set_name, mode)]
            ang = ANGLES[:len(g[sl])]
            ax.plot(ang, g[sl], c=colors[set_name], label=set_name)
            ax.axhline(np.percentile(g_sh, 95), c=colors[set_name], ls=':', lw=1)
        ax.axhline(0, c='k', lw=0.5)
        ax.set(xlabel=xlab if i == 1 else '', ylabel='gain', xticks=ang[::45])
        if i == 0:
            ax.set_title(('Rotations' if sl.start == 0 else 'Mirror flips') + ' (dotted: shuffle 95%)')
fig.suptitle('Does the RF center follow the EM arbor? Gain a of s ~ a T h per skeleton transform T '
             '(top: raw, bottom: field-centered)', y=1.01)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'skel_orientation_test.{ext}'), dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Top cells: skeleton as is vs rotated by 180 deg
#
# Soma-centered (soma on the ROI center): the skeleton as is (black) and rotated by 180 deg (light), their hull centers
# (orange diamond: as is, open: rotated), and the 2-sigma DoG center ellipse and center (blue cross) of the chosen RF
# channel.

# %%
skels = {tuple(r[KEY_COLS]): (r.skel, r['Cell Type']) for _, r in df_morph.iterrows()}
df_rf_rows = df.set_index(KEY_COLS, drop=False)
top = d_ref.nlargest(TOP_N, 'mc_rf_best_cc')
n_col = (TOP_N + 1) // 2
fig, axs = plt.subplots(2, n_col, figsize=(3 * n_col, 6.4), squeeze=False)
for ax, (_, r) in zip(axs.flat, top.iterrows()):
    key = tuple(r[KEY_COLS])
    row = df_rf_rows.loc[key]
    skel_raw, cell_type = skels[key]
    skel = space_mapping.align_and_place_skel(skel_raw, reg, field=r.field, target_xy=(0., 0.))
    xy = skel.nodes[:, :2]
    seg = np.stack([xy[skel.edges[:, 0]], xy[skel.edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(-seg, colors='0.75', lw=0.4))
    ax.add_collection(LineCollection(seg, colors='k', lw=0.4))
    ax.plot(r.roi_to_hull_dx, r.roi_to_hull_dy, 'D', c=C_HULL, ms=6, mec='k', mew=0.5, zorder=5)
    ax.plot(-r.roi_to_hull_dx, -r.roi_to_hull_dy, 'D', mfc='none', mec=C_HULL, ms=6, mew=1, zorder=5)
    ell = plot_rf.dog_ellipse(row, r.rf_channel)
    if ell is not None:
        ax.plot(*ell, c=C_ARROW, lw=1.2, zorder=4)
    ax.plot(r.roi_to_rf_dx, r.roi_to_rf_dy, 'x', c=C_ARROW, ms=7, mew=2, zorder=6)
    ax.plot(0, 0, 'o', c='r', ms=4, zorder=6)
    lim = max(plot_rf.cell_radius(skel), np.abs([r.roi_to_rf_dx, r.roi_to_rf_dy]).max() + 30, 80)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
    ax.set_title(f'{r.field} ROI {r.roi_id}: {cell_type}\ncc {r.mc_rf_best_cc:.2f}, {r.rf_channel}', fontsize=7)
    ax.tick_params(labelsize=6)
for ax in axs.flat[len(top):]:
    ax.axis('off')
handles = [Line2D([], [], c='k', lw=0.8, label='skeleton as is'), Line2D([], [], c='0.75', lw=0.8, label='rotated 180°'),
           Line2D([], [], marker='D', ls='', c=C_HULL, mec='k', label='hull center as is'),
           Line2D([], [], marker='D', ls='', mfc='none', mec=C_HULL, label='hull center rotated'),
           Line2D([], [], c=C_ARROW, marker='x', label='RF: DoG center, 2σ'),
           Line2D([], [], marker='o', ls='', c='r', label='soma = ROI center')]
fig.suptitle(f'Top {TOP_N} RFs by test correlation: skeleton and RF (µm from the soma, nasal right, dorsal up)', y=1.0)
plt.tight_layout()
fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'skel_orientation_top_cells.{ext}'), dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Per-field alignment corrections
#
# The RF centers of the top cells sit ~30 um nasal of soma and arbor. Two kinds of misalignment are possible, each
# estimated per field from many cells so that it can be applied to the top cells:
#
# 1. **Stimulus vs 2p** (a shift of the projected stimulus relative to the scan field). Estimated as the mean ROI -> RF
#    offset of the field (leave-one-out: without the cell itself), from the MC RFs or, independently of the MC RF
#    estimates, from the moving bar (below). The correction shifts the RF by minus that offset.
# 2. **EM vs 2p** (tissue warping / stretching). The skeleton is placed with the per-field registration rotation only.
#    A per-field affine map from the EM nucleus positions to the 2p ROI positions (all matched cells of the field) also
#    captures anisotropic stretch and shear; its linear part, relative to the rotation, is applied to the skeleton
#    about the soma.
#
# ### Moving-bar RF centers
#
# The bar (8 directions, 1000 um/s; script direction `d` moves along `(sin d, cos d)` in the retinal frame, see
# `data-2p/stimuli/README.md`) passes the stimulus (= field) center at the same time in every direction, and passes an
# RF at `p` `p . u(d) / v` later. Per ROI, the responses (mean over the 3 repeats per direction, upsampled to 10 ms) are
# aligned to a common template by cross-correlation; the lags are fitted with `lag(d) = t0 + p . u(d) / v`, weighted
# by the response amplitude per direction. The direction-independent part of the timing (latency, ON vs OFF edge, bar
# size) goes into `t0`. Precision: the same fit per repeat; `bar_rep_sd` is the SD of the per-repeat estimates.

# %%
BAR_DIRS = np.array([0, 180, 45, 225, 90, 270, 135, 315] * 3)  # snippet order (presentation order)
BAR_V = 1000.  # um/s
BAR_FS = 0.01  # s, upsampled time step
BAR_MAX_LAG = 0.7  # s
BAR_QIDX_MIN, BAR_REP_SD_MAX = 0.6, 50.
BAR_CACHE = os.path.join(RES_DIR, 'bar_centers.pkl')


def bar_traces(row):
    """Snippets upsampled to a common time axis relative to each snippet's trigger, baseline-subtracted."""
    from scipy.interpolate import CubicSpline
    sn = row.bar_snippets
    t_rel = row.bar_snippets_t0 - row.bar_triggertimes_snippets[0]
    grid = np.arange(0.6, 4.2, BAR_FS)
    return np.stack([CubicSpline(np.arange(sn.shape[0]) * row.bar_snippets_dt + t_rel[k],
                                 sn[:, k] - np.median(sn[:4, k]))(grid) for k in range(sn.shape[1])])


def bar_center(tr, idx, n_iter=4):
    """RF center (um, retinal frame, relative to the stimulus center) from the snippets ``idx`` of ``tr``."""
    dirs = np.arange(0, 360, 45)
    x = np.stack([tr[idx][BAR_DIRS[idx] == d].mean(0) for d in dirs])
    amp = x.max(1) - x.min(1)
    w = amp / amp.max()
    n_lag = int(BAR_MAX_LAG / BAR_FS)
    lag_range = np.arange(-n_lag, n_lag + 1)
    lags = np.zeros(len(dirs), int)
    for _ in range(n_iter):
        template = np.mean([np.roll(x[i], -lags[i]) * w[i] for i in range(len(dirs))], 0)
        for i in range(len(dirs)):
            lags[i] = lag_range[np.argmax([np.dot(np.roll(x[i], -l), template) for l in lag_range])]
        lags -= int(np.round(np.average(lags, weights=w)))
    u = np.stack([np.sin(np.deg2rad(dirs)), np.cos(np.deg2rad(dirs))], 1)
    a = np.column_stack([np.ones(len(dirs)), u / BAR_V]) * np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(a, lags * BAR_FS * np.sqrt(w), rcond=None)
    return beta[1:]


if os.path.exists(BAR_CACHE):
    df_bar = pd.read_pickle(BAR_CACHE)
else:
    recs = []
    for _, row in df_rois.iterrows():
        tr = bar_traces(row)
        p = bar_center(tr, np.arange(len(BAR_DIRS)))
        reps = np.array([bar_center(tr, np.arange(8) + 8 * i) for i in range(3)])
        recs.append(dict(zip(KEY_COLS, row[KEY_COLS]), bar_dx_stim=p[0], bar_dy_stim=p[1],
                         bar_rep_sd=float(np.sqrt(np.mean(np.var(reps, axis=0, ddof=1))))))
    df_bar = pd.DataFrame(recs)
    df_bar.to_pickle(BAR_CACHE)

df_bar = df_bar.merge(df_rois[KEY_COLS + ['bar_qidx', 'temporal_nasal_pos_um', 'ventral_dorsal_pos_um',
                                          'field_temporal_nasal_pos_um', 'field_ventral_dorsal_pos_um']], on=KEY_COLS)
df_bar['bar_x'] = df_bar.field_temporal_nasal_pos_um + df_bar.bar_dx_stim
df_bar['bar_y'] = df_bar.field_ventral_dorsal_pos_um + df_bar.bar_dy_stim
df_bar['roi_to_bar_dx'] = df_bar.bar_x - df_bar.temporal_nasal_pos_um
df_bar['roi_to_bar_dy'] = df_bar.bar_y - df_bar.ventral_dorsal_pos_um
df_bar['bar_ok'] = (df_bar.bar_qidx > BAR_QIDX_MIN) & (df_bar.bar_rep_sd <= BAR_REP_SD_MAX)
bar_ok = df_bar[df_bar.bar_ok]
print(f'{len(bar_ok)} / {len(df_bar)} ROIs with a reliable bar center (median repeat SD '
      f'{bar_ok.bar_rep_sd.median():.0f} um)')

# %% [markdown]
# ### Bar vs ROI vs MC
#
# - Bar center vs ROI position within the field (both relative to the field center): if the bar centers are
#   meaningful, they follow the ROI positions (slope ~1), offset by the stimulus shift.
# - Bar vs MC RF center per ROI, both field-centered (i.e. without the field shifts).
# - Per-field mean ROI -> RF offsets from the MC RFs and the bar.

# %%
d_mc = df_off[(df_off.variant == 'prior') & (df_off.orientation == ASSUMED)]
d_bm = d_mc.merge(bar_ok[KEY_COLS + ['bar_x', 'bar_y', 'roi_to_bar_dx', 'roi_to_bar_dy']], on=KEY_COLS)
shift_rows = []
for field in fields + ['all']:
    for src, d, cx, cy in [('MC', d_mc, 'roi_to_rf_dx', 'roi_to_rf_dy'), ('bar', bar_ok, 'roi_to_bar_dx', 'roi_to_bar_dy')]:
        dd = d if field == 'all' else d[d.field == field]
        shift_rows.append(dict(field=field, source=src, n=len(dd), dx=dd[cx].mean(), dy=dd[cy].mean(),
                               sem_dx=dd[cx].sem(), sem_dy=dd[cy].sem()))
df_shift = pd.DataFrame(shift_rows)
df_shift.to_csv(os.path.join(RES_DIR, 'field_shifts_mc_vs_bar.csv'), index=False)
print(df_shift.round(1).to_string(index=False))

fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (c_roi, c_bar, lab) in zip(axs[:2], [('temporal_nasal_pos_um', 'bar_x', 'nasal'),
                                             ('ventral_dorsal_pos_um', 'bar_y', 'dorsal')]):
    c_f = 'field_temporal_nasal_pos_um' if lab == 'nasal' else 'field_ventral_dorsal_pos_um'
    xr, yb = bar_ok[c_roi] - bar_ok[c_f], bar_ok[c_bar] - bar_ok[c_f]
    ax.scatter(xr, yb, s=6, c=[fields.index(f) for f in bar_ok.field], cmap='tab10', vmin=0, vmax=9, alpha=0.7, lw=0)
    slope, icpt = np.polyfit(xr, yb, 1)
    xs_ = np.array([xr.min(), xr.max()])
    ax.plot(xs_, xs_, 'k:', lw=1)
    ax.plot(xs_, slope * xs_ + icpt, c=C_MEAN, lw=1.5)
    ax.set(xlabel=f'ROI, {lab} (µm from field center)', ylabel=f'bar RF center, {lab} (µm)',
           ylim=(-150, 150), title=f'slope {slope:.2f}, r = {np.corrcoef(xr, yb)[0, 1]:.2f}, n = {len(xr)}')
ax = axs[2]
for c_mc, c_bar, mk, lab in [('rf_x', 'bar_x', 'o', 'nasal'), ('rf_y', 'bar_y', 's', 'dorsal')]:
    a = d_bm[c_mc] - d_bm.groupby('field')[c_mc].transform('mean')
    b = d_bm[c_bar] - d_bm.groupby('field')[c_bar].transform('mean')
    ax.scatter(a, b, s=6, marker=mk, alpha=0.6, lw=0, label=f'{lab}: r = {np.corrcoef(a, b)[0, 1]:.2f}')
ax.plot([-150, 150], [-150, 150], 'k:', lw=1)
ax.set(xlim=(-200, 200), ylim=(-200, 200), aspect='equal', xlabel='MC RF center (field-centered, µm)',
       ylabel='bar RF center (field-centered, µm)', title=f'bar vs MC per ROI (n = {len(d_bm)})')
ax.legend(frameon=False, fontsize=7)
ax = axs[3]
ax.axhline(0, c='0.8', lw=0.5)
ax.axvline(0, c='0.8', lw=0.5)
for i, field in enumerate(fields + ['all']):
    c = 'k' if field == 'all' else plt.cm.tab10(i)
    for src, mk in [('MC', 'o'), ('bar', 's')]:
        r = df_shift[(df_shift.field == field) & (df_shift.source == src)].iloc[0]
        ax.errorbar(r.dx, r.dy, xerr=r.sem_dx, yerr=r.sem_dy, fmt=mk, c=c, ms=6 if field == 'all' else 4,
                    mfc=c if src == 'MC' else 'none', lw=0.8, label=f'{field} {src}' if field == 'all' or src == 'MC' else None)
    rm, rb = [df_shift[(df_shift.field == field) & (df_shift.source == s)].iloc[0] for s in ('MC', 'bar')]
    ax.plot([rm.dx, rb.dx], [rm.dy, rb.dy], c=c, lw=0.6)
ax.set(xlabel='mean ROI → RF offset, nasal (µm)', ylabel='dorsal (µm)', aspect='equal',
       title='Per-field shift: MC (filled) vs bar (open)')
ax.legend(frameon=False, fontsize=6, ncol=2)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'bar_vs_mc.{ext}'), dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### EM -> 2p affine per field
#
# Affine fit of the 2p ROI positions on the EM nucleus positions of all matched cells of a field; `S` is its linear part
# relative to the registration rotation `R` (`S = M R^-1`, applied to the already rotated skeleton). Singular values
# below 1: the EM arbor is compressed along that axis to match the 2p frame.

# %%
em_xy = df_morph['Nuc Coords'].str.split(',', expand=True).apply(pd.to_numeric, errors='coerce').to_numpy()[:, :2] \
    * 16 / 1000  # voxels (16 nm) -> um
tp_xy = df_morph[['temporal_nasal_pos_um', 'ventral_dorsal_pos_um']].to_numpy()
em_stretch = {}
for field in fields:
    i = (df_morph.field.to_numpy() == field) & np.isfinite(em_xy).all(axis=1)
    x = np.column_stack([em_xy[i], np.ones(i.sum())])
    coef, *_ = np.linalg.lstsq(x, tp_xy[i], rcond=None)
    rmse_aff = np.sqrt(np.mean(np.sum((tp_xy[i] - x @ coef) ** 2, axis=1)))
    rmse_sim = np.sqrt(np.mean(np.sum((tp_xy[i] - registration.map_coords(em_xy[i], reg, 'em_to_2p', field=field)) ** 2,
                                      axis=1)))
    em_stretch[field] = coef[:2].T @ np.linalg.inv(registration.get_field_rotation_matrix(reg, 'em_to_2p', field=field))
    print(f'{field}: n = {i.sum()}, soma RMSE affine {rmse_aff:.1f} um vs registration {rmse_sim:.1f} um; '
          f'S = {np.round(em_stretch[field], 3).tolist()}, singular values {np.round(np.linalg.svd(em_stretch[field])[1], 3)}')

# %% [markdown]
# ### Corrections compared on the top cells
#
# Stimulus shifts are leave-one-out field means (MC: ROIs passing the RF criteria; bar: ROIs with a reliable bar
# center), so a cell does not correct itself. Per correction and set of top cells: mean distance of the RF center from
# the hull center and from the soma, and the gain / R² of `s ~ a h` (raw, as above).

# %%
def loo_field_mean(d_src, cx, cy, keys):
    """Leave-one-out field mean of (cx, cy) of ``d_src`` for the ROIs ``keys`` (DataFrame with KEY_COLS)."""
    g = d_src.groupby('field')
    tot, n = g[[cx, cy]].sum(), g.size()
    out = []
    for _, k in keys.iterrows():
        t, m = tot.loc[k.field].to_numpy().copy(), n.loc[k.field]
        own = d_src[(d_src[KEY_COLS] == k[KEY_COLS].to_numpy()).all(axis=1)]
        if len(own):
            t, m = t - own[[cx, cy]].to_numpy()[0], m - 1
        out.append(t / m)
    return np.array(out)


def loo_global_mean(d_src, cx, cy, keys):
    out = []
    for _, k in keys.iterrows():
        rest = d_src[~(d_src[KEY_COLS] == k[KEY_COLS].to_numpy()).all(axis=1)]
        out.append(rest[[cx, cy]].mean().to_numpy())
    return np.array(out)


CORRECTIONS = {  # name: (stimulus shift source, apply EM affine to the skeleton)
    'none': (None, False),
    'MC shift (global)': ('MC global', False),
    'MC shift (per field)': ('MC field', False),
    'bar shift (per field)': ('bar field', False),
    'EM affine (per field)': (None, True),
    'bar shift + EM affine': ('bar field', True),
}


def stim_shift(kind, keys):
    if kind is None:
        return np.zeros((len(keys), 2))
    if kind == 'MC global':
        return loo_global_mean(d_mc, 'roi_to_rf_dx', 'roi_to_rf_dy', keys)
    if kind == 'MC field':
        return loo_field_mean(d_mc, 'roi_to_rf_dx', 'roi_to_rf_dy', keys)
    if kind == 'bar field':
        return loo_field_mean(bar_ok, 'roi_to_bar_dx', 'roi_to_bar_dy', keys)
    raise ValueError(kind)


corr_res = []
for n_top in (TOP_N, 30, 60):
    d = d_ref.nlargest(n_top, 'mc_rf_best_cc')
    s0 = d[['roi_to_rf_dx', 'roi_to_rf_dy']].to_numpy()
    h0 = d[['roi_to_hull_dx', 'roi_to_hull_dy']].to_numpy()
    for name, (shift_kind, affine) in CORRECTIONS.items():
        s = s0 - stim_shift(shift_kind, d)
        h = np.einsum('nij,nj->ni', np.stack([em_stretch[f] for f in d.field]), h0) if affine else h0
        gain = np.sum(s * h) / np.sum(h ** 2)
        corr_res.append(dict(top=n_top, correction=name, rf_to_hull_um=np.linalg.norm(s - h, axis=1).mean(),
                             rf_to_soma_um=np.linalg.norm(s, axis=1).mean(), gain=gain,
                             r2=1 - np.sum((s - gain * h) ** 2) / np.sum(s ** 2)))
df_corr = pd.DataFrame(corr_res)
df_corr.to_csv(os.path.join(RES_DIR, 'alignment_corrections.csv'), index=False)
print(df_corr.round(2).to_string(index=False))

fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
for ax, col, lab in zip(axs, ['rf_to_hull_um', 'rf_to_soma_um', 'r2'],
                        ['mean RF → hull center (µm)', 'mean RF → soma (µm)', 'R² of s ~ a h']):
    piv = df_corr.pivot(index='correction', columns='top', values=col).loc[list(CORRECTIONS)]
    w = 0.8 / piv.shape[1]
    for j, (n_top, c) in enumerate(zip(piv.columns, ['k', '0.4', '0.7'])):
        ax.bar(np.arange(len(piv)) + (j - 1) * w, piv[n_top], w, color=c, label=f'top {n_top}')
    ax.set_xticks(np.arange(len(piv)), piv.index, rotation=35, ha='right', fontsize=7)
    ax.set(ylabel=lab)
axs[0].legend(frameon=False, fontsize=7)
fig.suptitle('Alignment corrections on the top cells (by MC test correlation)', y=1.02)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG_DIR, f'alignment_corrections.{ext}'), dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Top cells per correction
#
# One figure per correction, layout as `skel_orientation_top_cells`: skeleton (with the EM affine where applied) and
# its hull center, the corrected MC RF (2-sigma DoG center ellipse and center, blue; the uncorrected center light blue)
# and the bar RF center (purple square, same stimulus shift as the MC RF), if reliable.

# %%
C_BAR = '#8e44ad'
bar_by_key = bar_ok.set_index(KEY_COLS)
top = d_ref.nlargest(TOP_N, 'mc_rf_best_cc')
for name, (shift_kind, affine) in CORRECTIONS.items():
    shifts = stim_shift(shift_kind, top)
    fig, axs = plt.subplots(2, n_col, figsize=(3 * n_col, 6.4), squeeze=False)
    for ax, (_, r), shift in zip(axs.flat, top.iterrows(), shifts):
        key = tuple(r[KEY_COLS])
        row = df_rf_rows.loc[key]
        skel_raw, cell_type = skels[key]
        skel = space_mapping.align_and_place_skel(skel_raw, reg, field=r.field, target_xy=(0., 0.))
        lin = em_stretch[r.field] if affine else np.eye(2)
        xy = skel.nodes[:, :2] @ lin.T
        seg = np.stack([xy[skel.edges[:, 0]], xy[skel.edges[:, 1]]], axis=1)
        ax.add_collection(LineCollection(seg, colors='k', lw=0.4))
        hx, hy = lin @ [r.roi_to_hull_dx, r.roi_to_hull_dy]
        ax.plot(hx, hy, 'D', c=C_HULL, ms=6, mec='k', mew=0.5, zorder=5)
        ell = plot_rf.dog_ellipse(row, r.rf_channel)
        if ell is not None:
            ax.plot(ell[0] - shift[0], ell[1] - shift[1], c=C_ARROW, lw=1.2, zorder=4)
        if shift_kind is not None:
            ax.plot(r.roi_to_rf_dx, r.roi_to_rf_dy, 'x', c=C_ARROW, alpha=0.35, ms=7, mew=2, zorder=6)
        ax.plot(r.roi_to_rf_dx - shift[0], r.roi_to_rf_dy - shift[1], 'x', c=C_ARROW, ms=7, mew=2, zorder=6)
        if key in bar_by_key.index:
            b = bar_by_key.loc[key]
            ax.plot(b.roi_to_bar_dx - shift[0], b.roi_to_bar_dy - shift[1], 's', mfc='none', mec=C_BAR, ms=7, mew=1.5,
                    zorder=6)
        ax.plot(0, 0, 'o', c='r', ms=4, zorder=6)
        lim = max(plot_rf.cell_radius(skel), np.abs([r.roi_to_rf_dx, r.roi_to_rf_dy]).max() + 30, 80)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
        ax.set_title(f'{r.field} ROI {r.roi_id}: {cell_type}\ncc {r.mc_rf_best_cc:.2f}, {r.rf_channel}; '
                     f'shift ({-shift[0]:.0f}, {-shift[1]:.0f}) µm', fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axs.flat[len(top):]:
        ax.axis('off')
    handles = [Line2D([], [], c='k', lw=0.8, label='skeleton' + (' (EM affine)' if affine else '')),
               Line2D([], [], marker='D', ls='', c=C_HULL, mec='k', label='hull center'),
               Line2D([], [], c=C_ARROW, marker='x', label='MC RF: DoG center, 2σ' + (' (corrected)' if shift_kind else '')),
               Line2D([], [], marker='s', ls='', mfc='none', mec=C_BAR, label='bar RF center'),
               Line2D([], [], marker='o', ls='', c='r', label='soma = ROI center')]
    if shift_kind:
        handles.insert(3, Line2D([], [], marker='x', ls='', c=C_ARROW, alpha=0.35, mew=2, label='MC RF center uncorrected'))
    m = df_corr[(df_corr.top == TOP_N) & (df_corr.correction == name)].iloc[0]
    fig.suptitle(f'Top {TOP_N} RFs, correction: {name}  (mean RF → hull {m.rf_to_hull_um:.0f} µm, RF → soma '
                 f'{m.rf_to_soma_um:.0f} µm)', y=1.0)
    plt.tight_layout()
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    slug = name.replace(' (', '_').replace(')', '').replace(' + ', '_').replace(' ', '_').lower()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'top_cells_corrected_{slug}.{ext}'), dpi=200, bbox_inches='tight')
    plt.show()
