from matplotlib import pyplot as plt
import numpy as np
import warnings

from eyewire2_functional_analysis.ds import MB_DIRS, MB_DIRS_SYMBOLS_D_UP, MB_DIRS_SYMBOLS_V_UP, get_time_dir_kernels, preprocess_mb_snippets
from eyewire2_functional_analysis.plot_utils import plot_scale_bar

# Time (s) of stimulus onset / direction reversal / offset within a moving-bar
# snippet, matching the vlines used in plot_dataframe.plot_df_chirp_and_bar.
BAR_STIM_TIMES = (1.152, 2.432, 3.712)

# (row, col, sorted_direction_index) placement of each of the 8 moving-bar
# directions (sorted ascending: 0, 45, 90, ..., 315 deg) around a 3x3 compass
# grid, leaving the center cell (1, 1) free for the polar tuning plot. Matches
# the "manuscript" retinal-orientation convention (data/stimuli/README.md):
# dorsal=top, ventral=bottom, temporal=left, nasal=right, i.e. 0 deg (ventral
# -> dorsal) at the top, with the remaining directions following clockwise
# from there (0, 45, 90, ..., 315 deg -> top, top-right, right, ...).
DIR_GRID_LAYOUT = ((0, 0, 7), (0, 1, 0), (0, 2, 1),
                    (1, 0, 6),           (1, 2, 2),
                    (2, 0, 5), (2, 1, 4), (2, 2, 3))


def get_repeat_colors(n_repeats, cmap='viridis', color_range=(0.15, 0.85)):
    """Colour for each repeat index, as used by `plot_snippets_and_average` -- exposed separately
    so callers (e.g. a colorbar legend keyed by repeat number) can reproduce the exact same mapping.

    Args:
        n_repeats: Number of repeats to generate colours for.
        cmap: Colormap name.
        color_range: ``(low, high)`` fraction of `cmap` to sample from.

    Returns:
        numpy.ndarray: Array of shape ``(n_repeats, 4)`` RGBA colours, indexed by repeat.
    """
    return plt.get_cmap(cmap)(np.linspace(*color_range, n_repeats))


def plot_mean_and_sd(ax, traces, time, color='black', alt_color='dimgray', facealpha=0.2, offset=0.0):
    """Plot the mean of multiple traces with a shaded ±1 SD band.

    If fewer than three traces are provided, individual traces are plotted instead
    of a mean ± SD.

    Args:
        ax: Matplotlib Axes to plot on.
        traces: Array of shape ``(n_traces, time)`` containing the trace data.
        time: 1-D time array aligned with the trace axis.
        color: Colour for the mean line and SD band.
        alt_color: Colour for the second trace when only two traces are provided.
        facealpha: Alpha transparency for the SD fill-between band.
        offset: Scalar offset added to the mean before plotting.
    """
    if traces.shape[0] <= 2:
        ax.plot(time, traces[0] - np.mean(traces[0]) + offset, color=color)
        if len(traces) == 2:
            ax.plot(time, traces[1] - np.mean(traces[1]) + offset, color=alt_color)
    else:
        mu = np.mean(traces, axis=0)
        mu = mu - np.mean(mu) + offset
        sd = np.std(traces, axis=0)

        ax.plot(time, mu, color=color)
        ax.fill_between(time, mu - sd, mu + sd, color=color, alpha=facealpha)


def plot_snippets_and_average(
        ax, time, snippets, average=None, average_time=None,
        vlines=None, vline_ymin=None, vline_ymax=None, hline=False,
        snippet_lw=1.0, snippet_alpha=1.0, average_lw=1.5,
        snippet_cmap='viridis', snippet_color_range=(0.15, 0.85),
        average_color='black', vline_color='gray', vline_ls='--', vline_lw=1.0,
        hline_color='dimgray', hline_ls='--', clip_on=False,
):
    """Plot repeated stimulus-response snippets, colour-coded by repeat, plus their average.

    Shared drawing logic behind `plot.plot_chirp`, `plot.plot_bar`,
    `plot.plot_bar_dir_grid`, and the mouse-cam snippet panel in
    `scripts/tools/interactive_explorer/interactive_explorer.py`: each
    repeat gets a distinct colour from `snippet_cmap` (restricted to
    `snippet_color_range` so no repeat ends up white/near-white, and so the
    same repeat index gets the same colour across separate calls/panels),
    with the average drawn on top. Any per-stimulus specifics (normalising
    snippets, picking which columns belong to one direction/repetition
    group, aligning snippets from different time windows onto a common time
    axis, ...) stay in the caller -- this only draws what's handed to it.

    Args:
        ax: Matplotlib Axes to plot on.
        time: 1-D time array shared by all columns of `snippets`.
        snippets: 2-D array of shape ``(time, n_repeats)``.
        average: Optional precomputed average trace (e.g. a stored/otherwise
            official average, rather than a plain mean of `snippets`). If
            ``None``, computed as ``np.mean(snippets, axis=1)``.
        average_time: Optional time array for `average`, if it uses a
            different sampling than `time`. Defaults to `time`.
        vlines: Optional 1-D array of x-positions to mark with vertical
            lines (trigger times, or known stimulus-event times).
        vline_ymin / vline_ymax: y-extent of `vlines`. Defaults to the full
            data range of `snippets`/`average`.
        hline: If ``True``, draw a dashed horizontal reference line at y=0.
        snippet_lw / snippet_alpha: Style of the individual repeat traces.
        average_lw: Line width of the average trace.
        snippet_cmap: Colormap name used to colour-code repeats by index.
        snippet_color_range: ``(low, high)`` fraction of `snippet_cmap` to
            sample from, avoiding the colormap's darkest/brightest ends.
        average_color: Colour of the average trace.
        vline_color / vline_ls / vline_lw: Style of the optional vertical lines.
        hline_color / hline_ls: Style of the optional horizontal line.
        clip_on: Forwarded to all plot calls.

    Returns:
        matplotlib.lines.Line2D: The average trace's line artist.
    """
    snippets = np.asarray(snippets)
    n_repeats = snippets.shape[1]
    colors = get_repeat_colors(n_repeats, cmap=snippet_cmap, color_range=snippet_color_range)

    for i in range(n_repeats):
        ax.plot(time, snippets[:, i], color=colors[i], lw=snippet_lw, alpha=snippet_alpha, clip_on=clip_on)

    if average is None:
        average = np.mean(snippets, axis=1)
    if average_time is None:
        average_time = time
    avg_line, = ax.plot(average_time, average, color=average_color, lw=average_lw, clip_on=clip_on)
    ax.fill_between(average_time, average, np.zeros_like(average), color=average_color, alpha=0.1, clip_on=clip_on)

    if hline:
        ax.axhline(0, color=hline_color, ls=hline_ls)

    if vlines is not None:
        if vline_ymin is None or vline_ymax is None:
            data_min = min(np.min(snippets), np.min(average))
            data_max = max(np.max(snippets), np.max(average))
        vmin = data_min if vline_ymin is None else vline_ymin
        vmax = data_max if vline_ymax is None else vline_ymax
        ax.vlines(vlines, ymin=vmin, ymax=vmax, colors=vline_color, linestyles=vline_ls, lw=vline_lw, clip_on=clip_on)

    ax.set(xlabel='Times (s)', ylabel='Norm. Ca.')

    return avg_line


def plot_trace_and_trigger(time, trace, triggertimes, trace_norm=None, title=None, ax=None, label=None):
    """Plot a fluorescence trace with trigger-time markers.

    Optionally overlays a normalised version of the trace on a twin y-axis.

    Args:
        time: 1-D time array.
        trace: 1-D fluorescence trace array aligned with ``time``.
        triggertimes: 1-D array of trigger times (in the same units as ``time``).
            Empty arrays are handled gracefully.
        trace_norm: Optional normalised trace to overlay on a twin y-axis.
        title: Optional axes title string.
        ax: Existing Matplotlib Axes to plot on. If ``None``, a new figure is created.
        label: Legend label for the main trace.

    Returns:
        matplotlib.axes.Axes: The primary axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    if title is not None:
        ax.set_title(title)
    ax.plot(time, trace, label=label)
    ax.set(xlabel='time', ylabel='trace')
    if len(triggertimes) > 0:
        vmin, vmax = np.nanmin(trace), np.nanmax(trace)
        vrng = vmax - vmin
        ax.vlines(triggertimes, vmin - 0.22 * vrng, vmin - 0.02 * vrng, color='r', label='trigger', zorder=-2)
    ax.legend(loc='upper right')

    if trace_norm is not None:
        tax = ax.twinx()
        tax.plot(time, trace_norm, ':')
        if len(triggertimes) > 0:
            vmin, vmax = np.nanmin(trace_norm), np.nanmax(trace_norm)
            vrng = vmax - vmin
            tax.vlines(triggertimes, vmin - 0.22 * vrng, vmin - 0.02 * vrng, color='r', label='trigger', ls=':',
                       zorder=-1)
        tax.set(ylabel='normalized')

    return ax


def plot_traces(time, traces, ax=None, title=None):
    """Plot multiple traces on a single axes with automatic alpha scaling.

    Args:
        time: 1-D time array.
        traces: 2-D array of shape ``(n_traces, time)`` or iterable of 1-D arrays.
        ax: Existing Matplotlib Axes to plot on. If ``None``, a new figure is created.
        title: Optional axes title string.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    if title is not None:
        ax.set_title(title)
    for trace in traces:
        ax.plot(time, trace, alpha=np.maximum(1. / len(traces), 0.3))


def get_aligned_snippets_times(snippets_times, raise_error=True, tol=1e-4):
    """Return a single aligned time vector from a 2-D array of snippet time stamps.

    Subtracts the per-snippet offset (first row), checks consistency across snippets,
    and returns the mean time axis.

    Args:
        snippets_times: 2-D array of shape ``(time_points, n_snippets)`` containing
            absolute time stamps for each snippet.
        raise_error: If ``True``, raise a ``ValueError`` when the standard deviation
            across snippets exceeds ``tol``; otherwise issue a warning.
        tol: Maximum acceptable per-sample standard deviation across snippets.

    Returns:
        numpy.ndarray: 1-D array of aligned (mean) time values.

    Raises:
        ValueError: If snippet times are inconsistent and ``raise_error`` is ``True``.
    """
    snippets_times = snippets_times - snippets_times[0, :]

    is_inconsistent = np.any(np.std(snippets_times, axis=1) > tol)
    if is_inconsistent:
        if raise_error:
            raise ValueError(f'Failed to snippet times: max_std={np.max(np.std(snippets_times, axis=1))}')
        else:
            warnings.warn(f'Snippet times are inconsistent: max_std={np.max(np.std(snippets_times, axis=1))}')

    aligned_times = np.mean(snippets_times, axis=1)
    return aligned_times


def plot_mc_test_snippets(ax, row, test_indices=(0, 59, 118)):
    """Zoomed mouse-cam response snippets, overlaid across all 3 test-clip repetitions.

    Adapts the ``axs['C']`` panel of `scripts/tutorial/stimuli/mouse_cam_movies.py`
    (which only showed one repetition) to overlay all of them, aligned to a
    common local time axis, plus their average, via the same
    `plot_traces.plot_snippets_and_average` helper used by `plot.plot_chirp`/
    `plot.plot_bar`/`plot.plot_bar_dir_grid`.
    """
    mc_trace = row.mc_pp_trace
    mc_time = np.arange(mc_trace.size) * row.mc_trace_dt + row.mc_trace_t0
    mc_tt = row.mc_triggertimes
    mc_tt = np.append(mc_tt, mc_tt[-1] + np.median(np.diff(mc_tt)))
    mc_ylim = (mc_trace.min(), mc_trace.max())

    t_common = None
    snippets = []
    for test_i in test_indices:
        t0, t1 = mc_tt[test_i], mc_tt[test_i + 5]
        ilim = (mc_time >= t0) & (mc_time <= t1)
        t_rel = mc_time[ilim] - t0
        if t_common is None:
            t_common = t_rel
        snippets.append(np.interp(t_common, t_rel, mc_trace[ilim]))
    snippets = np.stack(snippets, axis=1)

    rel_tt = mc_tt[test_indices[0]:test_indices[0] + 6] - mc_tt[test_indices[0]]
    plot_snippets_and_average(
        ax, t_common, snippets,
        vlines=rel_tt, vline_ymin=mc_ylim[0], vline_ymax=mc_ylim[1],
    )


def plot_chirp(ax, row, stimulus_ms=None, plot_hline=True, plot_vlines=True, lw=1):
    """Plot chirp stimulus response snippets and their normalised average onto ``ax``.

    Args:
        ax: Matplotlib Axes to plot on.
        row: DataFrame row containing ``'chirp_snippets'``, ``'chirp_snippets_dt'``,
            ``'chirp_average_norm'``, and ``'chirp_average_dt'``.
        stimulus_ms: Optional 1-D array of stimulus values in mV (sampled at 1 ms).
            If provided, a scaled stimulus trace is drawn above the response.
        plot_hline: If ``True``, draw a dashed horizontal line at y=0.
        plot_vlines: If ``True``, draw dashed vertical lines at t=2, 5, 8, 30 s.
    """
    snippets = row['chirp_snippets']
    time = np.arange(snippets.shape[0]) * row['chirp_snippets_dt']
    norm_snippets = snippets / np.max(np.abs(snippets), axis=0, keepdims=True)
    average = row['chirp_average_norm']
    average_time = np.arange(len(average)) * row['chirp_average_dt']

    plot_snippets_and_average(
        ax, time, norm_snippets, average=average, average_time=average_time,
        hline=plot_hline, vlines=[2, 5, 8, 30] if plot_vlines else None,
        snippet_lw=lw, average_lw=lw,
    )
    if stimulus_ms is not None:
        y0 = np.max(row['chirp_average_norm'])
        yrng = np.max(row['chirp_average_norm']) - np.min(row['chirp_average_norm'])
        stimulus_ms_norm = (stimulus_ms - stimulus_ms.min()) / (stimulus_ms.max() - stimulus_ms.min())
        ax.plot(np.arange(len(stimulus_ms)) * 1e-3, stimulus_ms_norm * 0.2 * yrng + y0 * 1.1,
                c='k', clip_on=False, lw=1, solid_capstyle='butt')


def plot_bar(ax, row, annotate_dirs=False, annotate_symbols=False, ventral_up=False, lw=1):
    """Plot moving-bar response snippets for all 8 directions in a single axes.

    Snippets are grouped by direction and plotted consecutively along the time axis.
    Individual repetitions are colour-coded by repeat; the per-direction mean is plotted in black.

    Args:
        ax: Matplotlib Axes to plot on.
        row: DataFrame row containing ``'bar_snippets'`` and ``'bar_snippets_dt'``.
        annotate_dirs: If ``True``, annotate each direction block with its angle in degrees.
        annotate_symbols: If ``True``, annotate each direction block with a Unicode arrow.
        ventral_up: Determines the arrow-symbol convention for direction labels.
    """
    vmax = np.max(row['bar_snippets'])

    if ventral_up:
        mb_symbols = MB_DIRS_SYMBOLS_V_UP
    else:
        mb_symbols = MB_DIRS_SYMBOLS_D_UP

    for i, (dir_deg, symbol) in enumerate(zip(MB_DIRS, mb_symbols)):
        snippets = row['bar_snippets'][:, np.array([0, 8, 16]) + i] / vmax
        time = (np.arange(0, snippets.shape[0]) + (snippets.shape[0] * 1.2 * i)) * row['bar_snippets_dt']
        plot_snippets_and_average(ax, time, snippets, hline=True, snippet_lw=lw, average_lw=lw)
        if annotate_dirs or annotate_symbols:
            x = time[0] + 0.5 * (time[-1] - time[0])
            y = 1.15

            if annotate_dirs:
                ax.text(x, y, f'{dir_deg}°', ha='center', va='top', fontsize=8)
            else:
                ax.text(
                    x, y, symbol,
                    ha='center', va='top',
                    fontsize=10,
                    fontweight='bold',
                    fontname='DejaVu Sans',
                )


def plot_bar_dir(ax, row, ventral_up=False, lw=1, annotate_ticks=True, plot_ref_lines=False):
    """Plot the moving-bar direction-tuning curve (mean +/- min/max across repeats) and
    preferred-direction vector on a polar axes.

    Derives the direction-tuning curve by projecting each individual repeat (not just
    the direction-average) onto the SVD time kernel of the direction-average response,
    then combines it with the fitted preferred-direction vector (`row['bar_pref_dir']`,
    `row['bar_ds_index']`).

    Args:
        ax: Matplotlib polar Axes to plot on.
        row: DataFrame row containing ``'bar_snippets'``, ``'bar_snippets_dt'``,
            ``'bar_pref_dir'``, and ``'bar_ds_index'``.
        ventral_up: Determines the arrow-symbol convention for direction tick labels.
        lw: Line width for the tuning curve.
        annotate_ticks: If ``True``, label the cardinal directions with arrow
            symbols and show radial ticks; if ``False``, hide all ticks (used
            for the compact center cell in `plot_bar_dir_grid`).
        plot_ref_lines: If ``True``, draw light gray horizontal/vertical
            reference lines through the origin (used in `plot_bar_dir_grid` to
            visually separate the polar cell from the surrounding grid).

    Raises:
        ValueError: If ``bar_snippets`` or the projected per-repeat responses contain
            non-finite values.
    """
    if np.any(~np.isfinite(row['bar_snippets'])):
        raise ValueError('bar_snippets not finite')

    sorted_directions_rad, sorted_responses, avg_sorted_resp = preprocess_mb_snippets(snippets=row['bar_snippets'])
    time_component, _ = get_time_dir_kernels(avg_sorted_resp, dt=row['bar_snippets_dt'])

    t, d, r = sorted_responses.shape
    projected = np.reshape(np.reshape(sorted_responses, (t, d * r)).T @ time_component, (d, r))
    mean_resp = np.mean(projected, axis=-1)
    min_resp = np.min(projected, axis=-1)
    max_resp = np.max(projected, axis=-1)

    if np.any(~np.isfinite(projected)):
        raise ValueError('projected response not finite')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    theta = np.append(sorted_directions_rad, sorted_directions_rad[0])
    mean_closed = np.append(mean_resp, mean_resp[0])
    min_closed = np.append(min_resp, min_resp[0])
    max_closed = np.append(max_resp, max_resp[0])

    temp = np.max(np.abs(np.concatenate([mean_resp, min_resp, max_resp])))

    if plot_ref_lines:
        ax.plot((0, np.pi), (temp * 1.2, temp * 1.2), color='gray')
        ax.plot((np.pi / 2, np.pi / 2 * 3), (temp * 1.2, temp * 1.2), color='gray')

    r_min = float(np.min(min_resp))
    if r_min < 0:
        # r=0 no longer sits at the plot center once rorigin is negative, so draw it explicitly
        ax.set_rorigin(r_min * 1.1)
        theta_circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta_circle, np.zeros_like(theta_circle), color='gray', linestyle='--', linewidth=1)
    else:
        ax.set_rmin(0)

    ax.plot([row['bar_pref_dir'], row['bar_pref_dir']], [0, row['bar_ds_index'] * temp], color='k')
    ax.fill_between(theta, min_closed, max_closed, color='red', alpha=0.3)
    ax.plot(theta, mean_closed, color='red', lw=lw)

    if annotate_ticks:
        ax.xaxis.set_tick_params(pad=-20)
        dirs = [0, 90, 180, 270]
        mb_symbols = MB_DIRS_SYMBOLS_V_UP if ventral_up else MB_DIRS_SYMBOLS_D_UP
        ylim_min = r_min * 1.1 if r_min < 0 else 0
        ax.set(xlabel=None, ylabel=None, yticks=[0, temp])
        ax.set_ylim(ylim_min, temp)
        ax.set_xticks(np.deg2rad(dirs))
        ax.set_xticklabels([mb_symbols[np.argmax(np.array(MB_DIRS) == d)] for d in dirs],
                           fontsize=10, fontweight='bold', fontname='DejaVu Sans', color='#999999')
        ax.set_yticklabels([])
    else:
        ax.set_thetalim([0, 2 * np.pi])
        ax.set_yticks([])


def plot_bar_dir_grid(fig, gs, row, plot_hline=True):
    """Plot per-direction moving-bar repeats/averages plus a polar tuning plot in a 3x3 grid.

    Reproduces the DS/OS summary layout of the original DataJoint ``plot1``
    method: the 8 moving-bar directions are arranged around a compass (sorted
    ascending: 0, 45, 90, ..., 315 deg), with a polar tuning plot in the center
    cell showing the direction-tuning curve and the preferred-direction vector.

    Args:
        fig: Matplotlib Figure to add axes to.
        gs: A ``matplotlib.gridspec.GridSpec`` or ``SubplotSpec`` (the latter must
            support ``.subgridspec(3, 3)``) region to lay the 3x3 grid out in.
        row: DataFrame row containing ``'bar_snippets'``, ``'bar_snippets_dt'``,
            ``'bar_pref_dir'``, ``'bar_ds_index'``, ``'bar_ds_pvalue'``,
            ``'bar_pref_or'``, ``'bar_os_index'``, and ``'bar_os_pvalue'``.
        plot_hline: Whether to plot a horizontal line at y=0.

    Returns:
        tuple: ``(axs, ax_polar)`` -- dict of the 8 Cartesian axes keyed by
        ``(row, col)`` grid position, and the center polar Axes.
    """
    sub_gs = gs.subgridspec(3, 3) if hasattr(gs, "subgridspec") else gs

    _, sorted_responses, avg_sorted_resp = preprocess_mb_snippets(row['bar_snippets'])
    dt = row['bar_snippets_dt']

    # Range across individual repeats (not just the average), since those are
    # now drawn too and can have larger excursions than their mean.
    ymin, ymax = np.min(sorted_responses), np.max(sorted_responses)

    time = np.arange(avg_sorted_resp.shape[0]) * dt
    xmin, xmax = time[0], time[-1]

    axs = {}
    for r, c, dir_idx in DIR_GRID_LAYOUT:
        ax = fig.add_subplot(sub_gs[r, c])
        axs[(r, c)] = ax
        plot_snippets_and_average(
            ax, time, sorted_responses[:, dir_idx, :], average=avg_sorted_resp[:, dir_idx],
            vlines=BAR_STIM_TIMES, hline=plot_hline,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelleft=(c == 0), labelbottom=(r == 2))
        if c != 0:
            ax.set_ylabel(None)
        if r != 2:
            ax.set_xlabel(None)

    ax_polar = fig.add_subplot(sub_gs[1, 1], projection='polar', frameon=False)
    plot_bar_dir(ax_polar, row, annotate_ticks=False, plot_ref_lines=True)

    axs[0, 1].set_title(
        f"Moving bar direction tuning\n"
        f"DSI: {row['bar_ds_index']:.2f}, Pref-Dir: {(360 + np.rad2deg(row['bar_pref_dir'])) % 360:3.0f}°; p={row['bar_ds_pvalue']:.2f}\n"
        f"OSI: {row['bar_os_index']:.2f}, Pref-Or:  {(180 + np.rad2deg(row['bar_pref_or'])) % 180:3.0f}°; p={row['bar_os_pvalue']:.2f}"
    )

    return axs, ax_polar
