from matplotlib import pyplot as plt
import numpy as np


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
    colors = plt.get_cmap(snippet_cmap)(np.linspace(*snippet_color_range, n_repeats))

    for i in range(n_repeats):
        ax.plot(time, snippets[:, i], color=colors[i], lw=snippet_lw, alpha=snippet_alpha, clip_on=clip_on)

    if average is None:
        average = np.mean(snippets, axis=1)
    if average_time is None:
        average_time = time
    avg_line, = ax.plot(average_time, average, color=average_color, lw=average_lw, clip_on=clip_on)

    if hline:
        ax.axhline(0, color=hline_color, ls=hline_ls)

    if vlines is not None:
        if vline_ymin is None or vline_ymax is None:
            data_min = min(np.min(snippets), np.min(average))
            data_max = max(np.max(snippets), np.max(average))
        vmin = data_min if vline_ymin is None else vline_ymin
        vmax = data_max if vline_ymax is None else vline_ymax
        ax.vlines(vlines, ymin=vmin, ymax=vmax, colors=vline_color, linestyles=vline_ls, lw=vline_lw, clip_on=clip_on)

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