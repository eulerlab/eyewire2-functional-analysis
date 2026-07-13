from matplotlib import patches as patches
import numpy as np


def plot_scale_bar(
        ax, x0=0, y0=0, size=1000, tdist=70,
        fontsize=10, text=True, unit="µm", orientation='h'
):
    """
    Draws a horizontal or vertical scale bar.

    orientation: 'h' for horizontal, 'v' for vertical
    """

    if orientation == 'h':
        # horizontal bar
        ax.plot(
            [-size / 2 + x0, +size / 2 + x0],
            [y0, y0],
            c='k', solid_capstyle='butt', clip_on=False
        )
        if text:
            ax.text(
                x0, y0 - tdist,
                f'{size:.0f} {unit}',
                c='k', va='top', ha='center', fontsize=fontsize
            )

    elif orientation == 'v':
        # vertical bar
        ax.plot(
            [x0, x0],
            [-size / 2 + y0, +size / 2 + y0],
            c='k', solid_capstyle='butt', clip_on=False
        )
        if text:
            ax.text(
                x0 + tdist, y0,
                f'{size:.0f} {unit}',
                c='k', va='center', ha='left', fontsize=fontsize
            )

    else:
        raise ValueError("orientation must be 'h' or 'v'")


def get_extent(stack_avg, pixel_size_um, x_offset, y_offset):
    """Compute the ``[left, right, bottom, top]`` extent for a square image stack average.

    Args:
        stack_avg: 2-D (or higher) array whose first two dimensions define the
            square pixel grid.
        pixel_size_um: Physical size of one pixel in µm.
        x_offset: X coordinate of the image centre in µm.
        y_offset: Y coordinate of the image centre in µm.

    Returns:
        numpy.ndarray: 1-D array ``[xmin, xmax, ymin, ymax]`` in µm.
    """
    ps = pixel_size_um
    w, h = stack_avg.shape[:2]
    assert w == h
    extent = np.array([-w / 2 * ps, +w / 2 * ps, -w / 2 * ps, +w / 2 * ps])
    extent += (x_offset, x_offset, y_offset, y_offset)
    return extent


def add_rect(ax, box_xlim, box_ylim, color_crop, linewidth=1.2):
    """Add a dashed rectangular patch to the axes.

    Args:
        ax: Matplotlib Axes to add the rectangle to.
        box_xlim: ``(xmin, xmax)`` of the rectangle in data coordinates.
        box_ylim: ``(ymin, ymax)`` of the rectangle in data coordinates.
        color_crop: Edge colour of the rectangle.
        linewidth: Line width of the rectangle edge.
    """
    rect = patches.Rectangle(
        (box_xlim[0], box_ylim[0]), box_xlim[1] - box_xlim[0], box_ylim[1] - box_ylim[0],
        linewidth=linewidth, edgecolor=color_crop, facecolor='none', linestyle='--', clip_on=False
    )
    ax.add_patch(rect)
