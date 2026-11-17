"""Mouse-cam (MC) natural movie stimulus: loading, binning and frame timing.

The MC arrays are generated with ``mc_to_numpy.py`` (see ``data/stimuli/README.md``) and have shape
``(n_frames, 56, 56, 3)``. Channel 0 is unused, channel 1 drove the green and channel 2 the UV LED.
Each MC stimulus consists of 123 five-second clips (test1/train1/test2/train2/test3) with one trigger per clip.
"""
import os

import numpy as np

MC_N_CLIPS = 123
MC_FRAME_RATE_HZ = 30.
MC_N_FRAMES_PER_CLIP = 150
MC_PIXEL_SIZE_UM = 12.5  # QDSpy movScale
MC_CHANNELS = ('green', 'uv')
MC_CHANNEL_IDXS = (1, 2)
MC_BLOCKS = {'test1': (0, 5), 'train1': (5, 59), 'test2': (59, 64), 'train2': (64, 118), 'test3': (118, 123)}


def mc_array_filename(mc_name: str) -> str:
    """Map a stimulus identifier as used in ``df_rois['mc']`` (e.g. ``'mc08'``) to its array file (``'MC8.npy'``)."""
    return f"MC{int(mc_name.lower().removeprefix('mc'))}.npy"


def bin_movie(mov: np.ndarray, bin_xy: int) -> np.ndarray:
    """Spatially bin a movie of shape ``(n_frames, ny, nx, n_channels)`` by averaging ``bin_xy x bin_xy`` pixels."""
    n_frames, ny, nx, n_channels = mov.shape
    assert ny % bin_xy == 0 and nx % bin_xy == 0, (mov.shape, bin_xy)
    return mov.reshape(n_frames, ny // bin_xy, bin_xy, nx // bin_xy, bin_xy, n_channels).mean(axis=(2, 4))


def load_mc_movie(mc_name: str, mc_array_dir: str, bin_xy: int = 1) -> np.ndarray:
    """Load the green and UV channels of an MC movie.

    Args:
        mc_name: Stimulus identifier, e.g. ``'mc16'``.
        mc_array_dir: Folder containing the ``MC*.npy`` arrays.
        bin_xy: Spatial binning factor (1 = native 56x56 pixels).

    Returns:
        np.ndarray: Movie of shape ``(n_frames, ny, nx, 2)`` with intensities in [0, 1], channels ``MC_CHANNELS``.
    """
    mov = np.load(os.path.join(mc_array_dir, mc_array_filename(mc_name)))
    assert mov.shape[0] == MC_N_CLIPS * MC_N_FRAMES_PER_CLIP, mov.shape
    mov = mov[..., list(MC_CHANNEL_IDXS)].astype(np.float32) / 255.
    if bin_xy > 1:
        mov = bin_movie(mov, bin_xy)
    return mov


def get_frame_times(triggertimes: np.ndarray) -> np.ndarray:
    """Frame times from the per-clip triggers; frames within a clip are assumed to be at ``MC_FRAME_RATE_HZ``."""
    assert triggertimes.size == MC_N_CLIPS, triggertimes.size
    return (triggertimes[:, None] + np.arange(MC_N_FRAMES_PER_CLIP)[None, :] / MC_FRAME_RATE_HZ).ravel()


def get_frame_blocks() -> np.ndarray:
    """Block name (``'test1'``, ``'train1'``, ...) of every frame."""
    clip_blocks = np.empty(MC_N_CLIPS, dtype=object)
    for name, (i0, i1) in MC_BLOCKS.items():
        clip_blocks[i0:i1] = name
    return np.repeat(clip_blocks, MC_N_FRAMES_PER_CLIP)
