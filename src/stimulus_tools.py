import numpy as np


TRAIN_CLIPS = 108
_TRAIN_CLIPS_HALF = TRAIN_CLIPS // 2
FRAMES_PER_SECOND = 30
FRAMES_PER_CLIP = 150


def create_displayed_movie_sequence(
        train_movie: np.ndarray,
        test_movie: np.ndarray,
        random_sequence: np.ndarray,
        scan_sequence_id: int,
) -> np.ndarray:
    """
    Create stimulus movie shown in recording based on train and test movie and the specific random sequence used.
    :param train_movie: shape: (channel, time, height, width)
    :param test_movie: shape: (channel, time, height, width)
    :param random_sequence: shape: (num_clips, num_scan_sequences)
    :param scan_sequence_id: integer to define what scan sequence was used
    :return: the full movie in the desired random sequence
    """
    # add test movie shown at the start
    full_movie_array = [test_movie]

    # add first 54 train clips
    for clip_id in random_sequence[:_TRAIN_CLIPS_HALF, scan_sequence_id]:
        clip_start_idx = int(clip_id) * FRAMES_PER_CLIP  # cast uint8 to integer to avoid overflow
        clip_shown = train_movie[:, clip_start_idx:clip_start_idx + FRAMES_PER_CLIP]
        full_movie_array.append(clip_shown)

    # add test movie shown in the middle
    full_movie_array.append(test_movie)

    # add second 54 stimulus clips
    for clip_id in random_sequence[_TRAIN_CLIPS_HALF:, scan_sequence_id]:
        clip_start_idx = int(clip_id) * FRAMES_PER_CLIP  # cast uint8 to integer to avoid overflow
        clip_shown = train_movie[:, clip_start_idx:clip_start_idx + FRAMES_PER_CLIP]
        full_movie_array.append(clip_shown)

    # add test movie shown at the end
    full_movie_array.append(test_movie)
    # concatenate array over temporal dimension
    full_movie = np.concatenate(full_movie_array, axis=1)
    return full_movie
