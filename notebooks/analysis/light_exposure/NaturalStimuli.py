# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Natural Stimulus Explanation

# %%
from typing import Any
import pickle
import os

# %% [markdown]
# ## Download and inspect the stimulus file
# You can download the natural stimulus in a normalized version that is useful for neural network training from huggingface [here](https://huggingface.co/datasets/open-retina/open-retina/blob/main/euler_lab/hoefling_2024/stimuli/rgc_natstim_72x64_joint_normalized_2024-10-11.zip).
# If you unzip this file you will find a pickle file called `2024-10-11_RGC_natural_movies_dict_30Hz_72x64_joint_normalised.pkl`. The following code assumes that you store this file in your download folder at the following location (change this path according to your setup):

# %%
STIMULUS_FILE_NAME = os.path.expanduser("~/Downloads/rgc_natstim_72x64_joint_normalized_2024-10-11.pkl")
assert os.path.exists(STIMULUS_FILE_NAME)

# %%
# load the stimulus file
with open(STIMULUS_FILE_NAME, "rb") as f:
    stimuli_dict: dict[str, Any] = pickle.load(f)

print(f"{stimuli_dict.keys()=}")

# %% [markdown]
# It contains both the training movie and the test movie. 
# Both movies consist of two channels (green and UV), each frame is shown at 30HZ, and the spatial dimensions are 72x64.
# See [Qiu et al. 2021](https://www.sciencedirect.com/science/article/pii/S096098222100676X) to learn how this video was recorded, and [Höfling et al. 2024](https://elifesciences.org/articles/86860)to see how this stimulus was used for neural network training.
#
# The shape of both stimuli are (input_channel, time, height, width):

# %%
FRAMES_PER_SECOND = 30
print(f"{stimuli_dict['train'].shape=} {stimuli_dict['test'].shape=}")
print(f"Video duration train: {stimuli_dict['train'].shape[1] / FRAMES_PER_SECOND}s, "
      f"test: {stimuli_dict['test'].shape[1] / FRAMES_PER_SECOND}")

# %% [markdown]
# The test clips contains 25 seconds.
# The training gets divided into 108 clips, each consisting of 16200/108 = 150 frames, such being recorded for 5 seconds.
# The experimenters showed these clips into different orders defined in the array random sequences.
# This array is stored both in this repository [here](https://github.com/eulerlab/eyewire2-functional-analysis/blob/main/data/stimuli/mc_train_sequences.npy) and in the stimuli_dict:

# %%
print(f'{stimuli_dict["random_sequences"].shape}\n{stimuli_dict["random_sequences"]}')

# %% [markdown]
# Which random sequence was shown in a recording is defined by the scan_sequence_index. 
# For the recordings uploaded on [huggingface](https://huggingface.co/datasets/open-retina/open-retina/blob/main/euler_lab/hoefling_2024/responses/rgc_natsim_subset_only_naturalspikes_2024-08-14.h5) (which is a different recording than used here) this index is stored for each recording session as an attribute called `scan_sequence_index`.
#
# In the database you can figure out the scan_sequence_index by looking at the filename of the preprocessed h5 file, e.g. for the filepath`/gpfs01/euler/data/Data/Szatko/20200226/1/Pre/SMP_M1_LR_GCL5_MC14.h5` MC14 stands for mouse cam 14, and indicates that the `scan_sequence_index used was 14. You can then reconstruct the first training clip shown as follows:

# %%
# one trigger per clip, so 108 train trigger + 3 * 5 test triggers = 123 triggers

# %%
sequence_id = 14
random_sequence = stimuli_dict["random_sequences"][:, sequence_id]
clip_length_frames = 150

first_clip_id_shown = int(random_sequence[0])
first_clip_start_idx = first_clip_id_shown * clip_length_frames
first_clip_shown = stimuli_dict["train"][:, first_clip_start_idx:first_clip_start_idx+clip_length_frames]
print(f"{first_clip_shown.shape=}")

# %% [markdown]
# During the recording the test movie was shown three times in total in the following sequence as visualized in [this figure](https://iiif.elifesciences.org/lax:86860%2Felife-86860-fig1-v1.tif/full/1500,/0/default.jpg):
# - Test movie
# - First half of the training clips (54 clips)
# - Test movie
# - Second half of the training clips (54 clips)
# - Test movie
#
# During the recording every five seconds a trigger appeared, which are five triggers per test movie presentation and one triger for each shown training clip for a total of `5 + 54 + 5 + 54 + 5 = 123` triggers.

# %% [markdown]
# You can reconstruct the full movie for the scan_sequence_id 14 as follows using the function create_displayed_movie_sequence:

# %%
from eyewire2_functional_analysis.stimulus.stimulus_tools import create_displayed_movie_sequence

full_movie = create_displayed_movie_sequence(stimuli_dict["train"], stimuli_dict["test"], 
                                             stimuli_dict["random_sequences"], scan_sequence_id=14)
print(f"{full_movie.shape=}")

# %% [markdown]
# Or you use the code provided in the src directory:

# %% [markdown]
# ## Visulize movie as a video

# %% [markdown]
# If you are interested to see this movie as a video you can install [openretina](https://github.com/open-retina/open-retina) which provides a useful visualization tool. The UV component is mapped to violet there and the green component to green. Therefore, first install openretina and then play the video in the notebook:

# %%
# !pip install openretina

# %%
from openretina.utils.plotting import numpy_to_mp4_video

# %%
numpy_to_mp4_video(full_movie, fps=30, display_video=True)

# %%
