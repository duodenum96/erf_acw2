import mne
import numpy as np
import os
from os.path import join as pathjoin
import sys
from erf_acw2.src import get_commonsubj

os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from scipy import signal
import matplotlib.pyplot as plt
from erf_acw2.src import pklload, pklsave, badchan_padnan_2d

taskname = "haririhammer"
subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

# adj = mne.spatial_src_adjacency(src)

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"
output_dir = pathjoin(source_results_dir, taskname)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tasknames = [
    "encode_face_happy",
    "probe_face_happy",
    "encode_face_sad",
    "probe_face_sad",
    "probe_shape",
    "encode_shape",
]

example_labels = np.array(
    pklload(
        pathjoin(
            preprocpath, subjs_common[0], "haririhammer", f"{tasknames[0]}_lcmv_stc.pkl"
        )
    )["labels"]
)

data = pklload(pathjoin(output_dir, "all_data.pkl"))

all_data = data["all_data"]
times = data["times"]

f, ax = plt.subplots(1, len(tasknames), figsize=(30, 5))
for i, i_task in enumerate(tasknames):
    ax[i].plot(times, np.nanmean(all_data[:, i, :, :], axis=0).T, alpha=0.5)
    ax[i].set_title(f"{i_task}")

f.savefig(pathjoin(output_dir, "grand_average.png"))


