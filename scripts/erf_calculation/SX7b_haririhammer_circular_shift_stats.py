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

n_timepoints = 1201
n_rois = 185
n_task = len(tasknames)
all_data = np.zeros((nsubj, n_task, n_rois, n_timepoints))

for i, i_subj in enumerate(subjs_common):
    subj_preprocpath = pathjoin(preprocpath, i_subj)

    for j, j_taskname in enumerate(tasknames):
        stc = pklload(
            pathjoin(subj_preprocpath, "haririhammer", f"{j_taskname}_lcmv_stc.pkl")
        )
        times = stc["stc"].times
        labels = np.array(stc["labels"])

        # Create mapping from subject's labels to reference labels
        # Initialize with NaN for missing labels
        all_data[i, j, :, :] = np.nan
        for k, label in enumerate(labels):
            if label in example_labels:
                ref_idx = np.where(example_labels == label)[0][0]
                all_data[i, j, ref_idx, :] = stc["parcellated_ts"][k, :]

    print(f"{i+1} / {nsubj}")

######################################### Circular shift stats #########################################
poststim_times = np.where((times > 0) & (times < 0.6))[0]
prestim_times = np.where((times < 0))[0]

n_poststim_times = len(poststim_times)

poststim_data = np.sqrt(all_data[:, :, :, poststim_times] ** 2)
prestim_data = np.nanmean(np.sqrt(all_data[:, :, :, prestim_times] ** 2), axis=-1, keepdims=True)

real_differences = poststim_data - prestim_data

n_surrogates = 1000
surrogate_data = np.zeros(
    (nsubj, n_task, n_rois, n_poststim_times, n_surrogates)
)  # (nsubj, n_task, n_rois, n_poststim_times, n_surrogates)

for i in range(nsubj):
    for j in range(n_task):
        for k in range(n_rois):
            for s in range(n_surrogates):
                random_subj_idx = np.random.randint(0, nsubj)
                random_task_idx = np.random.randint(0, n_task)
                random_roi_idx = np.random.randint(0, n_rois)
                random_data = np.roll(
                    all_data[random_subj_idx, random_task_idx, random_roi_idx, :],
                    np.random.randint(0, n_timepoints),
                )
                surrogate_data[i, j, k, :, s] = poststim_data[i, j, k, :] - np.nanmean(
                    np.sqrt(random_data[prestim_times]**2)
                )
            print(f"{k+1} / {n_rois}")
        print(f"{j+1} / {n_task}")
    print(f"{i+1} / {nsubj}")

# np.savez(
#     pathjoin(output_dir, "circular_shift_stats.npz"),
#     real_differences=real_differences,
#     surrogate_data=surrogate_data,
# ) # too big to save!!!

# Identify significant ROIs

p_values = np.mean(surrogate_data < real_differences[:, :, :, :, np.newaxis], axis=-1) # (nsubj, n_task, n_rois, n_poststim_times)

# Identify significant ROIs with 60 consecutive time points
mask = p_values < 0.05  # Boolean mask for significant p-values
kernel = np.ones(60)    # Convolution kernel to count consecutive occurrences

# Use convolution to find consecutive runs of significant time points
consecutive_counts = np.apply_along_axis(
    lambda x: np.convolve(x.astype(int), kernel, mode='valid'), 
    axis=-1, 
    arr=mask
)

# Check which (subject, task, roi) combinations have 60 consecutive significant time points
has_60_consecutive = np.any(consecutive_counts == 60, axis=-1)  # Shape: (nsubj, n_task, n_rois)

# Optional: Get indices where this occurs
significant_indices = np.where(has_60_consecutive)
print(f"Found {len(significant_indices[0])} combinations with 60+ consecutive significant time points")

pklsave(pathjoin(output_dir, "circular_shift_stats.pkl"), {
    "p_values": p_values,
    "consecutive_counts": consecutive_counts,
    "has_60_consecutive": has_60_consecutive,
    "significant_indices": significant_indices,
})