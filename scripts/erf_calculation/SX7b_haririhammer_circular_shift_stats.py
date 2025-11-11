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

pklsave(pathjoin(output_dir, "all_data.pkl"), {"all_data": all_data, "times": times, "labels": example_labels})

######################################### Circular shift stats #########################################

#################################################################
all_data = pklload(pathjoin(output_dir, "all_data.pkl"))
times = all_data["times"]
labels = all_data["labels"]
all_data = all_data["all_data"]
#################################################################
ctx_indices = np.array(["ctx" in i for i in labels])
ctx_all_data = all_data[:, :, ctx_indices, :]
n_ctx_rois = ctx_all_data.shape[2]
ctx_labels = labels[ctx_indices]

poststim_times = np.where((times > 0) & (times < 0.6))[0]
prestim_times = np.where((times < 0))[0]

n_poststim_times = len(poststim_times)

poststim_data = np.sqrt(ctx_all_data[:, :, :, poststim_times] ** 2)
prestim_data = np.nanmean(np.sqrt(ctx_all_data[:, :, :, prestim_times] ** 2), axis=-1, keepdims=True)

# poststim_data_averaged = np.nanmean(poststim_data, axis=0)
# prestim_data_averaged = np.nanmean(prestim_data, axis=0)
# real_differences = poststim_data_averaged - prestim_data_averaged # (n_task, n_rois, ntime)
real_differences = poststim_data - prestim_data # (nsubj, n_task, n_rois, ntime)

n_surrogates = 1000
surrogate_data = np.zeros(
    (nsubj, n_task, n_ctx_rois, n_poststim_times, n_surrogates)
)
# surrogate_data = np.zeros(
#     (n_task, n_ctx_rois, n_poststim_times, n_surrogates)
# )

for j in range(n_task):
    for k in range(n_ctx_rois):
        for s in range(n_surrogates):
            random_subj_idx = np.random.randint(0, nsubj)
            random_task_idx = np.random.randint(0, n_task)
            random_roi_idx = np.random.randint(0, n_ctx_rois)
            random_data = np.roll(
                ctx_all_data[random_subj_idx, random_task_idx, random_roi_idx, :],
                np.random.randint(0, n_timepoints),
            )
            surrogate_data[:, j, k, :, s] = poststim_data[:, j, k, :] - np.nanmean(
                np.sqrt(random_data[prestim_times]**2)
            )
            # surrogate_data[j, k, :, s] = poststim_data_averaged[j, k, :] - np.nanmean(
            #     np.sqrt(random_data[prestim_times]**2)
            # )
        print(f"{k+1} / {n_ctx_rois}")
    print(f"{j+1} / {n_task}")

# np.savez(
#     pathjoin(output_dir, "circular_shift_stats.npz"),
#     real_differences=real_differences,
#     surrogate_data=surrogate_data,
# ) # too big to save!!!

# Identify significant ROIs

p_values = np.mean(surrogate_data < real_differences[:, :, :, :, np.newaxis], axis=-1) # (n_task, n_ctx_rois, n_poststim_times)

# Identify significant ROIs with 60 consecutive time points
mask = p_values > 0.95  # Boolean mask for significant p-values
cons_time_points = 60
kernel = np.ones(cons_time_points)    # Convolution kernel to count consecutive occurrences

# Use convolution to find consecutive runs of significant time points
consecutive_counts = np.apply_along_axis(
    lambda x: np.convolve(x.astype(int), kernel, mode='valid'), 
    axis=-1, 
    arr=mask
)

# Check which (subject, task, roi) combinations have 60 consecutive significant time points
has_consecutive = np.any(consecutive_counts == cons_time_points, axis=-1)  # Shape: (n_task, n_ctx_rois)
sig_subj_count = np.sum(has_consecutive, axis=0)
sig_subj_indices = sig_subj_count > 20

# Optional: Get indices where this occurs
# significant_indices = np.where(has_consecutive)
significant_indices = np.where(has_consecutive & sig_subj_indices[np.newaxis, :, :])
print(f"Found {len(significant_indices[0])} combinations with {cons_time_points}+ consecutive significant time points")


pklsave(pathjoin(output_dir, "circular_shift_stats.pkl"), {
    "p_values": p_values,
    "consecutive_counts": consecutive_counts,
    "has_consecutive": has_consecutive,
    "significant_indices": significant_indices,
})