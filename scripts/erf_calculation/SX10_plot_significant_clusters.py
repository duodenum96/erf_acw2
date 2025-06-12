import mne
import numpy as np
import os
from os.path import join as pathjoin
from erf_acw2.src import pklload
import matplotlib.pyplot as plt
import matplotlib

def get_cluster_time_series(cluster_data, i_vertices):
    """
    Average over vertices
    """
    i_data = np.mean(cluster_data.data[np.unique(i_vertices), :], axis=0)
    return i_data



# Colors:
# encode face happy: crimson, encode face sad: steelblue, encode shape: darkorchid
# probe face happy: maroon, probe face sad: turquoise, probe shape: forestgreen
colors_list = [
    ["crimson", "steelblue", "darkorchid"],
    ["maroon", "turquoise", "forestgreen"],
    ["crimson", "maroon"],
    ["steelblue", "turquoise"],
    ["darkorchid", "forestgreen"],
]

# Text and line coordinates for statistics
stat_text_coords = [0.5, 1.5, 1.0]
stat_text_ycoords = [115, 115, 125]
stat_line_coords = [[0, 0, 1, 1], [1, 1, 2, 2], [0, 0, 2, 2]]
stat_line_ycoords = [[110, 115, 115, 110], [110, 115, 115, 110], [115, 125, 125, 115]]

tasknames = [
    "encode_face_happy",
    "probe_face_happy",
    "encode_face_sad",
    "probe_face_sad",
    "encode_shape",
    "probe_shape",
]

comparisons = [
    ["encode_face_happy", "encode_face_sad", "encode_shape"],
    ["probe_face_happy", "probe_face_sad", "probe_shape"],
    ["encode_face_happy", "probe_face_happy"],
    ["encode_face_sad", "probe_face_sad"],
    ["encode_shape", "probe_shape"],
]

comparisons_nicer = [
    ["encode face happy", "encode face sad", "encode shape"],
    ["probe face happy", "probe face sad", "probe shape"],
    ["encode face happy", "probe face happy"],
    ["encode face sad", "probe face sad"],
    ["encode shape", "probe shape"],
]
comparisons_short = [
    ["efh", "efs", "es"],
    ["pfh", "pfs", "ps"],
    ["efh", "pfh"],
    ["efs", "pfs"],
    ["es", "ps"],
]

comparisons_nicer2 = [
    "Happy - Sad - Shape",
    "Happy - Sad - Shape",
    "Encode - Probe",
    "Encode - Probe",
    "Encode - Probe",
]
suptitles = ["Encode", "Probe", "Happy", "Sad", "Shape"]

matplotlib.rcParams.update({"font.size": 16})

effect_names = ["factor_encprob", "factor_emo"]
# Plan:
# 1) Happy vs Sad vs Shape (for encode and probe)
# 2) Encode vs Probe (for happy, sad and shape)

tests = [
    "factor_emo",
    "factor_emo",
    "factor_encprob",
    "factor_encprob",
    "factor_encprob",
]

# Xs = [
#     [tasks["encode_face_happy"], tasks["encode_face_sad"], tasks["encode_shape"]], 
#     [tasks["probe_face_happy"], tasks["probe_face_sad"], tasks["probe_shape"]],
#     [tasks["encode_face_happy"], tasks["probe_face_happy"]],
#     [tasks["encode_face_sad"], tasks["probe_face_sad"]],
#     [tasks["encode_shape"], tasks["probe_shape"]]
# ]

taskname = "haririhammer"
source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"
output_dir = pathjoin(source_results_dir, taskname)


# Load grand averages
fname = pathjoin(output_dir, "grand_average.pkl")
grand_averages = pklload(fname)["grand_averages"]
grand_averages_std = pklload(fname)["grand_averages_std"]

# Load significant clusters
fname = pathjoin(output_dir, f"{taskname}_erp_permutationtest_st_all_cluster_vis.pkl")
cluster_data = pklload(fname)
cluster_results = cluster_data["cluster_results"]
clus = cluster_data["clus"]

# Load source space for visualization
fsaverage_bem_path = "/BICNAS2/group-northoff/NIMH_source_reconstruction/fsaverage_bem"
fname_fsaverage_src = pathjoin(fsaverage_bem_path, "fsaverage-ico-4-src.fif")
src = mne.read_source_spaces(fname_fsaverage_src)
fsave_vertices = [s["vertno"] for s in src]

factors = list(cluster_results.keys())

# Example code to plot single cluster

f, ax = plt.subplots(1, 3, figsize=(19, 4))

i_factor = factors[0]
significant_clusters = np.where(cluster_results[i_factor]["cluster_p"] < 0.01)[0]

i_cluster = significant_clusters[0]

i_cluster = cluster_results[i_factor]["clusters"][i_cluster]
i_times = i_cluster[0]
i_vertices = i_cluster[1]

if i_factor == "factor_emo":
    # Compare the the elements in each list
    comparisons = [
        ["encode_face_happy", "encode_face_sad", "encode_shape"],
        ["probe_face_happy", "probe_face_sad", "probe_shape"],
    ]
    comparisons_data_ts = [
        [get_cluster_time_series(grand_averages["encode_face_happy"], i_vertices), get_cluster_time_series(grand_averages["encode_face_sad"], i_vertices), get_cluster_time_series(grand_averages["encode_shape"], i_vertices)],
        [get_cluster_time_series(grand_averages["probe_face_happy"], i_vertices), get_cluster_time_series(grand_averages["probe_face_sad"], i_vertices), get_cluster_time_series(grand_averages["probe_shape"], i_vertices)],
    ]
    comparisons_data_std_ts = [
        [get_cluster_time_series(grand_averages_std["encode_face_happy"], i_vertices), get_cluster_time_series(grand_averages_std["encode_face_sad"], i_vertices), get_cluster_time_series(grand_averages_std["encode_shape"], i_vertices)],
        [get_cluster_time_series(grand_averages_std["probe_face_happy"], i_vertices), get_cluster_time_series(grand_averages_std["probe_face_sad"], i_vertices), get_cluster_time_series(grand_averages_std["probe_shape"], i_vertices)],
    ]
    i_color = [colors_list[0], colors_list[1]]
elif i_factor == "factor_encprob":
    comparisons = [
        ["encode_face_happy", "probe_face_happy"],
        ["encode_face_sad", "probe_face_sad"],
        ["encode_shape", "probe_shape"],
    ]
    comparisons_data_ts = [
        [get_cluster_time_series(grand_averages["encode_face_happy"], i_vertices), get_cluster_time_series(grand_averages["probe_face_happy"], i_vertices)],
        [get_cluster_time_series(grand_averages["encode_face_sad"], i_vertices), get_cluster_time_series(grand_averages["probe_face_sad"], i_vertices)],
        [get_cluster_time_series(grand_averages["encode_shape"], i_vertices), get_cluster_time_series(grand_averages["probe_shape"], i_vertices)],
    ]
    comparisons_data_std_ts = [
        [get_cluster_time_series(grand_averages_std["encode_face_happy"], i_vertices), get_cluster_time_series(grand_averages_std["probe_face_happy"], i_vertices)],
        [get_cluster_time_series(grand_averages_std["encode_face_sad"], i_vertices), get_cluster_time_series(grand_averages_std["probe_face_sad"], i_vertices)],
        [get_cluster_time_series(grand_averages_std["encode_shape"], i_vertices), get_cluster_time_series(grand_averages_std["probe_shape"], i_vertices)],
    ]
    i_color = [colors_list[2], colors_list[3], colors_list[4]]

for i, i_comparison in enumerate(comparisons):
    for j, j_trial in enumerate(i_comparison):
        ax[1].plot(comparisons_data_ts[i][j], color=i_color[i][j], label=j_trial)
        ax[1].plot(comparisons_data_std_ts[i][j], color=i_color[i][j], linestyle="--")

ax[1].legend()
ax[1].set_title("Cluster time series")
ax[1].set_xlabel("Time (ms)")
ax[1].set_ylabel("ERF (µV)")

plt.savefig("anan.jpg")
