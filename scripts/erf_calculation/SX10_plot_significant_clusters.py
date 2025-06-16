import mne
import numpy as np
import os
from os.path import join as pathjoin
from erf_acw2.src import pklload
import matplotlib.pyplot as plt
import matplotlib

results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source/permutation_test"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def get_cluster_time_series(cluster_data, i_vertices):
    """
    Average over vertices
    """
    i_data = np.mean(cluster_data.data[np.unique(i_vertices), :], axis=0)
    return i_data

def get_cluster_time_series_ci(cluster_data, i_vertices):
    i_data = np.mean(cluster_data[:, :, np.unique(i_vertices)], axis=2)
    return i_data


def plot_source_time_series(
    time_series_data,
    std_data,
    times,
    colors,
    labels,
    ax=None,
    title="",
    show_legend=True,
    ci=True,
    sig_times=None,
):
    """
    Plot time series data similar to plot_compare_evokeds but for source space data

    Parameters:
    -----------
    time_series_data : list of arrays
        List of time series data for each condition
    std_data : list of arrays
        List of standard error/deviation data for each condition
    times : array
        Time points
    colors : list
        Colors for each condition
    labels : list
        Labels for each condition
    ax : matplotlib axis
        Axis to plot on
    title : str
        Plot title
    show_legend : bool
        Whether to show legend
    ci : bool
        Whether to show confidence intervals
    sig_times : array or None
        Array of significant time points to highlight
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for i, (ts_data, std_data_i, color, label) in enumerate(
        zip(time_series_data, std_data, colors, labels)
    ):
        # Plot main time series
        ax.plot(times, ts_data, color=color, label=label, linewidth=2)

        # Plot confidence interval if requested
        if ci and std_data_i is not None:
            ax.fill_between(
                times,
                std_data_i[0],
                std_data_i[1],
                color=color,
                alpha=0.2,
            )

    # Add significant time window if provided
    if sig_times is not None and len(sig_times) > 0:
        ymin, ymax = ax.get_ylim()
        ax.fill_betweenx(
            [ymin, ymax],
            sig_times[0],
            sig_times[-1],
            color='orange',
            alpha=0.3,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Source Activity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(loc="best")

    return ax


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
data = pklload(fname)
grand_averages = data["grand_averages"]
grand_averages_ci = data["grand_averages_ci"] # each key: 2 x 1201 x n_vertices

times = grand_averages["encode_face_happy"].times

p_thresh = 0.001

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

i_factor = factors[0]
significant_clusters = np.where(cluster_results[i_factor]["cluster_p"] < p_thresh)[0]

cluster_idx = 0
i_cluster = significant_clusters[cluster_idx]

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
        [
            get_cluster_time_series(grand_averages["encode_face_happy"], i_vertices),
            get_cluster_time_series(grand_averages["encode_face_sad"], i_vertices),
            get_cluster_time_series(grand_averages["encode_shape"], i_vertices),
        ],
        [
            get_cluster_time_series(grand_averages["probe_face_happy"], i_vertices),
            get_cluster_time_series(grand_averages["probe_face_sad"], i_vertices),
            get_cluster_time_series(grand_averages["probe_shape"], i_vertices),
        ],
    ]
    comparisons_data_ci_ts = [
        [
            get_cluster_time_series_ci(
                grand_averages_ci["encode_face_happy"], i_vertices
            ),
            get_cluster_time_series_ci(grand_averages_ci["encode_face_sad"], i_vertices),
            get_cluster_time_series_ci(grand_averages_ci["encode_shape"], i_vertices),
        ],
        [
            get_cluster_time_series(grand_averages_ci["probe_face_happy"], i_vertices),
            get_cluster_time_series(grand_averages_ci["probe_face_sad"], i_vertices),
            get_cluster_time_series(grand_averages_ci["probe_shape"], i_vertices),
        ],
    ]
    i_color = [colors_list[0], colors_list[1]]
elif i_factor == "factor_encprob":
    comparisons = [
        ["encode_face_happy", "probe_face_happy"],
        ["encode_face_sad", "probe_face_sad"],
        ["encode_shape", "probe_shape"],
    ]
    comparisons_data_ts = [
        [
            get_cluster_time_series(grand_averages["encode_face_happy"], i_vertices),
            get_cluster_time_series(grand_averages["probe_face_happy"], i_vertices),
        ],
        [
            get_cluster_time_series(grand_averages["encode_face_sad"], i_vertices),
            get_cluster_time_series(grand_averages["probe_face_sad"], i_vertices),
        ],
        [
            get_cluster_time_series(grand_averages["encode_shape"], i_vertices),
            get_cluster_time_series(grand_averages["probe_shape"], i_vertices),
        ],
    ]

    comparisons_data_ci_ts = [
        [
            get_cluster_time_series_ci(
                grand_averages_ci["encode_face_happy"], i_vertices
            ),
            get_cluster_time_series_ci(grand_averages_ci["probe_face_happy"], i_vertices),
        ],
        [
            get_cluster_time_series_ci(grand_averages_ci["encode_face_sad"], i_vertices),
            get_cluster_time_series_ci(grand_averages_ci["probe_face_sad"], i_vertices),
        ],
        [
            get_cluster_time_series_ci(grand_averages_ci["encode_shape"], i_vertices),
            get_cluster_time_series_ci(grand_averages_ci["probe_shape"], i_vertices),
        ],
    ]
    i_color = [colors_list[2], colors_list[3], colors_list[4]]

for i, i_comparison in enumerate(comparisons):
    
    f, ax = plt.subplots(1, 3, figsize=(19, 4))
    
    # Get significant times if they exist
    sig_times = times[np.unique(i_times)] if len(i_times) > 0 else None
    
    plot_source_time_series(
        comparisons_data_ts[i],
        comparisons_data_ci_ts[i],
        times,
        i_color[i],
        i_comparison,
        ax=ax[1],
        sig_times=sig_times  # Pass the significant times
    )

    plt.savefig(os.path.join(results_dir, f"{taskname}_cluster_{cluster_idx}_{i_factor}_{i_comparison[0]}X{i_comparison[1]}.jpg"))
