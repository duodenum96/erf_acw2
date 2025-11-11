# cd /BICNAS2/ycatal/erf_acw2/scripts/erf_calculation
# nohup python SZ3_induced_permutation.py > log/SZ3_induced_permutation_p_0d0001_baseline.log &
# echo $! > log/SZ3_induced_permutation_p_0d0001_baseline.pid
##### Spatiotemporal permutation testing

import numpy as np
import mne
import os
from os.path import join as pathjoin
import matplotlib.pyplot as plt
from erf_acw2.meg_chlist import chlist
from erf_acw2.src import (
    subjs,
    pklsave,
    pklload,
    badchan_padnan_2d,
    auto_threshold_F,
    get_commonsubj,
    badchan_padnan_tfr,
    badchan_padnan
)
import scipy as sp
from mne.viz import plot_compare_evokeds
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fooof.bands import Bands
import fooof

taskname = "haririhammer"

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

# MNE ANOVA f'n requires the first factor to be slow and second to be fast
# first factor: emotion (happy, sad, shape); second factor: encode vs probe
# So encode: happy, sad, shape; probe: happy, sad, shape
tasknames = [
    "encode_face_happy",
    "encode_face_sad",
    "encode_shape",
    "probe_face_happy",
    "probe_face_sad",
    "probe_shape",
]

nchan = 272
# load fooof results
freqbands = ["delta", "theta", "alpha", "beta", "gamma"]
freqranges = [[1, 4], [4, 8], [8, 12], [12, 30], [30, 50]]
nfreq = len(freqbands)
bands = Bands(
    {freqbands[i]: [freqranges[i][0], freqranges[i][1]] for i in range(nfreq)}
)

fooof_results = {k: {
    i: np.zeros((nsubj, nchan, 3)) for i in freqbands
} for k in tasknames}

induced_powers = {k: [] for k in tasknames}
induced_r2 = {k: np.zeros((nsubj, nchan, 2)) for k in tasknames}

for i, i_subj in enumerate(subjs_common):
    filename = pathjoin(
        "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
        f"{i_subj}_{taskname}_erf_emo_induced_fooof.pkl",
    )
    data_dict = pklload(filename)
    induced_power = data_dict["induced_power"]
    i_chlist = induced_power["encode_face_happy"].info["ch_names"]
    i_n_ch = len(i_chlist)
    low_fgs = data_dict["low_fgs"]
    high_fgs = data_dict["high_fgs"]
    for k in tasknames:
        induced_r2[k][i, :, 0] = badchan_padnan(
            i_chlist,
            np.array([low_fgs[k].group_results[i].r_squared for i in range(i_n_ch)])
        )
        induced_r2[k][i, :, 1] = badchan_padnan(
            i_chlist,
            np.array([high_fgs[k].group_results[i].r_squared for i in range(i_n_ch)])
        )

    for j, j_band in enumerate(freqbands):
        if j_band in ["delta", "theta", "alpha"]:
            j_fg = low_fgs
        else:
            j_fg = high_fgs
        for k in tasknames:
            fooof_results[k][j_band][i, :, :] = badchan_padnan_2d(
                i_chlist,
                fooof.analysis.get_band_peak_fg(j_fg[k], bands[j_band])
            ) 
    for k in tasknames:
        induced_powers[k].append(induced_power[k])

savename = f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}/induced_r2_fooof.pkl"
pklsave(savename, induced_r2)
############################### 
# Plot grand averages for each task
for k in tasknames:
    template_bad = induced_powers[k][17]
    bad_channels = template_bad.info["ch_names"]
    ga_spectrum = mne.time_frequency.combine_spectrum(
        [induced_powers[k][i].pick(bad_channels) for i in range(nsubj)], weights="equal"
    )
    ga_spectrum.plot(xscale="log")
    plt.savefig(f"ga_spectrum_{k}.jpg")
############################### 
# Report FOOOF summary

import pandas as pd

# Calculate percent fits as a DataFrame
prc_fits_df = pd.DataFrame(
    {
        k: {
            j_band: 1 - np.mean(np.isnan(fooof_results[k][j_band][1, :, :]))
            for j_band in freqbands
        }
        for k in tasknames
    }
).T  # tasks as rows, bands as columns
prc_fits_df
##############################
# Work only with theta, beta and gamma fits
selected_freqbands = ["theta", "beta", "gamma"]


Xs = {i: [fooof_results[j][i][:, :, 1] for j in tasknames] for i in selected_freqbands}
X_names = selected_freqbands
all_data = {i: {j: fooof_results[i][j][:, :, 1] for j in selected_freqbands} for i in tasknames}

factor_levels = [2, 3]
adj = pklload("/BICNAS2/ycatal/erf_acw2/erf_acw2/adjacency.pkl")
return_pvals = False
pthresh = 0.05
n_replications = nsubj
all_effects = ["A", "B"]
effect_names = ["factor_encprob", "factor_emo"]


def stat_fun(*args):
    # get f-values only.
    return mne.stats.f_mway_rm(
        np.swapaxes(args, 1, 0),
        factor_levels=factor_levels,
        effects=effects,
        return_pvals=return_pvals,
    )[0]


for i, effects in enumerate(all_effects):
    f_thresh = mne.stats.f_threshold_mway_rm(
        n_replications, factor_levels, effects, pthresh
    )
    for j, j_freq in enumerate(selected_freqbands):
        X_name = X_names[j]
        stats, clusters, cluster_p, h0 = mne.stats.permutation_cluster_test(
            Xs[j_freq],
            adjacency=adj,
            stat_fun=stat_fun,
            threshold=f_thresh,
            n_permutations=5000,
            n_jobs=16,
        )

        savename = (
            f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}/"
            f"{taskname}_erp_permutationtest_st_emo_{effect_names[i]}_induced_{X_name}_fooof.pkl"
        )
        pklsave(
            savename,
            {"stats": stats, "clusters": clusters, "cluster_p": cluster_p, "h0": h0},
        )
        print(f"Finished {X_name} {effect_names[i]}")

print("Done")
# Design can't be fully balanced. 
filename = (
    f"/BICNAS2/ycatal/erf_acw2/results/erf/" + 
    f"all_data_induced_fooof.pkl"
)
pklsave(filename, {"all_data": all_data, "selected_freqbands": selected_freqbands, "tasknames": tasknames})