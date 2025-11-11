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
)
import scipy as sp
from mne.viz import plot_compare_evokeds
from mpl_toolkits.axes_grid1 import make_axes_locatable

taskname = "haririhammer"

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

# MNE ANOVA f'n requires the first factor to be slow and second to be fast
# first factor: emotion (happy, sad, shape); second factor: encode vs probe
# So encode: happy, sad, shape; probe: happy, sad, shape
tasknames = ["encode_face_happy", "encode_face_sad", "encode_shape",
             "probe_face_happy", "probe_face_sad", "probe_shape"]

nfreq = 20
ntime = 151

tasks_power = {i: np.zeros((nsubj, nfreq, ntime, 272)) for i in tasknames}
tasks_itpc = {i: np.zeros((nsubj, nfreq, ntime, 272)) for i in tasknames}

for i, i_subj in enumerate(subjs_common):
    filename = pathjoin(
        f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}",
        f"{i_subj}_{taskname}_erf_emo_induced_power.pkl",
    )
    data_dict = pklload(filename)
    for j in tasknames:
        i_data_power = data_dict["induced_power"][j].get_data()
        i_data_itpc = data_dict["induced_itpc"][j].get_data()
        
        i_ch = data_dict["induced_power"][j].info["ch_names"]

        tasks_power[j][i, :, :, :] = badchan_padnan_tfr(i_ch, i_data_power).transpose(1, 2, 0)
        tasks_itpc[j][i, :, :, :] = badchan_padnan_tfr(i_ch, i_data_itpc).transpose(1, 2, 0)

X_power = [tasks_power[i] for i in tasknames]
X_itpc = [tasks_itpc[i] for i in tasknames]
factor_levels = [2, 3]
adj = pklload("/BICNAS2/ycatal/erf_acw2/erf_acw2/adjacency.pkl")
return_pvals = False
pthresh = 0.0001
n_replications = nsubj
all_effects = ["A", "B"]
effect_names = ["factor_encprob", "factor_emo"]

adj_combined = mne.stats.combine_adjacency(adj, nfreq, ntime)

all_X = [X_power, X_itpc]
X_names = ["power", "itpc"]

def stat_fun(*args):
# get f-values only.
    return mne.stats.f_mway_rm(
        np.swapaxes(args, 1, 0),
        factor_levels=factor_levels,
        effects=effects,
        return_pvals=return_pvals,
    )[0]


for i, effects in enumerate(all_effects):
    f_thresh = mne.stats.f_threshold_mway_rm(n_replications, factor_levels, effects, pthresh)
    for j, X in enumerate(all_X):
        X_name = X_names[j]
        stats, clusters, cluster_p, h0 = mne.stats.spatio_temporal_cluster_test(
            X,
            adjacency=adj_combined,
            stat_fun=stat_fun,
            threshold=f_thresh,
            n_permutations=5000,
            n_jobs=16
        )

        savename = f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}/" \
                   f"{taskname}_erp_permutationtest_st_emo_{effect_names[i]}_induced_{X_name}_p_0d001_baseline.pkl"
        pklsave(
            savename,
            {"stats": stats, "clusters": clusters, "cluster_p": cluster_p, "h0": h0},
        )
        print(f"Finished {X_name} {effect_names[i]}")

print("Done")