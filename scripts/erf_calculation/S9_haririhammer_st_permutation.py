# pid: 26507
##### Spatiotemporal permutation testing
import numpy as np
import mne
import os
from os.path import join as pathjoin
import matplotlib.pyplot as plt
from erf_acw2.meg_chlist import chlist
from erf_acw2.src import subjs, pklsave, pklload, badchan_padnan_2d, auto_threshold_F, get_commonsubj
import scipy as sp
from mne.viz import plot_compare_evokeds
from mpl_toolkits.axes_grid1 import make_axes_locatable

taskname = "haririhammer"

subjs_common = get_commonsubj(taskname)
nsubj = len(subjs_common)

# MNE ANOVA f'n requires the first factor to be slow and second to be fast
# first factor: emotion (happy, sad, shape); second factor: encode vs probe
# So encode: happy, sad, shape; probe: happy, sad, shape
tasknames = ["encode_face_happy", "encode_face_sad", "encode_shape",
             "probe_face_happy", "probe_face_sad", "probe_shape"]

tasks = {i: np.zeros((nsubj, 1201, 272)) for i in tasknames}

for i, i_subj in enumerate(subjs_common):
    filename = pathjoin(
        f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}",
        f"{i_subj}_{taskname}_erf_emo.pkl",
    )
    epochsdict = pklload(filename)
    for j in tasknames:
        i_data = epochsdict[j].get_data()
        i_ch = epochsdict[j].info["ch_names"]
        tasks[j][i, :, :] = badchan_padnan_2d(i_ch, i_data).T
    
X = [tasks[i] for i in tasknames]
factor_levels = [2, 3]
adj = pklload("/BICNAS2/ycatal/erf_acw2/erf_acw2/adjacency.pkl")
return_pvals = False
pthresh = 0.001
n_replications = nsubj
all_effects = ["A:B", "A", "B"]
effect_names = ["factor_interaction", "factor_encprob", "factor_emo"]

for i, effects in enumerate(all_effects):
    f_thresh = mne.stats.f_threshold_mway_rm(n_replications, factor_levels, effects, pthresh)

    def stat_fun(*args):
        # get f-values only.
        return mne.stats.f_mway_rm(
            np.swapaxes(args, 1, 0),
            factor_levels=factor_levels,
            effects=effects,
            return_pvals=return_pvals,
        )[0]

    stats, clusters, cluster_p, h0 = mne.stats.spatio_temporal_cluster_test(
        X,
        adjacency=adj,
        stat_fun=stat_fun,
        threshold=f_thresh,
        n_permutations=5000,
        n_jobs=8
    )

    savename = f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}/{taskname}_erp_permutationtest_st_emo_{effect_names[i]}.pkl"
    pklsave(
        savename,
        {"stats": stats, "clusters": clusters, "cluster_p": cluster_p, "h0": h0},
    )
