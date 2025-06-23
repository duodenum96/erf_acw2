import os
from os.path import join as pathjoin
import numpy as np
from erf_acw2.src import pklsave, pklload

import mne
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from erf_acw2.meg_chlist import chlist
import pandas as pd
import pingouin as pg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir("/BICNAS2/ycatal/erf_acw/")
from erf_acw2.src import get_commonsubj
######################### Prelude  ##############################
taskname = "haririhammer"

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

######################## Import ACW ##############################

task = "haririhammer"

effect_names = ["factor_interaction", "factor_encprob", "factor_emo"]

hh_acw = pklload(f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_ints.pkl")["rest_ints"]
hh_acw = 1 / (2*np.pi * hh_acw)

restmean = hh_acw # backwards compatibility

###################################################################

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure5"

tasknames = ["encode_face_happy", "probe_face_happy", 
             "encode_face_sad", "probe_face_sad", 
             "encode_shape", "probe_shape"]

tasks = {i: [] for i in tasknames}

for i, i_subj in enumerate(subjs_common):
    filename = pathjoin(
        f"/BICNAS2/ycatal/meg_intdiff/results/erp/{taskname}",
        f"{i_subj}_{taskname}_erf_emo.pkl",
    )
    epochsdict = pklload(filename)
    for j in tasknames:
        tasks[j].append(epochsdict[j])


# Equalize channels, there is one bad channel in some subjects that is messing up things everything. Unfortunately we need to 
# take it out from all subjects.
template_bad = tasks["encode_face_happy"][17]
for i, i_subj in enumerate(subjs_common):
    for j in tasknames:
        tasks[j][i] = mne.channels.equalize_channels([template_bad, tasks[j][i]])[1]
# done


all_effects = ["A:B", "A", "B"]
effect_names = ["factor_interaction", "factor_encprob", "factor_emo"]
# Plan:
# 1) Happy vs Sad vs Shape (for encode and probe)
# 2) Encode vs Probe (for happy, sad and shape)
# 3) Interaction?????

tests = ["factor_emo", "factor_emo"] # , "factor_encprob", "factor_encprob", "factor_encprob"]


# Will use this template for loading stat results
# f"/BICNAS2/ycatal/meg_intdiff/scripts/erps/{taskname}_erp_permutationtest_st_emo_{effect_names[i]}.pkl"
# loadname = f"/BICNAS2/ycatal/meg_intdiff/scripts/erps/{taskname}_erp_permutationtest_st_emo.pkl"
# cluster_results = pklload(loadname)

# cluster_stats = cluster_results["stats"]
# clusters = cluster_results["clusters"]
# cluster_pvals = cluster_results["cluster_p"]

stat_text_coords = [0.5, 1.5, 1.0]
stat_text_ycoords = [115, 115, 125]
stat_line_coords = [[0,0,1,1], [1,1,2,2], [0,0,2,2]]
stat_line_ycoords = [[110,115,115,110], [110,115,115,110], [115,125,125,115]]


Xs = [
    [tasks["encode_face_happy"], tasks["encode_face_sad"], tasks["encode_shape"]], 
    [tasks["probe_face_happy"], tasks["probe_face_sad"], tasks["probe_shape"]]
    # [tasks["encode_face_happy"], tasks["probe_face_happy"]],
    # [tasks["encode_face_sad"], tasks["probe_face_sad"]],
    # [tasks["encode_shape"], tasks["probe_shape"]]
]

comparisons = [
    ["encode_face_happy", "encode_face_sad", "encode_shape"], 
    ["probe_face_happy", "probe_face_sad", "probe_shape"]
    # ["encode_face_happy", "probe_face_happy"],
    # ["encode_face_sad", "probe_face_sad"],
    # ["encode_shape", "probe_shape"]
]

comparisons_nicer = [
    ["encode face happy", "encode face sad", "encode shape"], 
    ["probe face happy", "probe face sad", "probe shape"]
    # ["encode face happy", "probe face happy"],
    # ["encode face sad", "probe face sad"],
    # ["encode shape", "probe shape"]
]
comparisons_nicer3 = [
    ["encode happy", "encode sad", "encode shape"], 
    ["probe happy", "probe sad", "probe shape"]
    # ["encode face happy", "probe face happy"],
    # ["encode face sad", "probe face sad"],
    # ["encode shape", "probe shape"]
]
comparisons_short = [
    ["efh", "efs", "es"], 
    ["pfh", "pfs", "ps"]
    # ["efh", "pfh"],
    # ["efs", "pfs"],
    # ["es", "ps"]
]

comparisons_nicer2 = ["Happy - Sad - Shape", "Happy - Sad - Shape"] #, "Encode - Probe", "Encode - Probe", "Encode - Probe"]
suptitles = ["Encode", "Probe"] # , "Happy", "Sad", "Shape"]

adj = pklload("/BICNAS2/ycatal/meg_intdiff/meg_intdiff/adjacency.pkl")
ttestfunc = lambda a, b: sp.stats.ttest_ind(a, b, nan_policy="omit")[0]
matplotlib.rcParams.update({'font.size': 16})

# Colors: 
# encode face happy: crimson, encode face sad: steelblue, encode shape: darkorchid
# probe face happy: maroon, probe face sad: turquoise, probe shape: forestgreen
colors_list = [
    ["crimson", "steelblue",  "darkorchid"],
    ["maroon", "turquoise", "forestgreen"]
    # ["crimson", "maroon"],
    # ["steelblue", "turquoise"],
    # ["darkorchid", "forestgreen"]
]

def p2str(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return " (n.s.)"

pass_or_fail = pklload("/BICNAS2/ycatal/erf_acw/figures/figs/figure2/pass_or_fail.pkl")["pass_or_fail"][0:12] # first 12 items are from first 2 comparisons

# We are looking into encode (happy vs sad vs shape) and probe (happy vs sad vs shape)
# There are 12 significant clusters on this shit
# We need 12 scatter plots but also somehow indicate topo and time series

nrow = 2
ncol =  3

rhos = np.zeros((nrow,ncol,3)) # row x col x happy/sad/shape
pvals = np.zeros((nrow,ncol,3))

row = 0
col = 0
pf_idx = 0 # pass or fail index
for i in range(len(Xs)):
    #######################################
    # plot
    #######################################
    loadname = f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}/{taskname}_erp_permutationtest_st_emo_{tests[i]}.pkl"
    cluster_results = pklload(loadname)
    
    p_accept = 0.001
    good_cluster_inds = np.where(cluster_results["cluster_p"] < p_accept)[0]

    # configure variables for visualization
    colors = {comparisons[i][j]: colors_list[i][j] for j in range(len(colors_list[i]))}
    
    # organize data for plotting
    # evokeds = {cond: mne.grand_average(tasks[cond]) for cond in comparisons[i]}
    evokeds = {cond: tasks[cond] for cond in comparisons[i]}

    # find the two largest clusters and only use them (maybe not????)
    space_lengths = []
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        _, i_space_inds = np.squeeze(cluster_results["clusters"][clu_idx])
        space_lengths.append(len(np.unique(i_space_inds)))
    large_idx = [np.where(np.array(space_lengths) == j)[0][0] for j in np.sort(space_lengths)[-1:-3:-1]]


    # loop over clusters
    # for i_clu, clu_idx in enumerate(good_cluster_inds[large_idx]): # Code for only largest clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        if not pass_or_fail[pf_idx]:
            pf_idx += 1
            continue
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(cluster_results["clusters"][clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        i_chlist = template_bad.info["ch_names"]
        
        badchan = np.where(chlist == np.setdiff1d(chlist, i_chlist))[0]

        # get signals at the sensors contributing to the cluster
        times = evokeds[comparisons[i][0]][0].times
        sig_times = template_bad.times[time_inds]

        # Find the matching picks
        goodchans = np.array(chlist)[ch_inds]
        picks = []
        for k in goodchans:
            idx = np.where(k == np.array(i_chlist))[0]
            if len(np.where(k == np.array(i_chlist))[0]) == 0:
                continue
            else:
                picks.append(idx[0])
        picks = np.array(picks)


        # Average the activity inside the shaded area for each subject
        erf_violins = {comparisons[i][k]: [] for k in range(len(comparisons[i]))}
        for k, k_comp in enumerate(comparisons[i]):
            nsubj = len(evokeds[k_comp])
            for l in range(nsubj):
                erf_violins[k_comp].append(np.sqrt(np.mean( # rms
                    evokeds[k_comp][l].get_data()[np.ix_(picks, time_inds)] ** 2, axis=1))
                      * 1e15) # 1e15: tesla to femtotesla

        for m, m_key in enumerate(erf_violins.keys()):
            corr_results = sp.stats.spearmanr(np.nanmean(np.array(erf_violins[m_key]), axis=0), np.nanmean(restmean[picks, :], axis=1))
            rhos[row, col, m] = corr_results.statistic
            pvals[row, col, m] = corr_results.pvalue
        
        pf_idx += 1
        col += 1
        if col == 3:
            col = 0
            row += 1

pvals_correct = pg.multicomp(pvals, method="fdr_bh")[1]



#### Now that we corrected p values, we can run the loop again to plot (fuck my life, I'm not paid enough to do this shit)
# fig, ax = plt.subplots(nrow, ncol, figsize=(20, 15), layout="constrained")
fig, ax = plt.subplots(nrow, ncol, figsize=(20, 15))

row = 0
col = 0
pf_idx = 0 # pass or fail index
insets = []
ts_insets = []
for i in range(len(Xs)):
    #######################################
    # plot
    #######################################
    loadname = f"/BICNAS2/ycatal/meg_intdiff/scripts/erps/{taskname}_erp_permutationtest_st_emo_{tests[i]}.pkl"
    cluster_results = pklload(loadname)
    
    p_accept = 0.001
    good_cluster_inds = np.where(cluster_results["cluster_p"] < p_accept)[0]

    # configure variables for visualization
    colors = {comparisons[i][j]: colors_list[i][j] for j in range(len(colors_list[i]))}
    
    # organize data for plotting
    # evokeds = {cond: mne.grand_average(tasks[cond]) for cond in comparisons[i]}
    evokeds = {cond: tasks[cond] for cond in comparisons[i]}

    # find the two largest clusters and only use them (maybe not????)
    space_lengths = []
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        _, i_space_inds = np.squeeze(cluster_results["clusters"][clu_idx])
        space_lengths.append(len(np.unique(i_space_inds)))
    large_idx = [np.where(np.array(space_lengths) == j)[0][0] for j in np.sort(space_lengths)[-1:-3:-1]]


    # loop over clusters
    # for i_clu, clu_idx in enumerate(good_cluster_inds[large_idx]): # Code for only largest clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        if not pass_or_fail[pf_idx]:
            pf_idx += 1
            continue
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(cluster_results["clusters"][clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        i_chlist = template_bad.info["ch_names"]
        
        badchan = np.where(chlist == np.setdiff1d(chlist, i_chlist))[0]

        # get signals at the sensors contributing to the cluster
        times = evokeds[comparisons[i][0]][0].times
        sig_times = template_bad.times[time_inds]

        # Find the matching picks
        goodchans = np.array(chlist)[ch_inds]
        picks = []
        for k in goodchans:
            idx = np.where(k == np.array(i_chlist))[0]
            if len(np.where(k == np.array(i_chlist))[0]) == 0:
                continue
            else:
                picks.append(idx[0])
        picks = np.array(picks)


        # Average the activity inside the shaded area for each subject
        erf_violins = {comparisons[i][k]: [] for k in range(len(comparisons[i]))}
        for k, k_comp in enumerate(comparisons[i]):
            nsubj = len(evokeds[k_comp])
            for l in range(nsubj):
                erf_violins[k_comp].append(np.sqrt(np.mean( # rms
                    evokeds[k_comp][l].get_data()[np.ix_(picks, time_inds)] ** 2, axis=1))
                      * 1e15) # 1e15: tesla to femtotesla

        for m, m_key in enumerate(erf_violins.keys()):
            # if col == 0:
            #     label = f"{comparisons_nicer3[i][m]} $ \\rho $={rho:.2f}{p2str(p)}"
            # else:
            #     label = f"$ \\rho $={rho:.2f}{p2str(p)}"
            rho = rhos[row, col, m]
            p = pvals_correct[row, col, m]
            label = f"{comparisons_nicer3[i][m]} $ \\rho $={rho:.2f}{p2str(p)}"
            # label = f"{comparisons_nicer3[i][m]}"
            ax[row, col].scatter(np.nanmean(restmean[picks, :], axis=1), np.nanmean(np.array(erf_violins[m_key]), axis=0), c=colors[m_key], label=label)
            m, n = np.polyfit(np.nanmean(restmean[picks, :], axis=1), np.nanmean(np.array(erf_violins[m_key]), axis=0), 1)
            ax[row, col].plot(np.nanmean(restmean[picks, :], axis=1), m * np.nanmean(restmean[picks, :], axis=1) + n, c=colors[m_key])
        
        ax[row, col].legend(frameon=True, fancybox=False)
        ax[row, col].spines[['right', 'top']].set_visible(False)
        # ax[row, col].set_xlim((0.008, 0.02))
        # ax[row, col].set_xlim((0.0, 0.02))
        ax[row, col].set_ylim((-10, 105))
        ax[row, col].grid(visible=True)

        if col != 0:
            ax[row,col].set_yticklabels([])
        if row == 0:
            ax[row,col].set_xticklabels([])

        ax[row,col].set_yticks(np.arange(20, 110, 20))
        if col == 0:
            ax[row,col].set_ylabel("ERF")
        if row == 1:
            ax[row,col].set_xlabel("ACW")

        ################################## DO THE INSETS (FUUUUUUUUUUUUUUUUUUUUUCK) ##################################
        # plot average test statistic and mark significant sensors
        insets.append(inset_axes(ax[row, col], width="20%", height="20%", loc="lower left"))
                                 

        mask = np.zeros((271, 1), dtype=bool) 
        mask[picks, :] = True

        mne.viz.plot_topomap(data=np.zeros(271), pos=template_bad.info,
            mask=mask,
            axes=insets[-1],
            cmap="viridis",
            vlim=(np.min, np.max),
            show=False,
            mask_params=dict(markersize=7),
        )
        
        
        # add new axis for time courses and plot time courses
        ts_insets.append(inset_axes(ax[row, col], width="70%", height="20%", loc="lower right"))
        
        mne.viz.plot_compare_evokeds(
            evokeds,
            picks=picks,
            axes=ts_insets[-1],
            colors=colors,
            # linestyles=linestyles,
            show=False,
            legend=False,
            truncate_yaxis="auto",
            truncate_xaxis=False,
            ci=False
        )
        ts_insets[-1].plot()
        ts_insets[-1].set_title("")

    # plot temporal cluster extent
        ymin, ymax = ts_insets[-1].get_ylim()
        ts_insets[-1].fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )

        ts_insets[-1].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ts_insets[-1].set_frame_on(False)
        ts_insets[-1].axis('off')


        ##############################################################################################################

        pf_idx += 1
        col += 1
        if col == 3:
            col = 0
            row += 1        

plt.tight_layout()
fig.savefig(pathjoin(figpath, "f5_acw_fooof2.png"), dpi=300, transparent=False)



