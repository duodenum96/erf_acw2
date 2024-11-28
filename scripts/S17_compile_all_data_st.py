import os
from os.path import join as pathjoin
import numpy as np
from erf_acw2.src import pklload, pklsave, get_commonsubj
from erf_acw2.src_importdata import get_template_bad
import mne
from erf_acw2.meg_chlist import chlist
import pandas as pd

######################### Prelude  ##############################
taskname = "haririhammer"

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

# Import ACW data
acw_results = pklload(f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl")
acws_all = acw_results["rest_acw_50s"]
acws = np.nanmean(acws_all, axis=0)

behavior_filename = (
    "/BICNAS2/ycatal/meg_intdiff/scripts/behavior/haririhammer_beh_measures.pkl"
)
beh = pklload(behavior_filename)
speed = beh["speed"]
speed_tiled = np.tile(speed, (272, 1)).T


#########################################################################################################

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure6"

tasknames = [
    "encode_face_happy",
    "probe_face_happy",
    "encode_face_sad",
    "probe_face_sad",
    "encode_shape",
    "probe_shape",
]

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

##################################################################################################################################

all_effects = ["A:B", "A", "B"]
effect_names = ["factor_interaction", "factor_encprob", "factor_emo"]

tests = ["factor_emo", "factor_emo"]

Xs = [
    [tasks["encode_face_happy"], tasks["encode_face_sad"], tasks["encode_shape"]],
    [tasks["probe_face_happy"], tasks["probe_face_sad"], tasks["probe_shape"]],
]

comparisons = [
    ["encode_face_happy", "encode_face_sad", "encode_shape"],
    ["probe_face_happy", "probe_face_sad", "probe_shape"],
]

comparisons_nicer = [
    ["encode face happy", "encode face sad", "encode shape"],
    ["probe face happy", "probe face sad", "probe shape"],
]
comparisons_nicer3 = [
    ["encode happy", "encode sad", "encode shape"],
    ["probe happy", "probe sad", "probe shape"],
]
comparisons_short = [["efh", "efs", "es"], ["pfh", "pfs", "ps"]]

comparisons_nicer2 = ["Happy - Sad - Shape", "Happy - Sad - Shape"]
suptitles = ["Encode", "Probe"]


pass_or_fail = pklload("/BICNAS2/ycatal/erf_acw/figures/figs/figure2/pass_or_fail.pkl")[
    "pass_or_fail"
][
    0:12
]  # first 12 items are from first 2 comparisons

# Get x and y coordinates as prior for gaussian process
template_good = get_template_bad(good=True)
info = template_good.info
lay = mne.channels.find_layout(info)
laynames = lay.names
chnames = [i[0:5] for i in info["ch_names"]]
xcoord = np.zeros(272)
ycoord = np.zeros(272)
for i in range(272):
    where = np.where(np.array(laynames) == chnames[i])[0][0]
    xcoord[i] = lay.pos[where, 0]
    ycoord[i] = lay.pos[where, 1]

# Initialize empty lists (i hate life)
clusters = []
subjects = []
erfs = []
restacws = []
rows = []
cols = []
channels = []
rts = []
xcoords = []
ycoords = []
erftype = []
clusternames = []
cluster_names = ["Encode #1", "Encode #2", "Encode #3", "Encode #4", "Probe #1", "Probe #2"]

nrow = 2
ncol = 3

row = 0
col = 0
pf_idx = 0  # pass or fail index
c = 0
for i in range(len(Xs)):
    #######################################
    # plot
    #######################################
    loadname = f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}/{taskname}_erp_permutationtest_st_emo_{tests[i]}.pkl"
    cluster_results = pklload(loadname)

    p_accept = 0.001
    good_cluster_inds = np.where(cluster_results["cluster_p"] < p_accept)[0]

    # organize data for plotting
    # evokeds = {cond: mne.grand_average(tasks[cond]) for cond in comparisons[i]}
    evokeds = {cond: tasks[cond] for cond in comparisons[i]}

    # find the two largest clusters and only use them (maybe not????)
    space_lengths = []
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        _, i_space_inds = np.squeeze(cluster_results["clusters"][clu_idx])
        space_lengths.append(len(np.unique(i_space_inds)))
    large_idx = [
        np.where(np.array(space_lengths) == j)[0][0]
        for j in np.sort(space_lengths)[-1:-3:-1]
    ]

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
                erf_violins[k_comp].append(
                    np.sqrt(
                        np.mean(  # rms
                            evokeds[k_comp][l].get_data()[np.ix_(picks, time_inds)] ** 2, axis=1
                        )
                    )
                    * 1e15
                )  # 1e15: tesla to femtotesla

        # erf_violins[k_comp]: nsubj
        # restmean: nchan x nsubj
        # speed: nsubj
        for k, k_comp in enumerate(comparisons[i]):
            for m in range(nsubj):
                for i_channel, channel in enumerate(picks):
                    channels.append(channel)
                    erfs.append(erf_violins[k_comp][m][i_channel])
                    restacws.append(acws[channel, m])
                    subjects.append(m)
                    clusters.append(c)
                    rts.append(speed[m])
                    rows.append(row)
                    cols.append(col)
                    xcoords.append(xcoord[channel])
                    ycoords.append(ycoord[channel])
                    erftype.append(k_comp)
                    clusternames.append(cluster_names[c])


        pf_idx += 1
        col += 1
        c += 1
        if col == 3:
            col = 0
            row += 1

data = pd.DataFrame(
    {
        "erfs": erfs,
        "restacws": restacws,
        "subjects": subjects,
        "clusters": clusters,
        "rows": rows,
        "cols": cols,
        "channels": channels,
        "rts": rts,
        "xcoords": xcoords,
        "ycoords": ycoords,
        "erftype": erftype,
        "clusternames": clusternames
    }
)

data.to_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")