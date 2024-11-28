##### Visualize spatiotemporal clusters
# In the main figure we only report significant clusters (after multiple comparisons)
# Pass or fail indicates those

import numpy as np
import mne
from os.path import join as pathjoin
import matplotlib.pyplot as plt
from erf_acw2.meg_chlist import chlist
from erf_acw2.src import pklsave, pklload, get_commonsubj, p2str
import scipy as sp
from mne.viz import plot_compare_evokeds
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import pingouin as pg
import pandas as pd

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure2"
taskname = "haririhammer"

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

tasknames = ["encode_face_happy", "probe_face_happy", 
             "encode_face_sad", "probe_face_sad", 
             "encode_shape", "probe_shape"]

tasks = {i: [] for i in tasknames}

for i, i_subj in enumerate(subjs_common):
    filename = pathjoin(
        f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}",
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

tests = ["factor_emo", "factor_emo", "factor_encprob", "factor_encprob", "factor_encprob"]

stat_text_coords = [0.5, 1.5, 1.0]
stat_text_ycoords = [115, 115, 125]
stat_line_coords = [[0,0,1,1], [1,1,2,2], [0,0,2,2]]
stat_line_ycoords = [[110,115,115,110], [110,115,115,110], [115,125,125,115]]

Xs = [
    [tasks["encode_face_happy"], tasks["encode_face_sad"], tasks["encode_shape"]], 
    [tasks["probe_face_happy"], tasks["probe_face_sad"], tasks["probe_shape"]],
    [tasks["encode_face_happy"], tasks["probe_face_happy"]],
    [tasks["encode_face_sad"], tasks["probe_face_sad"]],
    [tasks["encode_shape"], tasks["probe_shape"]]
]

comparisons = [
    ["encode_face_happy", "encode_face_sad", "encode_shape"], 
    ["probe_face_happy", "probe_face_sad", "probe_shape"],
    ["encode_face_happy", "probe_face_happy"],
    ["encode_face_sad", "probe_face_sad"],
    ["encode_shape", "probe_shape"]
]

comparisons_nicer = [
    ["encode face happy", "encode face sad", "encode shape"], 
    ["probe face happy", "probe face sad", "probe shape"],
    ["encode face happy", "probe face happy"],
    ["encode face sad", "probe face sad"],
    ["encode shape", "probe shape"]
]
comparisons_short = [
    ["efh", "efs", "es"], 
    ["pfh", "pfs", "ps"],
    ["efh", "pfh"],
    ["efs", "pfs"],
    ["es", "ps"]
]

comparisons_nicer2 = ["Happy - Sad - Shape", "Happy - Sad - Shape", "Encode - Probe", "Encode - Probe", "Encode - Probe"]
suptitles = ["Encode", "Probe", "Happy", "Sad", "Shape"]

adj = pklload("/BICNAS2/ycatal/erf_acw2/erf_acw2/adjacency.pkl")
ttestfunc = lambda a, b: sp.stats.ttest_ind(a, b, nan_policy="omit")[0]
matplotlib.rcParams.update({'font.size': 16})

# Colors: 
# encode face happy: crimson, encode face sad: steelblue, encode shape: darkorchid
# probe face happy: maroon, probe face sad: turquoise, probe shape: forestgreen
colors_list = [
    ["crimson", "steelblue",  "darkorchid"],
    ["maroon", "turquoise", "forestgreen"],
    ["crimson", "maroon"],
    ["steelblue", "turquoise"],
    ["darkorchid", "forestgreen"]
]

pass_or_fail = [] # is there a significance after violin plots
multcomps = []
for i in range(len(Xs)):
    #######################################
    # plot
    #######################################
    loadname = f"/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer/{taskname}_erp_permutationtest_st_emo_{tests[i]}.pkl"
    cluster_results = pklload(loadname)
    
    p_accept = 0.001
    good_cluster_inds = np.where(cluster_results["cluster_p"] < p_accept)[0]

    # configure variables for visualization
    colors = {comparisons[i][j]: colors_list[i][j] for j in range(len(colors_list[i]))}
    
    # organize data for plotting
    # evokeds = {cond: mne.grand_average(tasks[cond]) for cond in comparisons[i]}
    evokeds = {cond: tasks[cond] for cond in comparisons[i]}

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(cluster_results["clusters"][clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        f_map = cluster_results["stats"][time_inds, ...].mean(axis=0)
        i_chlist = template_bad.info["ch_names"]
        
        badchan = np.where(chlist == np.setdiff1d(chlist, i_chlist))[0]
        f_map_clean = np.delete(f_map, badchan)

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

        # create spatial mask
        mask = np.zeros((f_map_clean.shape[0], 1), dtype=bool)
        mask[picks, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(19, 4), layout="constrained")

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(f_map_clean[:, np.newaxis], template_bad.info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap="viridis",
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
        )
        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title(f"{suptitles[i]}     \n{comparisons_nicer2[i]}     ")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged F-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )
        ax_topo.ticklabel_format(useMathText=True)

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = "Cluster #{0}, {1} sensor".format(i_clu + 1, len(ch_inds))
        

        if len(ch_inds) > 1:
            title += "s"
            None
        
        plot_compare_evokeds(
            evokeds,
            title=title,
            picks=picks,
            axes=ax_signals,
            colors=colors,
            # linestyles=linestyles,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
            truncate_xaxis=False,
            ci=True
        )
        ax_signals.plot()

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )

        # Add new axis for comparing ERFs
        ax_comparison = divider.append_axes("right", size="100%", pad=1.2)
        title = "Cluster #{0}, {1} sensor".format(i_clu + 1, len(ch_inds))

        # Average the activity inside the shaded area for each subject
        erf_violins = {comparisons[i][k]: [] for k in range(len(comparisons[i]))}
        for k, k_comp in enumerate(comparisons[i]):
            nsubj = len(evokeds[k_comp])
            for l in range(nsubj):
                erf_violins[k_comp].append(np.sqrt(np.mean( # rms
                    evokeds[k_comp][l].get_data()[np.ix_(picks, time_inds)] ** 2))
                      * 1e15) # 1e15: tesla to femtotesla

        violins = ax_comparison.violinplot([erf_violins[m] for m in erf_violins.keys()], positions=np.arange(len(erf_violins)), showextrema=False)
        for i_pc, pc in enumerate(violins["bodies"]):
            pc.set_facecolor(colors[comparisons[i][i_pc]])
            pc.set_edgecolor("k")
        
        for m, m_key in enumerate(erf_violins.keys()):
            ax_comparison.scatter(m*np.ones_like(erf_violins[m_key]) + np.random.randn(len(erf_violins[m_key]))*0.1, erf_violins[m_key], color="black",
                                s=2)
            
        ax_comparison.set_xticks(np.arange(len(erf_violins.keys())), comparisons_short[i])
        ax_comparison.spines[['right', 'top']].set_visible(False)
        ax_comparison.set_ylabel("fT (AUC)")

        # Multiple comparisons
        multcomp = pg.pairwise_tests(pd.melt(pd.DataFrame(erf_violins), value_vars=erf_violins.keys()), dv="value", between="variable", 
                                     effsize="cohen", padjust="fdr_bh") # 1-2, 1-3, 2-3
        
        multcomps.append(multcomp)
        stattexts = []
        for m in range(len(multcomp)):
            m_row = multcomp.iloc[m, :]
            if len(multcomp) == 3:
                stattexts.append(f"{p2str(m_row["p-corr"])}")
            else:
                stattexts.append(f"{p2str(m_row["p-unc"])}")
        
        if len(multcomp) == 3:
            pass_or_fail.append(np.any(multcomp["p-corr"].to_numpy() < 0.05))
        else:
            pass_or_fail.append(np.any(multcomp["p-unc"].to_numpy() < 0.05))
        
        for m in range(len(multcomp)):
            plt.plot(stat_line_coords[m], stat_line_ycoords[m], color="black")
            plt.text(stat_text_coords[m], stat_text_ycoords[m], stattexts[m], ha='center', va='bottom', fontsize=10)
            
        ax_comparison.set_ylim((0, 140))

        

        fig.savefig(pathjoin(figpath, f"{taskname}_{i}_cluster{i_clu}_face_vs_shape.jpg"), dpi=800)

p_corrs = []
ts = []
cohens = []
As = []
Bs = []
for i, i_multcomp in enumerate(multcomps):
    if pass_or_fail[i]:
        if len(i_multcomp) != 3:
            p_corrs.append(i_multcomp["p-unc"].values[0])
            ts.append(i_multcomp["T"].values[0])
            cohens.append(i_multcomp["cohen"].values[0])
            As.append(i_multcomp["A"].values[0])
            Bs.append(i_multcomp["B"].values[0])
        else:
            for j in range(3):
                p_corrs.append(i_multcomp["p-corr"][j])
                ts.append(i_multcomp["T"][j])
                cohens.append(i_multcomp["cohen"][j])
                As.append(i_multcomp["A"][j])
                Bs.append(i_multcomp["B"][j])

pd.DataFrame({"As": As, "Bs": Bs, "T": ts, "p_corr": p_corrs, "cohen": cohens}).to_csv(pathjoin(figpath, f"multcomp_results.csv"))

pklsave(pathjoin(figpath, f"pass_or_fail.pkl"), {"pass_or_fail": pass_or_fail})
