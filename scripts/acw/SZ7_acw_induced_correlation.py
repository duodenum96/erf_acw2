import numpy as np
from erf_acw2.src import pklload, pklsave, get_commonsubj, import_exampleraw
from erf_acw2.meg_chlist import chlist
import fooof
import mne

chlist = np.asarray(chlist)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import join as pathjoin
import pingouin as pg
from matplotlib import colormaps

figfolder = "/BICNAS2/ycatal/erf_acw2/figures/figs/reviewer_comments"

nchan = 272

task = "haririhammer"

subjs_common = get_commonsubj()
nsubj = subjs_common.shape[0]

acw_ints = np.nanmean(
    pklload(f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl")["rest_acw_50s"],
    axis=0
)

filename = (
    f"/BICNAS2/ycatal/erf_acw2/results/erf/" + 
    f"all_data_induced_fooof.pkl"
)

induced_data = pklload(filename)
selected_freqbands = induced_data["selected_freqbands"]
tasknames = induced_data["tasknames"]
induced_data = induced_data["all_data"]
n_tasks = len(tasknames)
n_freqs = len(selected_freqbands)

freq_nicenames = ["Theta", "Beta", "Gamma"]
task_nicenames = ["Encode Face Happy", "Encode Face Sad", "Encode Shape", 
                  "Probe Face Happy", "Probe Face Sad", "Probe Shape"]

nchan = 272

raw = import_exampleraw()

corrs = np.zeros((nchan, n_tasks, n_freqs))
pvals = np.zeros((nchan, n_tasks, n_freqs))

for i in range(nchan):
    for j in range(n_tasks):
        for k in range(n_freqs):
            i_corr_result = pg.corr(acw_ints[i, :], induced_data[tasknames[j]][selected_freqbands[k]][:, i])
            corrs[i, j, k] = i_corr_result["r"].values[0]
            pvals[i, j, k] = i_corr_result["p-val"].values[0]

ps_correct = pg.multicomp(pvals, method="fdr_bh")[1]

# fig = plt.figure(layout="constrained", figsize=(7.20472,7.87402))
mask = ps_correct < 0.05
# colorbar stuff
fraction=0.05
pad=0.04

for j, j_task in enumerate(tasknames):
    for k, k_freq in enumerate(selected_freqbands):
        # Draw the resting state
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Set font size for all axes
        fontsize = 18
        cb_fontsize = 14
        cb_title_fontsize = 16

        im0, cm0 = mne.viz.plot_topomap(
            corrs[:, j, k], raw.info, axes=ax[1], res=500, size=15, cmap=colormaps["cool"], mask=mask
        )
        cb = fig.colorbar(im0, fraction=fraction, pad=pad)
        cb.set_label("r-value", rotation=270, labelpad=15, fontsize=cb_title_fontsize)
        # cb.ax.tick_params(labelsize=fontsize)
        ax[1].set_title(f"{task_nicenames[j]}\n{freq_nicenames[k]}", fontsize=fontsize)
        ax[1].tick_params(axis='both', which='major', labelsize=fontsize)

        # colorbar stuff
        im1, cm1 = mne.viz.plot_topomap(
            ps_correct[:, j, k], raw.info, axes=ax[2], res=500, size=15, cmap=colormaps["cool"], mask=mask
        )
        cb = fig.colorbar(im1, fraction=fraction, pad=pad)
        cb.set_label("p value (corrected)", rotation=270, labelpad=15, fontsize=cb_title_fontsize)
        # cb.ax.tick_params(labelsize=fontsize)
        ticks = cb.get_ticks()
        cb.set_ticks(np.append(ticks, 0.05))
        ax[2].tick_params(axis='both', which='major', labelsize=fontsize)

        average_induced = np.nanmean(induced_data[j_task][k_freq], axis=0)
        # colorbar stuff
        im2, cm2 = mne.viz.plot_topomap(
            average_induced, raw.info, axes=ax[0], res=500, size=15, cmap=colormaps["cool"]
        )
        cb = fig.colorbar(im2, fraction=fraction, pad=pad)
        cb.set_label("Induced Power", rotation=270, labelpad=15, fontsize=cb_title_fontsize)
        # cb.ax.tick_params(labelsize=fontsize)
        im2.set_clim((average_induced.min(), average_induced.max()))
        ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

        # Optionally, set font size for all axes labels (if any)
        # for a in ax:
        #     for label in (a.get_xticklabels() + a.get_yticklabels()):
        #         label.set_fontsize(fontsize)

        fig.savefig(pathjoin(figfolder, "induced", f"{j_task}_{k_freq}.png"), dpi=300, transparent=True)

print("done")
