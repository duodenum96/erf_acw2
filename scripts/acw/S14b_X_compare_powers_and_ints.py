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


from fooof.bands import Bands

figfolder = "/BICNAS2/ycatal/erf_acw2/figures/figs/reviewer_comments"

bands = Bands({'theta' : [4, 8],
               'alpha' : [8, 12],
               'beta1' : [12, 20],
               'beta2' : [20, 30],
               "gamma" : [30, 50]})

nchan = 272

task = "haririhammer"

rest_ntp = 120
rest_ntp = int((rest_ntp * 3) / 10)

subjs_common = get_commonsubj()
nsubj = subjs_common.shape[0]

band_powers = pklload("/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_all_powers_2fit.pkl")
band_keys = list(bands.bands.keys())

acw_ints = np.nanmean(
    pklload(f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl")["rest_acw_50s"],
    axis=0
)
tau_ints = np.nanmean(
    pklload("/BICNAS2/ycatal/erf_acw2/results/int/rest_int_oscillatory_fit.pkl")["rest_acws"],
    axis=0
)

band_fooof_results = pklload(f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_all_fooof_results_2fit.pkl")

band_fooof_shape = band_fooof_results["alpha"].shape # channels, subjects, (center freq, power, bandwidth)
band_all_results = np.zeros(
    (band_fooof_shape[0], band_fooof_shape[1], band_fooof_shape[2], len(band_keys))
)

for i, i_key in enumerate(band_keys):
    band_all_results[:, :, :, i] = band_fooof_results[i_key]

max_power_band_idx = np.nanargmax(band_all_results[:, :, 1, :], axis=2)  # Find band with max power
max_power_center_freq = np.zeros((band_fooof_shape[0], band_fooof_shape[1]))  # (channels, subjects)
# Extract center frequency for the band with maximum power
for ch in range(band_fooof_shape[0]):
    for subj in range(band_fooof_shape[1]):
        band_idx = max_power_band_idx[ch, subj]
        max_power_center_freq[ch, subj] = band_all_results[ch, subj, 0, band_idx]



corr_results = pg.corr(acw_ints.ravel(), max_power_center_freq.ravel(), method="spearman")
f, ax = plt.subplots()
ax.scatter(acw_ints.ravel(), max_power_center_freq.ravel(), s=1, c="k")
ax.text(0.04, 2.5, f"r = {corr_results['r'].values[0]:.3f}, p = {corr_results['p-val'].values[0]:.3f}", fontsize=12)
f.savefig(pathjoin(figfolder,"acw_center_freq.png"))

nchan = 272

raw = import_exampleraw()

corrs = np.zeros(nchan)
pvals = np.zeros(nchan)

for i in range(nchan):
    i_corr_result = pg.corr(acw_ints[i, :], max_power_center_freq[i, :])
    corrs[i] = i_corr_result["r"].values[0]
    pvals[i] = i_corr_result["p-val"].values[0]

ps_correct = pg.multicomp(pvals, method="holm")[1]

# fig = plt.figure(layout="constrained", figsize=(7.20472,7.87402))
mask = ps_correct < 0.05
fig, ax = plt.subplots()
# colorbar stuff
fraction=0.05
pad=0.04
# Draw the resting state
im0, cm0 = mne.viz.plot_topomap(
    corrs, raw.info, axes=ax, res=500, size=15, cmap=colormaps["cool"], mask=mask
)
cb = fig.colorbar(im0, fraction=fraction, pad=pad)
cb.set_label("r-value", rotation=270, labelpad=15)

fig.savefig(pathjoin(figfolder, "correlations_r.jpg"), dpi=800)


fig, ax = plt.subplots()
# colorbar stuff
im1, cm1 = mne.viz.plot_topomap(
    ps_correct, raw.info, axes=ax, res=500, size=15, cmap=colormaps["cool"], mask=mask
)
cb = fig.colorbar(im1, fraction=fraction, pad=pad)
cb.set_label("p value (corrected)", rotation=270, labelpad=15)
colorbar_ticks = cb.get_ticks()
cb.set_ticks(np.append(colorbar_ticks, 0.05))
fig.savefig(pathjoin(figfolder, "correlations_p.jpg"), dpi=800)


average_center_freq = np.nanmean(max_power_center_freq, axis=1)
fig, ax = plt.subplots()
# colorbar stuff
im2, cm2 = mne.viz.plot_topomap(
    average_center_freq, raw.info, axes=ax, res=500, size=15, cmap=colormaps["cool"]
)
cb = fig.colorbar(im2, fraction=fraction, pad=pad)
cb.set_label("center frequency of maximum power (Hz)", rotation=270, labelpad=15)
im2.set_clim((average_center_freq.min(), average_center_freq.max()))

fig.savefig(pathjoin(figfolder, "center_freqs_per_chan.jpg"), dpi=800)

print("done")
