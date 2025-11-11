import numpy as np
from erf_acw2.src import pklload, pklsave, get_commonsubj
from erf_acw2.meg_chlist import chlist

chlist = np.asarray(chlist)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nchan = 272

task = "haririhammer"

rest_ntp = 120
rest_ntp = int((rest_ntp * 3) / 10)

subjs_common = get_commonsubj()
nsubj = subjs_common.shape[0]

n_freqs = 6000

nlags = 300
psds = np.zeros((n_freqs, nchan, nsubj))

all_chidx = np.arange(272)
######## Load rest and task ACWs, store in a numpy array
for i, i_subj in enumerate(subjs_common):
    psdsname = f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest/{i_subj}_psds.pkl"
    psds_i = pklload(psdsname)
    

    # If there is a missing channel, find it and fill with nans
    missingchan_rest = np.setdiff1d(chlist, psds_i["chanlist"])
    if len(missingchan_rest) != 0:
        missing_idx_rest = np.where(chlist == missingchan_rest)[0]
        good_idx_rest = np.setdiff1d(all_chidx, missing_idx_rest)

        psds[:, missing_idx_rest, i] = np.nan
    else:
        good_idx_rest = all_chidx.copy()

    psds[:, good_idx_rest, i] = psds_i["psds"].T
    freqs = psds_i["freqs"]

allpsds = {"psds": psds}

savename = f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_psds.pkl"
pklsave(savename, allpsds)

############################# Plot the Spectra



psds = pklload(savename)["psds"]

f, ax = plt.subplots(1, 3, figsize=(15, 5))

psds_flat = psds.reshape(n_freqs, -1)
# All PSDs

ax[0].loglog(freqs, psds_flat, alpha=0.2)
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Power")
ax[0].set_xlim(1.0, 50)
ax[0].set_ylim(1e-31, 1e-23)
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].set_title("All PSDs")

# 2) Average over subjects

ax[1].loglog(freqs, np.nanmean(psds, axis=2), alpha=0.5)
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Power")
ax[1].set_xlim(1.0, 50)
ax[1].set_ylim(1e-29, 1e-25)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].set_title("PSDs (averaged over subjects)")


# 3) Average over Channels

ax[2].loglog(freqs, np.nanmean(psds, axis=1), alpha=0.5)
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Power")
ax[2].set_xlim(1.0, 50)
ax[2].set_ylim(1e-29, 1e-25)
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)
ax[2].set_title("PSDs (averaged over channels)")

[ax[i].axvline([20]) for i in range(3)]
f.savefig(f"/BICNAS2/ycatal/erf_acw2/figures/figs/figure5/all_psds_find_knee.png")

f.savefig(f"/BICNAS2/ycatal/erf_acw2/figures/figs/figure5/all_psds.png", dpi=300, transparent=True)

# see https://github.com/fooof-tools/Development/issues/7
# see https://github.com/fooof-tools/fooof/issues/35

# Calculate the slope from 1 Hz to 8 Hz

freqs_filter = (freqs >= 1) & (freqs <= 5)
n_freqs_filter = np.sum(freqs_filter)
psds_filter = psds[freqs_filter, :, :]
psds_filter_flat = psds_filter.reshape(n_freqs_filter, -1)

slopes = np.zeros(psds_filter_flat.shape[1])
for i in range(psds_filter_flat.shape[1]):
    slopes[i] = np.polyfit(np.log(freqs[freqs_filter]), np.log(psds_filter_flat[:, i]), 1)[0]

f, ax = plt.subplots()
ax.hist(slopes, bins=100)
ax.set_xlabel("Slope")
ax.set_ylabel("Count")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
f.savefig(f"/BICNAS2/ycatal/erf_acw2/figures/figs/figure5/slopes.png", dpi=300, transparent=True)

