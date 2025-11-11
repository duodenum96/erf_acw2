import numpy as np
import mne
import sys
import os
from os.path import join as pathjoin
import time
from scipy.optimize import curve_fit

os.chdir("/BICNAS2/ycatal/erf_acw2/")
from erf_acw2.src import (
    get_commonsubj,
    pklsave,
    pklload,
    import_exampleraw,
    acf_oscillatory_function
)

subjlist = get_commonsubj()

outputpath = "/BICNAS2/ycatal/erf_acw2/results/int"

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
example_data = import_exampleraw()
fs = example_data.info["sfreq"]

example_source_path = pathjoin(preprocpath, subjlist[0], "rest", "source_lcmv.pkl")
example_source_data = pklload(example_source_path)
example_labels = np.array(example_source_data["labels"])

nlags = int(0.25 * fs)
n_rois = 185

all_acfs = np.zeros((len(subjlist), n_rois, 5001))
all_psds = np.zeros((len(subjlist), n_rois, 6001))

for i, i_subj in enumerate(subjlist):
    subj_preprocpath = pathjoin(preprocpath, i_subj)
    source_path = pathjoin(subj_preprocpath, "rest", "source_lcmv.pkl")
    source_data = pklload(source_path)
    subject_labels = np.array(source_data["labels"])

    acfs = source_data["acfs"] # (n_rois, n_lags)
    lags = np.arange(0.0, nlags / fs, 1.0 / fs)
    psds = np.nanmean(source_data["psd"], axis=0)
    for j, j_label in enumerate(subject_labels):
        if j_label in example_labels:
            ref_idx = np.where(example_labels == j_label)[0][0]
            acfunc = acfs[j, :]
            all_acfs[i, ref_idx, :] = acfunc
            all_psds[i, ref_idx, :] = psds[j, :]
        else:
            print(f"Warning: Label {j_label} from subject {i_subj} not found in reference labels")
    print(f"{i+1} / {len(subjlist)}")


import matplotlib.pyplot as plt

f, ax = plt.subplots(1, 3, figsize=(15, 5))
lags = np.arange(0.0, 5001 / fs, 1.0 / fs)
ax[0].plot(lags, np.reshape(np.transpose(all_acfs, (2,0, 1)), (5001, 185*56)), alpha=0.5)
ax[0].set_xlim((0, 0.5))
ax[1].plot(lags, np.nanmean(all_acfs, axis=1).T)
ax[1].set_xlim((0, 0.5))
ax[1].set_title("Averaged over ROIs")
ax[2].plot(lags, np.nanmean(all_acfs, axis=0).T)
ax[2].set_title("Averaged over Subjects")
ax[2].set_xlim((0, 0.5))
f.savefig(pathjoin(outputpath, "acfs.png"))

freqs = source_data["freqs"]
f, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].loglog(freqs, np.reshape(np.transpose(all_psds, (2,0, 1)), (6001, 185*56)), alpha=0.5)
ax[0].set_xlim((1, 100))
ax[1].loglog(freqs, np.nanmean(all_psds, axis=1).T, alpha=0.5)
ax[1].set_title("Averaged over ROIs")
ax[1].set_xlim((1, 100))
ax[2].loglog(freqs, np.nanmean(all_psds, axis=0).T, alpha=0.5)
ax[2].set_title("Averaged over Subjects")
ax[2].set_xlim((1, 100))
f.savefig(pathjoin(outputpath, "psds.png"))

