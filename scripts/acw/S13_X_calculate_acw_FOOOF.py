import numpy as np
import mne
import sys
import os
from os.path import join as pathjoin
import time
import fooof

os.chdir("/BICNAS2/ycatal/erf_acw2/")
from erf_acw2.src import (
    loop_acw_acf,
    get_commonsubj,
    pklsave,
    pick_megchans,
    re_epoch,
)

from scipy import signal

subjlist = get_commonsubj()
i = int(sys.argv[1])  # Goes from 0 to 66 
i_subj = subjlist[i]

ntrial = 120  # 120 for resting state

tic = time.time()
print(f"Starting subj: {i_subj}")
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "rest", "rejection_ica")
pp_filename = pathjoin(outputpath, i_subj + "_rest_preprocessed-epo.fif.gz")

epochs = mne.read_epochs(pp_filename)
epochs_meg = pick_megchans(epochs)
epochs_meg = re_epoch(epochs_meg, 10.0) # trials x channels x time

chanlist = epochs.info["ch_names"]
nchan = len(chanlist)
ntrial_new = int((ntrial * 3) / 10)

freqs, psds = signal.periodogram(epochs_meg.get_data(), fs=epochs_meg.info["sfreq"], window="hamming", axis=2)
average_psd = np.nanmean(psds, axis=0) # channels x freqs

freqs = freqs[1:] # remove 0 Hz
average_psd = average_psd[:, 1:] # remove 0 Hz

fmin, fmax = 3, 50

freqs_filtered = freqs[(freqs >= fmin) & (freqs <= fmax)]
average_psd_filtered = average_psd[:, (freqs >= fmin) & (freqs <= fmax)]

# Visually inspect the PSD to determine frequency
# import matplotlib.pyplot as plt
# f, ax = plt.subplots()
# ax.loglog(freqs_filtered, average_psd_filtered.T)
# f.savefig("anan.jpg")

fm = fooof.FOOOFGroup(aperiodic_mode='knee')
fm.fit(freqs_filtered, average_psd_filtered)
# fm.plot(save_fig=True, file_name="anan.jpg")

ints = fm.get_params("aperiodic_params", "knee")

toc = time.time() - tic
print(f"Elapsed time: {toc}")

fname = "/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest/" + i_subj + "_fooof.pkl"

results = {
    "ints": ints,
    "chanlist": chanlist,
    "i_subj": i_subj,
    "fm": fm,
}
pklsave(fname, results)

print("DONE")
