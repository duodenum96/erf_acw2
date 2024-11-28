# Second round of preprocessing
# First round: autoreject -> ICA
# Second round: ICA visualization and marking bad ics
# Third round: Reject bad ICs and run autoreject again
import os
import sys

subj = sys.argv[1]
from os.path import join as pathjoin
import mne
import mne.preprocessing as pp
import numpy as np
import scipy as sp
# import autoreject
import matplotlib.pyplot as plt
from time import time


tic = time()
##### Set up paths, create directories for output
datadir = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
icapath = pathjoin(preprocpath, "all_ICAs_haririhammer")

subj_preprocpath = pathjoin(preprocpath, subj)

outputpath = pathjoin(subj_preprocpath, "haririhammer", "rejection_ica")

subjdir = os.path.join(datadir, subj, "ses-01", "meg")
os.chdir(subjdir)

raw = mne.io.read_raw_ctf(subj + "_ses-01_task-haririhammer_run-01_meg.ds", preload=True)
raw_filt = raw.copy().filter(l_freq=1, h_freq=None)  # filter before autoreject + ICA

ica = pp.read_ica(pathjoin(outputpath, subj + "_haririhammer_continuous-ica.fif"))
sources = ica.get_sources(raw)
times = sources.times

fig, axes = plt.subplots(20, 3, figsize=(25, 100))
for i in range(20):
    data = sources.get_data(picks=i, tmin=0, tmax=10)
    axes[i, 1].plot(times[(times >= 0) & (times < 10)], data[0, :])
    spectra = sources.compute_psd(picks=i, fmin=1, fmax=100)
    f = spectra.freqs
    psd = spectra.get_data()
    axes[i, 2].semilogy(f, psd[0])
    axes[i, 2].set_xticks([10, 20, 40, 60, 80])
    axes[i, 2].vlines([10, 60], np.min(psd), np.max(psd), color="r")
    pp.ica.plot_ica_components(ica, picks=i, axes=axes[i, 0], show=False)

fig.suptitle(subj, fontsize=16)
fig.savefig(pathjoin(icapath, subj+"_ica.jpg"))
