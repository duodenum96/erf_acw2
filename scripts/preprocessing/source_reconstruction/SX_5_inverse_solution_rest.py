import mne
import numpy as np
import os
from os.path import join as pathjoin
import sys
from erf_acw2.src import get_commonsubj
os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from scipy import signal
import matplotlib.pyplot as plt
from erf_acw2.src import pklsave

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

i = int(sys.argv[1])
subjlist = get_commonsubj()
i_subj = subjlist[i]

if i_subj in exclude:
    sys.exit("Bad Subject")

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
subj_raw_path = pathjoin(data_root, i_subj, "ses-01", "meg")

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "rest", "rejection_ica")

data = mne.read_epochs(pathjoin(outputpath, i_subj + "_rest_preprocessed-epo.fif.gz"))

# Load the required ingredients: covariance matrix, forward solution, source space
fwd = mne.read_forward_solution(pathjoin(subj_preprocpath, "rest", "forward-fwd.fif"))
src = fwd["src"]

# Load noise covariance
noise_cov = mne.read_cov(pathjoin(subj_preprocpath, "noise", "rejection_ica", f"{i_subj}_noise_cov.fif"))

# Compute inverse solution
inverse_operator = mne.minimum_norm.make_inverse_operator(
    data.info, fwd, noise_cov
)

mne.minimum_norm.write_inverse_operator(
    pathjoin(subj_preprocpath, "rest", "inverse-inv.fif"), inverse_operator
)


# See https://mne.discourse.group/t/estimating-the-regularization-parameter-for-hcp-meg-resting-state-source-reconstruction/5177/15
snr = 1.0
lambda2 = 1.0 / snr**2

method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
stc = mne.minimum_norm.apply_inverse_epochs(
    data,
    inverse_operator,
    lambda2,
    method=method,
    verbose=True,
)

pklsave(pathjoin(subj_preprocpath, "rest", "stc.pkl"), stc)

# freqs, psd = signal.periodogram(all_data, fs=data.info["sfreq"])

# psd_mean = psd.mean(axis=0) # shape: (n_sources, n_times)
# f, ax = plt.subplots()
# ax.loglog(freqs, psd_mean)
# ax.set_xlabel("Frequency (Hz)")
# ax.set_ylabel("PSD (V^2/Hz)")
# ax.set_title("PSD of the source estimate")
# f.savefig(pathjoin(subj_preprocpath, "rest", "source_psd.png"))

