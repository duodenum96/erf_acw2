import mne
import numpy as np
import os
from os.path import join as pathjoin
import sys
from scipy import signal
from erf_acw2.src import get_commonsubj

os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from scipy import signal
import matplotlib.pyplot as plt
from erf_acw2.src import pklload, pklsave, pick_megchans, re_epoch
from statsmodels.tsa.stattools import acf

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
epochs_meg = pick_megchans(data)
data_cov = mne.compute_covariance(epochs_meg, method="auto")
epochs_meg = re_epoch(epochs_meg, 10.0, conservative=True)


# Load the required ingredients: covariance matrix, forward solution, source space
fwd = mne.read_forward_solution(pathjoin(subj_preprocpath, "rest", "forward_volume-fwd.fif"))
src = fwd["src"]

# Load noise covariance
noise_cov = mne.read_cov(
    pathjoin(subj_preprocpath, "noise", "rejection_ica", f"{i_subj}_noise_cov.fif")
)


filters = mne.beamformer.make_lcmv(
    epochs_meg.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)

stcs = mne.beamformer.apply_lcmv_epochs(epochs_meg, filters)

labels_path = pathjoin(subjects_dir, i_subj, "mri", "aparc.a2009s+aseg.mgz")
parcellated_ts = mne.extract_label_time_course(
    stcs, labels_path, src, # allow_empty=True
)

labels = mne.get_volume_labels_from_aseg(labels_path)
# print(labels)

catenated_data = np.array(parcellated_ts)


# freqs, psds = signal.periodogram(catenated_data, fs=epochs_meg.info["sfreq"], window="hamming", axis=2)

nlags = 5000

n_good_trials = np.sum(np.all(~np.isnan(catenated_data), axis=(1, 2)))

print("Starting ACF calculation")

acfs = np.zeros((catenated_data.shape[1], nlags+1))
for i in range(catenated_data.shape[0]):
    for j in range(catenated_data.shape[1]):
        data = catenated_data[i, j, :]
        if np.isnan(data).any():
            print(f"NaN found in data for {i_subj} {j}")
            continue
        autocorrelation_func = acf(data, nlags=nlags)
        acfs[j, :] += autocorrelation_func
    print(f"{i+1} / {catenated_data.shape[0]}")

acfs /= n_good_trials

freqs, psd = signal.periodogram(catenated_data, fs=epochs_meg.info["sfreq"], window="hamming", axis=2)

pklsave(pathjoin(subj_preprocpath, "rest", "source_lcmv.pkl"), {"acfs": acfs, "catenated_data": catenated_data, "freqs": freqs, "psd": psd, "labels": labels})
print(f"Saved source_lcmv.pkl for {i_subj}")
