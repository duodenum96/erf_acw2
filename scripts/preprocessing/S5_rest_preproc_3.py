# Third round of preprocessing
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
import autoreject
import matplotlib.pyplot as plt
from time import time

os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preproc")
from rest_badICs import exclude, badics

if subj in exclude:
    sys.exit("Bad Subject")

tic = time()
##### Set up paths, create directories for output
datadir = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
icapath = pathjoin(preprocpath, "all_ICAs")

subj_preprocpath = pathjoin(preprocpath, subj)

outputpath = pathjoin(subj_preprocpath, "rest", "rejection_ica")

subjdir = os.path.join(datadir, subj, "ses-01", "meg")
os.chdir(subjdir)

raw = mne.io.read_raw_ctf(subj + "_ses-01_task-rest_run-01_meg.ds", preload=True)

ica = pp.read_ica(pathjoin(outputpath, subj + "_rest-ica.fif"))
ica.exclude = badics[subj]

reconst_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
ica.apply(reconst_raw)
reconst_raw.filter(l_freq=None, h_freq=100.0).notch_filter(freqs=[60])

epochs = mne.make_fixed_length_epochs(reconst_raw, duration=3, preload=True)
megchans = mne.pick_types(epochs.info, meg="mag", exclude=[])
epochs.pick(megchans)  # pick meg channels

ar = autoreject.AutoReject(random_state=666, n_jobs=8, verbose=True)
ar.fit(epochs)

epochs_ar2, reject_log = ar.transform(epochs, return_log=True)
reject_log.save(pathjoin(outputpath, "reject_log_round2.npz"), overwrite=True)

epochs_ar2.save(
    pathjoin(outputpath, subj + "_rest_preprocessed-epo.fif.gz"), fmt="double", overwrite=True
)

