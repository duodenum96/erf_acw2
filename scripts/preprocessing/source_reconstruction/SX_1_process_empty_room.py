import mne
import os 
from os.path import join as pathjoin
import numpy as np
from erf_acw2.src import get_commonsubj, pklsave, pick_megchans, re_epoch
import autoreject
from time import time
import mne.preprocessing as pp
import sys
os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from time import time

i = int(sys.argv[1])

subjlist = get_commonsubj()
i_subj = subjlist[i]

if i_subj in exclude:
    sys.exit("Bad Subject")

data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
subj_raw_path = pathjoin(data_root, i_subj, "ses-01", "meg")

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "noise", "rejection_ica")
rest_outputpath = pathjoin(subj_preprocpath, "rest", "rejection_ica")

if not os.path.isdir(outputpath):
    os.makedirs(outputpath)

pp_filename = pathjoin(outputpath, i_subj + "_noise_preprocessed-epo.fif.gz")

# Load the empty room recording
noise_file = f"{i_subj}_ses-01_task-noise_run-01_meg.ds"
noise_path = pathjoin(subj_raw_path, noise_file)

# Load ICA from rest preprocessing
ica = pp.read_ica(pathjoin(rest_outputpath, i_subj + "_rest-ica.fif"))
ica.exclude = badics[i_subj]

rawdata = mne.io.read_raw_ctf(noise_path, preload=True)
reconst_raw = rawdata.copy().filter(l_freq=1, h_freq=None) # filter before autoreject + ICA

ica.apply(reconst_raw)
reconst_raw.filter(l_freq=None, h_freq=100.0).notch_filter(freqs=[60])


epochs = mne.make_fixed_length_epochs(reconst_raw, duration=3, preload=True)
megchans = mne.pick_types(epochs.info, meg="mag", exclude=[])
epochs.pick(megchans)  # pick meg channels

tic = time()
ar = autoreject.AutoReject(random_state=666, n_jobs=8, verbose=True)
ar.fit(epochs)
toc = time() - tic
print(f"Autoreject completed, elapsed time: {toc} seconds")

epochs_ar2, reject_log = ar.transform(epochs, return_log=True)
reject_log.save(pathjoin(outputpath, "reject_log_epochs.npz"), overwrite=True)

epochs_ar2.save(
    pathjoin(outputpath, i_subj + "_noise_preprocessed-epo.fif.gz"), fmt="double", overwrite=True
)

print(f"Epochs saved to {pathjoin(outputpath, i_subj + '_noise_preprocessed-epo.fif.gz')}")
print(f"Subject {i_subj} done")