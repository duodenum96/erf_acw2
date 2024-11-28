# Firs round of preprocessing
# First round: autoreject -> ICA
# Second round: ICA visualization and marking bad ics
# Third round: Reject bad ICs and run autoreject again
import os
import sys

subj = sys.argv[1]
from os.path import join as pathjoin
import mne
import numpy as np
import scipy as sp
import autoreject
import matplotlib.pyplot as plt
from time import time

tic = time()
##### Set up paths, create directories for output
datadir = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

subj_preprocpath = pathjoin(preprocpath, subj)
if not os.path.isdir(pathjoin(subj_preprocpath, "rest")):
    os.mkdir(pathjoin(subj_preprocpath, "rest"))
    os.mkdir(pathjoin(subj_preprocpath, "rest", "rejection_ica"))

outputpath = pathjoin(subj_preprocpath, "rest", "rejection_ica")

subjdir = os.path.join(datadir, subj, "ses-01", "meg")
os.chdir(subjdir)

rawdata = mne.io.read_raw_ctf(
    subj + "_ses-01_task-rest_run-01_meg.ds", preload=True
)
data = rawdata.copy().filter(l_freq=1, h_freq=None) # filter before autoreject + ICA

epochs = mne.make_fixed_length_epochs(data, duration=3, preload=True)
megchans = mne.pick_types(epochs.info, meg="mag", exclude=[])
epochs.pick(megchans) # pick meg channels

rejectlog_filename = pathjoin(outputpath, "reject_log.npz")
if os.path.isfile(rejectlog_filename):
    reject_log = autoreject.read_reject_log(rejectlog_filename)
    print(f"Autoreject already done")
else:
    ar = autoreject.AutoReject(random_state=666, n_jobs=20, verbose=True) 
    ar.fit(epochs)

    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    reject_log.save(pathjoin(outputpath, "reject_log.npz")) # save rejection log (must end with .npz)
    toc = time() - tic
    print(f"Autoreject completed, elapsed time: {toc} seconds")

if np.any(reject_log.bad_epochs):
    figure = epochs[reject_log.bad_epochs].plot()
    figure.savefig(pathjoin(outputpath, "autoreject_beforeICA.jpg"))

figure = reject_log.plot("horizontal")
figure.savefig(pathjoin(outputpath, "rejectlog_beforeICA.jpg"))

ica = mne.preprocessing.ICA(n_components=20, random_state=666)
ica.fit(epochs[~reject_log.bad_epochs])

toc = time() - tic
print(f"ICA completed, elapsed time: {toc} seconds")

figure = ica.plot_components()
plt.savefig(pathjoin(outputpath, "ICA_components.jpg"))
plt.close()

ica.plot_sources(rawdata, show_scrollbars=False)
plt.savefig(pathjoin(outputpath, "ICA_sources.jpg"))
plt.close()

ica.plot_properties(rawdata, picks=np.arange(20))
plt.savefig(pathjoin(outputpath, "ICA_properties.jpg"))
plt.close()

ica.save(pathjoin(outputpath, subj + "_rest-ica.fif"))

toc = time() - tic
print(f"Subject {subj} done, elapsed time: {toc} seconds")