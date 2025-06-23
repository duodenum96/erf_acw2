import mne
import os 
from os.path import join as pathjoin
import numpy as np
from erf_acw2.src import get_commonsubj, pklsave, pick_megchans, re_epoch
import autoreject
from time import time
import mne.preprocessing as pp
import sys
import matplotlib.pyplot as plt
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
inputpath = pathjoin(subj_preprocpath, "haririhammer", "rejection_ica")

epochs_ar2 = mne.read_epochs(pathjoin(inputpath, i_subj + "_haririhammer_preprocessed-epo.fif.gz"))

cov = mne.compute_covariance(epochs_ar2, method="auto", tmax=0)
cov.plot(epochs_ar2.info, proj=True)
plt.savefig(pathjoin(outputpath, i_subj + "_prestim_cov.jpg"))

mne.write_cov(pathjoin(outputpath, i_subj + "_prestim-cov.fif"), cov)

print(f"Covariance saved to {pathjoin(outputpath, i_subj + '_prestim-cov.fif')}")
print(f"Subject {i_subj} done")

