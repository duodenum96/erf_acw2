import os
import numpy as np
from fooof import FOOOF
import time
import sys

os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from scipy import signal
import matplotlib.pyplot as plt
from erf_acw2.src import pklload, pklsave, pick_megchans, re_epoch, get_commonsubj, pathjoin
import fooof

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


tic = time.time()

# Load source PSDs
source_psds = pklload(pathjoin(subj_preprocpath, "rest", "source_psds.pkl"))
freqs = source_psds["freqs"]
psds = source_psds["psds"] # Trials x vertices x frequencies

toc = time.time()
print(f"Loaded source PSDs in {toc - tic} seconds")

# Average PSDs across epochs
average_psd = np.nanmean(psds, axis=0)

fmin, fmax = 3, 50

freqs_filtered = freqs[(freqs >= fmin) & (freqs <= fmax)]
average_psd_filtered = average_psd[:, (freqs >= fmin) & (freqs <= fmax)]

tic = time.time()
fm = fooof.FOOOFGroup(aperiodic_mode='knee', max_n_peaks=3)
fm.fit(freqs_filtered, average_psd_filtered, progress="tqdm", n_jobs=20)
toc = time.time()
print(f"Fitted FOOOF in {toc - tic} seconds")

# Get knee parameter
ints = fm.get_params("aperiodic_params", "knee")

toc = time.time() - tic
print(f"Elapsed time: {toc}")

# Save results
fname = pathjoin(subj_preprocpath, "rest", f"{i_subj}_source_fooof.pkl")
results = {
    "ints": ints,
    "i_subj": i_subj,
    "fm": fm,
}
pklsave(fname, results)

print("DONE")
