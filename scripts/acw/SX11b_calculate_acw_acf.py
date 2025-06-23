import os
import numpy as np
import time
import sys
import pandas as pd

os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from scipy import signal
import matplotlib.pyplot as plt
from erf_acw2.src import pklload, pklsave, pick_megchans, re_epoch, get_commonsubj, pathjoin, import_exampleraw

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
resultspath = "/BICNAS2/ycatal/erf_acw2/results/source"

epochs = import_exampleraw()
fs = epochs.info["sfreq"]
n_vertices = 5124

subjlist = get_commonsubj()
nsubj = len(subjlist)

all_ints = np.zeros((nsubj, n_vertices))

subjs = []
vertices = []
ints_list = []

for i, i_subj in enumerate(subjlist):
    subj_raw_path = pathjoin(data_root, i_subj, "ses-01", "meg")

    preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
    subj_preprocpath = pathjoin(preprocpath, i_subj)

    tic = time.time()

    # Load source ACFs
    source_acfs = pklload(pathjoin(subj_preprocpath, "rest", "source_acfs.pkl"))
    acfs = source_acfs["acfs"]
    lags = np.arange(0.0, acfs.shape[1]) / fs

    toc = time.time()

    ints = np.argmax(acfs < 0.5, axis=1) / fs
    all_ints[i, :] = ints

    for i_source in range(acfs.shape[0]):
        subjs.append(i_subj)
        vertices.append(i_source)
        ints_list.append(ints[i_source])

    toc = time.time() - tic
    print(f"Elapsed time: {toc}")

# Save results
fname = pathjoin(resultspath, f"source_acws_acf.pkl")
pklsave(fname, all_ints)

fname = pathjoin(resultspath, f"source_acws_acf.csv")
df = pd.DataFrame({"subjs": subjs, "vertices": vertices, "ints": ints_list})
df.to_csv(fname, index=False)

print("DONE")
