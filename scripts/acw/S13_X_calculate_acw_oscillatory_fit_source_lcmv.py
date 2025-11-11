import numpy as np
import mne
import sys
import os
from os.path import join as pathjoin
import time
from scipy.optimize import curve_fit

os.chdir("/BICNAS2/ycatal/erf_acw2/")
from erf_acw2.src import (
    get_commonsubj,
    pklsave,
    pklload,
    import_exampleraw,
    acf_oscillatory_function
)

subjlist = get_commonsubj()

outputpath = "/BICNAS2/ycatal/erf_acw2/results/int"

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
example_data = import_exampleraw()
fs = example_data.info["sfreq"]

example_source_path = pathjoin(preprocpath, subjlist[0], "rest", "source_lcmv.pkl")
example_source_data = pklload(example_source_path)
example_labels = np.array(example_source_data["labels"])

nlags = int(0.25 * fs)
n_rois = 185

popt_all = np.zeros((len(subjlist), n_rois, 3))
acws_all = np.zeros((len(subjlist), n_rois))
popt_all[:] = np.nan
acws_all[:] = np.nan

for i, i_subj in enumerate(subjlist):
    subj_preprocpath = pathjoin(preprocpath, i_subj)
    source_path = pathjoin(subj_preprocpath, "rest", "source_lcmv.pkl")
    source_data = pklload(source_path)
    subject_labels = np.array(source_data["labels"])

    acfs = source_data["acfs"][:, :nlags] # (n_rois, n_lags)
    lags = np.arange(0.0, nlags / fs, 1.0 / fs)
    for j, j_label in enumerate(subject_labels):
        if j_label in example_labels:
            ref_idx = np.where(example_labels == j_label)[0][0]
            acfunc = acfs[j, :]
            acw_euler = np.argmax(acfunc <= (1 / np.e)) / fs
            acws_all[i, ref_idx] = acw_euler
            try:
                popt, pcov = curve_fit(acf_oscillatory_function, lags, acfunc, p0=[0.9, acw_euler, 10.0])
                popt_all[i, ref_idx, :] = popt
            except RuntimeError:
                # If curve fitting fails, leave as NaN
                print(f"Warning: Curve fitting failed for subject {i_subj}, ROI {j_label}")
                pass
        else:
            print(f"Warning: Label {j_label} from subject {i_subj} not found in reference labels")
    print(f"{i+1} / {len(subjlist)}")

pklsave(pathjoin(outputpath, "source_lcmv_acw_oscillatory_fit.pkl"), {"popt_all": popt_all, "acws_all": acws_all})
print("DONE")

