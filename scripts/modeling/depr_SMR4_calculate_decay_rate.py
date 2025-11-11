import h5py
import seaborn as sns
import pandas as pd
import numpy as np
from seaborn import objects as so
from seaborn import axes_style
import matplotlib.pyplot as plt
import pingouin as pg
from os.path import join
import matplotlib as mpl
import os
from erf_acw2.src import (
    get_commonsubj,
    pklsave,
    pklload,
    import_exampleraw,
    acf_oscillatory_function,
    acf_decay_function
)
from scipy.optimize import curve_fit

sns.set_theme()
sns.set_theme(style="whitegrid")
so.Plot.config.theme.update(axes_style("whitegrid"))

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_reviewer_comments"
if not os.path.exists(figpath):
    os.makedirs(figpath)


def p2str(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


f = h5py.File(
    "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/acfs_rest_10rois.jld2",
    "r",
)
acfs = f.get("acfs")[...]
acw50s = f.get("acw50s")[...]
acw50s = np.nanmean(acw50s, axis=2)
f.close()

nsims = acfs.shape[0]
nrois = acfs.shape[1]
nlags = acfs.shape[2]
mean_acfs = np.nanmean(acfs, axis=1)

nlags = 500
fs = 1200

lags = np.arange(0.0, nlags / fs, 1.0 / fs)

popt_all = np.zeros((nsims, nrois, 1))
acws_all = np.zeros((nsims,nrois))
ints_all = np.zeros((nsims, nrois))
popt_all[:] = np.nan
acws_all[:] = np.nan
ints_all[:] = np.nan

for i in range(nsims):
    for j in range(nrois):
        acfunc = mean_acfs[i, :nlags, j]
        acw_euler = np.argmax(acfunc <= (1 / np.e)) / fs
        acws_all[i, j] = acw_euler

        if np.any(np.isnan(acfunc)):
            popt_all[i, j, :] = np.nan
            ints_all[i, j] = np.nan
            continue

        try:
            popt, pcov = curve_fit(acf_decay_function, lags, acfunc, p0=[acw_euler])
            popt_all[i, j, :] = popt
            ints_all[i, j] = popt[0]
        except RuntimeError:
            # If curve fitting fails, leave as NaN
            print(f"Warning: Curve fitting failed for subject {i}, ROI {j}")
            pass

pklsave(os.path.join(figpath, "decay_rate_data.pkl"), {"popt_all":popt_all, "acws_all":acws_all, "ints_all":ints_all})
