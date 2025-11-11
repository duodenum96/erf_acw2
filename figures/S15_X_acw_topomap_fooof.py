from mne.viz import plot_topomap
import numpy as np
from erf_acw2.src import pklload, import_exampleraw, create_blockacws
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os
import sys
from os.path import join as pathjoin

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure4"
raw = import_exampleraw()

################## Import data ##################
task = "haririhammer"
loadname = f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_ints.pkl"
acw = 1 / (2*np.pi * pklload(loadname)["rest_ints"]) # Channels x subjects

################## Clean data: remove negatives and outliers ##################
# Remove negative values
acw_positive = acw.copy()
acw_positive[acw_positive <= 0] = np.nan

q1 = np.percentile(acw[~np.isnan(acw)], 90)
acw_positive[acw_positive > q1] = np.nan

##################################################################################

nchan = 272
# fig = plt.figure(layout="constrained", figsize=(7.20472,7.87402))
fig, ax = plt.subplots()
restmean = np.nanmedian(acw_positive, axis=1)
# colorbar stuff
fraction=0.05
pad=0.04
# Draw the resting state
im0, cm0 = plot_topomap(
    np.nanmean(acw, axis=1), raw.info, axes=ax, res=500, size=15, cmap=colormaps["cool"]
)
cb = fig.colorbar(im0, fraction=fraction, pad=pad)
cb.set_label("INT", rotation=270, labelpad=15)

fig.savefig(pathjoin(figpath, "acw_topomap_fooof_raw.jpg"), dpi=800)
print("done")
