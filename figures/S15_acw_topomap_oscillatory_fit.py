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
loadname = f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int_oscillatory_fit.pkl"
acw = pklload(loadname)["rest_acws"]

##################################################################################

nchan = 272
# fig = plt.figure(layout="constrained", figsize=(7.20472,7.87402))
fig, ax = plt.subplots()
restmean = np.nanmean(acw, axis=(0, 2))
# colorbar stuff
fraction=0.05
pad=0.04
# Draw the resting state
im0, cm0 = plot_topomap(
    restmean, raw.info, axes=ax, res=500, size=15, cmap=colormaps["cool"]
)
cb = fig.colorbar(im0, fraction=fraction, pad=pad)
cb.set_label("$ \\tau $", rotation=270, labelpad=15)

fig.savefig(pathjoin(figpath, "acw_topomap_oscillatory_fit.jpg"), dpi=800)
print("done")
