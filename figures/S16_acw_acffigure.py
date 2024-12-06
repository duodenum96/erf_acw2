import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from scipy import stats
from erf_acw2.src import pklload
from os.path import join

loadname = f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl"
figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure4"
acfs = pklload(loadname)["acfs"]

lags = np.arange(2000) / 1200
mean_acf = np.mean(acfs, axis=(0, 1))

plt.rcParams.update({"font.size": 16})
f, ax = plt.subplots(figsize=(8,8))
ax.plot(lags, mean_acf.T, alpha=0.5)
ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel("Lags (s)")
ax.set_ylabel("Autocorrelation")
ax.set_xlim(0, 0.25)

f.savefig(join(figpath, "acf_mean.png"), dpi=800, transparent=True)
###### For gimmick
mean_1d_acf = np.nanmean(mean_acf, axis=0)
f, ax = plt.subplots(
    
)
ax.plot(lags, mean_1d_acf, alpha=1.0, linewidth=4, color="k")
ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel("Lags")
ax.set_ylabel("Autocorrelation")
ax.set_xlim(0, 0.25)
# hide x and y ticks
ax.set_xticks([])
ax.set_yticks([])


f.savefig(join(figpath, "acf_mean_1d.png"), dpi=800, transparent=True)
