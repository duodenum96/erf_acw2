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

sns.set_theme()
sns.set_theme(style="whitegrid")
so.Plot.config.theme.update(axes_style("whitegrid"))

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_f1"

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
    "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/sensitivity/sensitivity_results_pythonic.jld2",
    "r",
)
rest_s1 = f.get("rest_s1")[...]
rest_st = f.get("rest_st")[...]
task_s1 = f.get("task_s1")[...]
task_st = f.get("task_st")[...]

rest_s1_ci = f.get("rest_s1_ci")[...]  # A_F, A_L, A_B, gamma1 XX S_1, S_T
rest_st_ci = f.get("rest_st_ci")[...]
task_s1_ci = f.get("task_s1_ci")[...]
task_st_ci = f.get("task_st_ci")[...]

restvars = f.get("restvars")[...]
taskvars = f.get("taskvars")[...]
param = f.get("param")[...]
f.close()

param = np.array([i.decode("UTF-8") for i in param])
param = np.array(["$A_F$", "$A_L$", "$A_B$", "$\\gamma_1$", "$\\gamma_4$"])
restvars = np.array([i.decode("UTF-8") for i in restvars])
taskvars = np.array([i.decode("UTF-8") for i in taskvars])

# Plot resting state S1 and Stotal
# Start with S1, then do ST
ylims = (-0.1, 1.2)
f, ax = plt.subplots(2, 2, figsize=(10, 5))

x1 = [1, 4, 7, 10]  # roi 1
x2 = [2, 5, 8, 11]  # roi 2
y1 = rest_s1[:, 0]
y2 = rest_s1[:, 1]
y1_ci = rest_s1_ci[:, 0]
y2_ci = rest_s1_ci[:, 1]

ax[0, 0].bar(x1, y1, label="Region 1")
ax[0, 0].errorbar(x1, y1, yerr=y1_ci, fmt="o", color="k")
ax[0, 0].bar(x2, y2, label="Region 2")
ax[0, 0].errorbar(x2, y2, yerr=y2_ci, fmt="o", color="k")
ax[0, 0].set_xticks([], labels=[])
ax[0, 0].set_ylabel("$S_1$")
ax[0, 0].set_title("ACW")
ax[0, 0].set_yticks(np.arange(0, 1.2, 0.2))
ax[0, 0].set_ylim(ylims)


x1 = [1, 4, 7, 10]  # roi 1
x2 = [2, 5, 8, 11]  # roi 2
y1 = task_s1[:, 0]
y2 = task_s1[:, 1]
y1_ci = task_s1_ci[:, 0]
y2_ci = task_s1_ci[:, 1]

ax[0, 1].bar(x1, y1, label="Region 1")
ax[0, 1].errorbar(x1, y1, yerr=y1_ci, fmt="o", color="k")
ax[0, 1].bar(x2, y2, label="Region 2")
ax[0, 1].errorbar(x2, y2, yerr=y2_ci, fmt="o", color="k")
ax[0, 1].set_xticks([], labels=[])
ax[0, 1].set_title("mERF")
ax[0, 1].set_yticks(np.arange(0, 1.2, 0.2))
ax[0, 1].set_ylim(ylims)
ax[0, 1].legend()

x1 = [1, 4, 7, 10]  # roi 1
x2 = [2, 5, 8, 11]  # roi 2
y1 = rest_st[:, 0]
y2 = rest_st[:, 1]
y1_ci = rest_st_ci[:, 0]
y2_ci = rest_st_ci[:, 1]
ax[1, 0].bar(x1, y1)
ax[1, 0].errorbar(x1, y1, yerr=y1_ci, fmt="o", color="k")
ax[1, 0].bar(x2, y2)
ax[1, 0].errorbar(x2, y2, yerr=y2_ci, fmt="o", color="k")
ax[1, 0].set_xticks([x1[i] + 0.5 for i in range(4)], labels=param[0:4])
ax[1, 0].set_ylabel("$S_T$")
ax[1, 0].set_yticks(np.arange(0, 1.2, 0.2))
ax[1, 0].set_ylim(ylims)
ax[1, 0].grid(axis="x")

x1 = [1, 4, 7, 10]  # roi 1
x2 = [2, 5, 8, 11]  # roi 2
y1 = task_st[:, 0]
y2 = task_st[:, 1]
y1_ci = task_st_ci[:, 0]
y2_ci = task_st_ci[:, 1]
ax[1, 1].bar(x1, y1)
ax[1, 1].errorbar(x1, y1, yerr=y1_ci, fmt="o", color="k")
ax[1, 1].bar(x2, y2)
ax[1, 1].errorbar(x2, y2, yerr=y2_ci, fmt="o", color="k")
ax[1, 1].set_xticks([x1[i] + 0.5 for i in range(4)], labels=param[0:4])
ax[1, 1].set_yticks(np.arange(0, 1.2, 0.2))
ax[1, 1].set_ylim(ylims)
ax[1, 1].grid(axis="x")

f.savefig(join(figpath, "sensitivity_scores.png"), dpi=300)

