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
    "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/sensitivity_results_pythonic.jld2",
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
ax[0, 0].set_ylim((-0.1, 1.1))


x1 = [1, 4, 7, 10]  # roi 1
x2 = [2, 5, 8, 11]  # roi 2
y1 = task_s1[:, 0]
y2 = task_s1[:, 1]
y1_ci = task_s1_ci[:, 0]
y2_ci = task_s1_ci[:, 1]

ax[0, 1].bar(x1, y1)
ax[0, 1].errorbar(x1, y1, yerr=y1_ci, fmt="o", color="k")
ax[0, 1].bar(x2, y2)
ax[0, 1].errorbar(x2, y2, yerr=y2_ci, fmt="o", color="k")
ax[0, 1].set_xticks([], labels=[])
ax[0, 1].set_title("ERF")
ax[0, 1].set_yticks(np.arange(0, 1.2, 0.2))
ax[0, 1].set_ylim((-0.1, 1.1))

################### S total
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
ax[1, 0].set_xticks([i + 0.5 for i in x1], labels=param[0:4])
ax[1, 0].set_ylabel("$S_T$")
ax[1, 0].set_yticks(np.arange(0, 1.2, 0.2))
ax[1, 0].set_ylim((-0.1, 1.1))
ax[1, 0].grid(False, axis="x")

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
ax[1, 1].set_xticks([i + 0.5 for i in x1], labels=param[0:4])
ax[1, 1].set_yticks(np.arange(0, 1.2, 0.2))
ax[1, 1].set_ylim((-0.1, 1.1))
ax[1, 1].grid(False, axis="x")

[
    ax[i, j].spines[["right", "top"]].set_visible(False)
    for i in range(2)
    for j in range(2)
]

f.legend()
f.savefig(join(figpath, "sensitivity_scores.png"), dpi=800, transparent=True)


##########
data = np.concatenate(
    [
        task_s1[:, 0],
        task_s1[:, 1],
        task_s1[:, 2],
        task_s1[:, 3],
        task_s1[:, 4],
        task_s1[:, 5],
        task_st[:, 0],
        task_st[:, 1],
        task_st[:, 2],
        task_st[:, 3],
        task_st[:, 4],
        task_st[:, 5],
    ]
)
outcomes = np.concatenate(
    [
        taskvars[0].repeat(5),
        taskvars[1].repeat(5),
        taskvars[2].repeat(5),
        taskvars[3].repeat(5),
        taskvars[4].repeat(5),
        taskvars[5].repeat(5),
        taskvars[0].repeat(5),
        taskvars[1].repeat(5),
        taskvars[2].repeat(5),
        taskvars[3].repeat(5),
        taskvars[4].repeat(5),
        taskvars[5].repeat(5),
    ]
)

s1_or_st = np.concatenate([np.tile("$S_1$", (30)), np.tile("$S_T$", (30))])
parameters = np.tile(param, (12))

df = pd.DataFrame(
    {
        "Sensitivity": data,
        "Outcome": outcomes,
        "Sobol Index": s1_or_st,
        "Parameter": parameters,
    }
)
df_filtered = df[
    (df["Outcome"] != "ACW (Region 1)") & (df["Outcome"] != "ACW (Region 2)")
]
plot = sns.catplot(
    df_filtered,
    kind="bar",
    x="Parameter",
    y="Sensitivity",
    hue="Sobol Index",
    errorbar=None,
    col="Outcome",
    col_wrap=2,
)
plot._figure.savefig(join(figpath, "task_sensitivity.jpg"), dpi=800)
