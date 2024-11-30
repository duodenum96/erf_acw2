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

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_f2"


def p2str(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


f = h5py.File("/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/rest.jld2", "r")
acw50s = f.get("acw50s")[...]
gamma1 = f.get("gamma_1_values")[...]
f.close()

f = h5py.File("/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/task.jld2", "r")
erfs = f.get("erfs")[...]
gamma1 = f.get("gamma_1_values")[...]
gamma1_cat = gamma1.astype(str)
f.close()

n_gamma = gamma1.shape[0]
nsim = acw50s.shape[1]

vars = ["acw50s", "erfs"]
data = pd.DataFrame(
    {
        "ACW": acw50s.ravel(),
        "Color:\n$\\gamma_1$": np.tile(gamma1.astype(str), [nsim, 1]).T.ravel(),
        "ERF": erfs.ravel(),
    }
)

corr_results = pg.pairwise_corr(data, padjust="fdr_bh")

xs = ["ACW"]
ys = ["ERF"]
rho = corr_results["r"][0]
p = p2str(corr_results["p-unc"][0])

plot = (
    so.Plot(data)
    .layout(size=(4, 4))
    .pair(x=["ACW"], y=["ERF"])
    .add(so.Dot(), color="Color:\n$\\gamma_1$")
    .add(so.Line(color="black"), so.PolyFit(order=1))
    .label(x="ACW (s)", y="mERF")
    .theme({"axes.labelsize": 16})
    .save(join(figpath, "variable_corrs.jpg"), dpi=800)
)

corr_results = pg.pairwise_corr(data, padjust="fdr_bh")
# ACW - ERF: r = 0.517***

################## Plot gamma / variables ####################
data = pd.DataFrame(
    {
        "ACW": acw50s.ravel(),
        "ERF": erfs.ravel(),
        "$\\gamma_1$": np.tile(gamma1, [nsim, 1]).T.ravel(),
        "Color:\n$\\gamma_1$": np.tile(gamma1_cat, [nsim, 1]).T.ravel(),
    }
)

data = data.rename(columns={"ACW": "ACW (s)", "ERF": "mERF"})

(
    so.Plot(data, x="$\\gamma_1$")
    .layout(size=(8, 4))
    .pair(y=["ACW (s)", "mERF"], wrap=1)
    .add(so.Dot(), color="Color:\n$\\gamma_1$")
    .add(so.Line(color="black"), so.PolyFit(order=1))
    .theme({"axes.labelsize": 16})
    .save(join(figpath, "gamma_corrs.jpg"), dpi=800)
)

pg.pairwise_corr(data, [["$\gamma_1$"], ["ACW (s)", "mERF"]], padjust="fdr_bh")

# ACW - gamma1: 0.578***
# ERF - gamma1: 0.904***
# n = 1240 in all correlations

######## Now, we'll do a 1 x 3 plot. In columns, we'll have the 3 different gamma1 values, and in rows, we'll have the scatterplot between 2 variables.
example_gamma1s = [40, 50, 60]
data_example = data[data["$\\gamma_1$"].isin(example_gamma1s)]

(
    so.Plot(data_example, x="ACW (s)", y="mERF", color="Color:\n$\\gamma_1$")
    .facet(col="$\\gamma_1$")
    .add(so.Dot(color="black"))
    .add(so.Line(color="black"), so.PolyFit(order=1))
    .label(title="$\\gamma_1$ = {}".format)
    .save(join(figpath, "gamma1_scatterplots.jpg"), dpi=800)
)

# To calculate correlations, make each gamma1 value a seperate column, then put it into pg.pairwise_corr
data_example_corr = []
for gamma in example_gamma1s:
    subset = data_example[data_example["$\\gamma_1$"] == gamma]
    corr = pg.corr(subset["ACW (s)"], subset["mERF"])
    data_example_corr.append({
        'gamma1': gamma,
        'r': corr['r'].iloc[0],
        'p-val': corr['p-val'].iloc[0]
    })

data_example_corr = pd.DataFrame(data_example_corr)
# Do correction for multiple comparisons
data_example_corr["p-cor"] = pg.multicomp(data_example_corr["p-val"], alpha=0.05, method="fdr_bh")[1]
print(data_example_corr)
# Correlations:
# gamma1 = 40: r = -0.0.058, p = 0.719
# gamma1 = 50: r = 0.096, p = 0.719
# gamma1 = 60: r = -0.278, p = 0.248
