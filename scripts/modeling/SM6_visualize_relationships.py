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

figpath = "/BICNAS2/ycatal/erf_acw/figures/figs/model_f2"

def p2str(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


f = h5py.File("/BICNAS2/ycatal/erf_acw/scripts/model/rest_extended.jld2", "r")
acw50s = f.get("acw50s")[...]
gamma1 = f.get("gamma_1")[...]
f.close()

f = h5py.File("/BICNAS2/ycatal/erf_acw/scripts/model/task_extended.jld2", "r")
erfs = f.get("erfs")[...]
gamma1 = f.get("gamma_1")[...]
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

pp = sns.pairplot(data, hue = "Color:\n$\\gamma_1$", corner = True, kind="scatter", plot_kws={'legend': False})
pp.figure.set_size_inches(10,9.5)
corr_results = pg.pairwise_corr(data, padjust="fdr_bh")

axes2reg = [pp.axes[1, 0], pp.axes[2, 0], pp.axes[2, 1]]
xs = ["ACW", "ACW", "ERF"]
ys = ["ERF"]
rho = [corr_results["r"][0], corr_results["r"][1], corr_results["r"][2]]
p = [p2str(corr_results["p-corr"][0]), p2str(corr_results["p-corr"][1]), p2str(corr_results["p-corr"][2])]
x_coordinates = [0.0, 0.5, 0.5]
y_coordinates = [0.9, 0.9, 0.9]
i = 0
for ax, x, y in zip(axes2reg, xs, ys):
    sns.regplot(ax=ax, data=data, x=x, y=y, order=1, scatter=False, line_kws={"color": "black"})
    ax.annotate(f"$\\rho$ = {rho[i]:.3f}{p[i]}", (x_coordinates[i], y_coordinates[i]), xycoords="axes fraction")
    ax.tick_params(axis="x", rotation=45)
    i += 1

fig = pp._figure
fig.savefig(join(figpath, "variable_corrs_extended.jpg"), dpi=800)

################## Same as above but as a row
data = pd.DataFrame(
    {
        "ACW": acw50s.ravel(), 
        "ERF": erfs.ravel(), 
        "$\\gamma_1$": np.tile(gamma1, [nsim, 1]).T.ravel(), 
        "Color:\n$\\gamma_1$": np.tile(gamma1_cat, [nsim, 1]).T.ravel(), 
    }
)

plot = (
    so.Plot(data)
    .layout(size=(12,4))
    .pair(x = ["ACW", "ERF", "ACW"], y=["ERF"], wrap=3, cross=False)
    .add(so.Dot(), color="Color:\n$\\gamma_1$")
    .add(so.Line(color="black"), so.PolyFit(order=1))
    .save(join(figpath, "variable_corrs_row_extended.jpg"), dpi=800)
)

corr_results = pg.pairwise_corr(data, padjust="fdr_bh")
# ACW - ERF: 0
# ACW - ERF: r = 0.464***

################## Plot gamma / variables ####################
data = pd.DataFrame(
    {
        "ACW": acw50s.ravel(), 
        "ERF": erfs.ravel(), 
        "$\\gamma_1$": np.tile(gamma1, [nsim, 1]).T.ravel(), 
        "Color:\n$\\gamma_1$": np.tile(gamma1_cat, [nsim, 1]).T.ravel(), 
    }
)

(
    so.Plot(data, x = "$\\gamma_1$")
    .layout(size=(12,4))
    .pair(y=["ACW", "ERF"], wrap=1)
    .add(so.Dot(), color="Color:\n$\\gamma_1$")
    .add(so.Line(color="black"), so.PolyFit(order=1))
    .save(join(figpath, "gamma_corrs_extended.jpg"), dpi=800)
)

# ACW - gamma1: 0.554***
# ERF - gamma1: 0.819***
# n = 420 in all correlations