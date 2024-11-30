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
import pymc as pm
import arviz as az
import bambi as bmb
from scipy.stats import boxcox
import bambi as bmb
import arviz.labels as azl
sns.set_theme()
sns.set_theme(style="whitegrid")
so.Plot.config.theme.update(axes_style("whitegrid"))

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_f2"

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

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
nsim = erfs.shape[1]

vars = ["acw50s", "erfs"]

data = pd.DataFrame(
    {
        "ACW": zscore(acw50s.ravel()), 
        "ERF": zscore(erfs.ravel()), 
        "gamma_1": zscore(gamma1), 
        "Color:\n$\\gamma_1$": np.tile(gamma1_cat, [nsim, 1]).T.ravel(), 
    }
)

# Model 1: 
# ERF ~ ACW
# Model 2:
# ERF ~ ACW + gamma1

a2e_model = bmb.Model("ERF ~ ACW", data, family="t")
tr_a2e = a2e_model.fit()
a2e_g_model = bmb.Model("ERF ~ ACW + gamma_1", data, family="t")
tr_a2e_g = a2e_g_model.fit()

all_traces = [tr_a2e, tr_a2e_g]
all_model_names = ["ERF ~ ACW", "ERF ~ ACW + gamma_1"]
all_model_names_map = all_model_names.copy()
all_model_names_map[1] = "ERF ~ ACW + $\\gamma_1$"
labeller = azl.MapLabeller(var_name_map={"gamma_1": "$\\gamma_1$"}, 
                           model_name_map={
                               all_model_names[i]: all_model_names_map[i] for i in range(len(all_model_names))
                               })
az.plot_forest(all_traces,  figsize=(10, 5), model_names=all_model_names, 
               combined=True, hdi_prob=0.94, var_names=["~sigma", "~Intercept"],
               labeller=labeller)

# Increase fontsize
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 24
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
# Plot model 1: ERF ~ ACW
az.plot_forest([tr_a2e], 
               model_names=["ERF ~ ACW"], 
               combined=True, hdi_prob=0.94, var_names=["~sigma", "~Intercept", "~nu"],
               ax=ax1, colors="k", rope=[-0.1, 0.1])
ax1.set_yticklabels(["ACW"])
ax1.set_title("Model 1: mERF ~ ACW")
ax1.axvline(0, color="black", linestyle="--", linewidth=1)
ax1.set_xlim(-0.12, 0.95)

# Plot model 2: ERF ~ ACW + gamma1 
az.plot_forest([tr_a2e_g], 
               model_names=["ERF ~ ACW + $\\gamma_1$"], 
               combined=True, hdi_prob=0.94, var_names=["~sigma", "~Intercept", "~nu"],
               ax=ax2, colors="k", rope=[-0.1, 0.1])
ax2.set_yticklabels(["$\\gamma_1$", "ACW"]) 
ax2.set_title("Model 2: mERF ~ ACW + $\\gamma_1$")
ax2.axvline(0, color="black", linestyle="--", linewidth=1)
ax2.set_xticks(np.arange(-0.1, 1.1, 0.1))
ax2.set_xlim(-0.12, 0.95)

plt.tight_layout()
plt.savefig(join(figpath, "forests_linregress_a2e_combined.jpg"), dpi=800)
plt.close()
