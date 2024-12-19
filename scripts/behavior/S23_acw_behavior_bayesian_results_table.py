import bambi as bmb
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import pickle
import numpy as np
import arviz.labels as azl
import pingouin as pg
import os
import cloudpickle
from erf_acw2.src import p_direction, rope, plot_channel_effects, create_channel_mask
import arviz as az
from erf_acw2.src_importdata import get_template_bad, shiftedColorMap
import mne
import matplotlib as mpl

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

def cloud_pklsave(filename, obj):
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)

def cloud_pklload(fname):
    with open(fname, "rb") as f:
        return cloudpickle.load(f)

resultpath = "/BICNAS2/ycatal/erf_acw2/results/hierarchical_model"
figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/supplementary_acw_rt"
figpath_supp = "/BICNAS2/ycatal/erf_acw2/figures/figs/supplementary_acw_rt/supplementary"

# Load and prepare data
data = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")
data = data.rename(
    columns={
        "erfs": "ERF",
        "restacws": "ACW",
        "rts": "RT",
        "clusters": "Cluster",
        "subjects": "Subject",
        "channels": "Channel",
        "erftype": "Trial",
    }
)
data2 = data.copy()
data2["ACW"] = zscore(data2["ACW"])
data2["RT"] = zscore(data2["RT"])
data3 = data2.copy()
data3["Channel"] = [str(i) for i in data2["Channel"]]
data3 = data3.dropna()

# Simplified model formula without Cluster and Trial random effects
model_formula = "RT ~ 1 + ACW + (1|Channel) + (ACW|Channel)"

model_priors = {
    "sigma": bmb.Prior("Exponential", lam=1),
    "RT": bmb.Prior("Normal", mu=0, sigma=1),
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    "1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "RT|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "nu": bmb.Prior("Exponential", lam=1),
}

unique_channels = data3["Channel"].unique()
coordinates_dict = {
    "Channel": unique_channels,
}

model = bmb.Model(
    model_formula,
    data3,
    priors=model_priors,
    family="t",
    noncentered=True,
    categorical=["Channel"],
)
model.build()

# Load results
results = az.from_netcdf(join(resultpath, "idata_rt_acw.nc"))
results = results.assign_coords(
    {
        "Channel": unique_channels,
        "Channel__factor_dim": unique_channels,
    }
)

# Create combined effect variables
results.posterior["ACW + ACW|Channel"] = (
    results.posterior["ACW"] + results.posterior["ACW|Channel"]
)

summary = az.summary(results, hdi_prob=0.94)

# Add columns for pd and rope
varnames_for_pd_rope = [
    "ACW",
    "1|Channel",
    "ACW|Channel",
    "ACW + ACW|Channel",
]
summary["pd"] = np.nan
summary["rope"] = np.nan

for varname in varnames_for_pd_rope:
    print(varname)
    pd = p_direction(results, varname)
    r = rope(results, varname)
    if type(r) == np.float64:
        summary.loc[varname, "pd"] = pd
        summary.loc[varname, "rope"] = r
    else:
        for i in range(len(r)):
            effect, coordinate = varname.split("|")
            name = f"{varname}[{coordinates_dict[coordinate][i]}]"
            summary.loc[name, "pd"] = pd[i]
            summary.loc[name, "rope"] = r[i]

summary.to_csv(join(figpath_supp, "summary_acw_rt.csv"))

# Prior and posterior predictive checks
prior_results = model.prior_predictive()
model.predict(results, kind="response")

plt.rcParams.update({"font.size": 12})
f, ax = plt.subplots(2, 2, figsize=(7, 10), dpi=300)
az.plot_ppc(prior_results, ax=ax[0, 0], num_pp_samples=20, group="prior")
ax[0, 0].set_xlim(-15, 15)
ax[0, 0].set_title("Prior predictive check")
ax[0, 0].set_xlabel("")

az.plot_ppc(prior_results, ax=ax[0, 1], num_pp_samples=20, group="prior", kind="cumulative")
ax[0, 1].set_xlim(-15, 15)
ax[0, 1].set_title("Prior predictive check")
ax[0, 1].set_xlabel("")

az.plot_ppc(results, ax=ax[1, 0], num_pp_samples=20, group="posterior")
ax[1, 0].set_xlim(-5, 5)
ax[1, 0].set_title("Posterior predictive check")
ax[1, 0].set_xlabel("RT")

az.plot_ppc(results, ax=ax[1, 1], num_pp_samples=20, group="posterior", kind="cumulative")
ax[1, 1].set_xlim(-5, 5)
ax[1, 1].set_title("Posterior predictive check")
ax[1, 1].set_xlabel("RT")
f.savefig(join(figpath_supp, "prior_and_posterior_acw_rt.png"), dpi=300)

# Simplified var_names without Cluster and Trial effects
var_names = [
    "Intercept",
    "ACW",
    "nu",
    "sigma",
    "1|Channel",
    "1|Channel_sigma",
    "ACW|Channel",
    "ACW|Channel_sigma",
]

# Trace plots
plt.close()
az.plot_trace(
    results,
    combined=True,
    compact=True,
    figsize=(12, 40),
    var_names=var_names,
)
plt.savefig(join(figpath_supp, "trace_acw_rt.png"), dpi=300)

# Forest plots of relevant parameters
plt.rcParams.update({"font.size": 12})
var_names_forest = ["ACW"]
f, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=300)
az.plot_forest(results, 
            var_names=var_names_forest, combined=True, ax=ax, rope=[-0.1, 0.1], colors="black")
ax.set_title("ACW ~ ACW + ACW|Channel")
plt.xticks(fontsize=12)
ax.axvline(0, color="black", linestyle="--")
f.savefig(join(figpath_supp, "forest_acw_rt.png"), dpi=300)

###### Topoplot of Channel Specific Effects #######

template_good = get_template_bad(good=True)
chans = data.dropna().Channel.unique()
chan_order = np.argsort(chans)

info = template_good.pick(chans).info
channel_coords = data.groupby("Channel")[["xcoords", "ycoords"]].mean()
coords = channel_coords.values

plt.rcParams.update({"font.size": 24})

mask_rt = create_channel_mask(summary, "ACW|Channel")
data_topo = results.posterior["ACW|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="ACW|Channel", hdi_prob=0.94)["ACW|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="ACW|Channel",
    figpath=join(figpath, "channels_rt_acw.jpg"),
    mask=mask_rt
)

mask_rt_plus = create_channel_mask(summary, "ACW + ACW|Channel")
data_topo = results.posterior["ACW + ACW|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="ACW + ACW|Channel", hdi_prob=0.94)["ACW + ACW|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="ACW + ACW|Channel",
    figpath=join(figpath, "channels_rt_acw_plus.jpg"),
    mask=mask_rt_plus
)
