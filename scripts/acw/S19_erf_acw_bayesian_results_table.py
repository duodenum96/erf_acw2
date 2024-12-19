import pymc as pm
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
from erf_acw2.src_importdata import get_template_bad

def zscore(x):
    return (x - np.mean(x)) / np.std(x)


def cloud_pklsave(filename, obj):
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)


def cloud_pklload(fname):
    with open(fname, "rb") as f:
        return cloudpickle.load(f)


resultpath = "/BICNAS2/ycatal/erf_acw2/results/hierarchical_model"
figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure5"
figpath_supp = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure5/supplementary"


data = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")
data = data.rename(
    columns={
        "erfs": "mERF",
        "restacws": "ACW",
        "rts": "RT",
        "clusters": "Cluster",
        "subjects": "Subject",
        "channels": "Channel",
        "erftype": "Trial",
    }
)
data2 = data.copy()
data2["mERF"] = zscore(data2["mERF"])
data2["ACW"] = zscore(data2["ACW"])
data2["RT"] = zscore(data2["RT"])
data3 = data2.copy()
data3["Channel"] = [str(i) for i in data2["Channel"]]
data3 = data3.dropna()


model_formula = (
    "ACW ~ 1 + mERF + (1|Cluster) + (1|Channel) + (1|Trial)"
    " + (mERF|Cluster) + (mERF|Channel) + (mERF|Trial)"
)

model_priors = {
    "sigma": bmb.Prior("Exponential", lam=1),
    "mERF": bmb.Prior("Normal", mu=0, sigma=1),
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    "1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "mERF|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "mERF|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "mERF|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "nu": bmb.Prior("Exponential", lam=1),
}

cluster_names = [
    "Enc #1",
    "Enc #2",
    "Enc #3",
    "Enc #4",
    "Prb #1",
    "Prb #2",
]
trial_names = ["efh", "efs", "es", "pfh", "pfs", "ps"]
unique_channels = data3["Channel"].unique()
coordinates_dict = {
    "Cluster": cluster_names,
    "Trial": trial_names,
    "Channel": unique_channels,
}

model = bmb.Model(
    model_formula,
    data3,
    priors=model_priors,
    family="t",
    noncentered=True,
    categorical=["Channel", "Trial", "Cluster"],
)
model.build()

results = az.from_netcdf(join(resultpath, "idata_acw_erf.nc"))
results = results.rename({"ERF": "mERF", "ERF|Channel": "mERF|Channel", "ERF|Trial": "mERF|Trial", "ERF|Cluster": "mERF|Cluster", 
                          "ERF|Trial_sigma": "mERF|Trial_sigma", "ERF|Cluster_sigma": "mERF|Cluster_sigma", "ERF|Channel_sigma": "mERF|Channel_sigma"})
results = results.assign_coords(
    {
        "Cluster": cluster_names,
        "Trial": trial_names,
        "Channel": unique_channels,
        "Cluster__factor_dim": cluster_names,
        "Trial__factor_dim": trial_names,
        "Channel__factor_dim": unique_channels,
    }
)
# Create new variables ERF + ERF|Channel, ERF + ERF|Trial, ERF + ERF|Cluster
results.posterior["mERF + mERF|Channel"] = (
    results.posterior["mERF"] + results.posterior["mERF|Channel"]
)
results.posterior["mERF + mERF|Trial"] = (
    results.posterior["mERF"] + results.posterior["mERF|Trial"]
)
results.posterior["mERF + mERF|Cluster"] = (
    results.posterior["mERF"] + results.posterior["mERF|Cluster"]
)
summary = az.summary(results, hdi_prob=0.94)
# Add columns for pd and rope
varnames_for_pd_rope = [
    "mERF",
    "1|Channel",
    "1|Trial",
    "1|Cluster",
    "mERF|Channel",
    "mERF|Trial",
    "mERF|Cluster",
    "mERF + mERF|Channel",
    "mERF + mERF|Trial",
    "mERF + mERF|Cluster",
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

summary.to_csv(join(figpath_supp, "summary_acw_erf.csv"))
# Do prior and posterior predictive check
prior_results = model.prior_predictive()
model.predict(results, kind="response")

plt.rcParams.update({"font.size": 12})
f, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
az.plot_ppc(prior_results, ax=ax[0, 0], num_pp_samples=20, group="prior")
ax[0, 0].set_xlim(-15, 15)
ax[0, 0].set_title("Prior predictive check")
ax[0, 0].set_xlabel("")

az.plot_ppc(prior_results, ax=ax[0, 1], num_pp_samples=20, group="prior", kind="cumulative")
ax[0, 1].set_xlim(-15, 15)
ax[0, 1].set_title("Prior predictive check")
ax[0, 1].set_xlabel("")

# Posterior predictive check
az.plot_ppc(results, ax=ax[1, 0], num_pp_samples=20, group="posterior")
ax[1, 0].set_xlim(-5, 5)
ax[1, 0].set_title("Posterior predictive check")
ax[1, 0].set_xlabel("ACW")

az.plot_ppc(results, ax=ax[1, 1], num_pp_samples=20, group="posterior", kind="cumulative")
ax[1, 1].set_xlim(-5, 5)
ax[1, 1].set_title("Posterior predictive check")
ax[1, 1].set_xlabel("ACW")

f.savefig(join(figpath_supp, "prior_and_posterior_acw_erf.png"), dpi=300)

var_names = [
    "Intercept",
    "mERF",
    "nu",
    "sigma",
    "1|Trial",
    "1|Cluster",
    "1|Channel",
    "1|Trial_sigma",
    "1|Cluster_sigma",
    "1|Channel_sigma",
    "mERF|Trial",
    "mERF|Cluster",
    "mERF|Channel",
    "mERF|Trial_sigma",
    "mERF|Cluster_sigma",
    "mERF|Channel_sigma",
]

# Trace plots
plt.close()
az.plot_trace(
    results,
    combined=True,
    compact=True,
    figsize=(12, 58),
    var_names=var_names,
)
plt.savefig(join(figpath_supp, "trace_acw_erf.png"), dpi=300)

# Forest plots of relevant parameters
plt.rcParams.update({"font.size": 24})
var_names_forest = ["mERF", "mERF|Trial", "mERF|Cluster", "mERF + mERF|Trial", "mERF + mERF|Cluster"]
f, ax = plt.subplots(1, 1, figsize=(25, 12), dpi=300)
az.plot_forest(results, 
            var_names=var_names_forest, combined=True, ax=ax, rope=[-0.1, 0.1], colors="black")
ax.set_title("ACW ~ mERF + mERF|Trial + mERF|Cluster + mERF|Channel")
# plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
ax.axvline(0, color="black", linestyle="--")
f.savefig(join(figpath_supp, "forest_acw_erf.png"), dpi=300)

###### Topoplot of Channel Specific Effects #######

template_good = get_template_bad(good=True)
chans = data.dropna().Channel.unique()
chan_order = np.argsort(chans)

info = template_good.pick(chans).info
channel_coords = data.groupby("Channel")[["xcoords", "ycoords"]].mean()
coords = channel_coords.values

plt.rcParams.update({"font.size": 24})

mask_erf = create_channel_mask(summary, "mERF|Channel")
data_topo = results.posterior["mERF|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="mERF|Channel", hdi_prob=0.94)["mERF|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="mERF|Channel",
    figpath=join(figpath, "channels_erf_acw.jpg"),
    mask=mask_erf
)

mask_erf_plus = create_channel_mask(summary, "mERF + mERF|Channel")
data_topo = results.posterior["mERF + mERF|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="mERF + mERF|Channel", hdi_prob=0.94)["mERF + mERF|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

# Maybe needed just in case for future
# midpoint = 1 - vmax / (vmax + abs(vmin))
# cmap = shiftedColorMap(mpl.cm.PiYG, midpoint=midpoint)

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="mERF + mERF|Channel",
    figpath=join(figpath, "channels_erf_acw_plus.jpg"),
    mask=mask_erf_plus
)

