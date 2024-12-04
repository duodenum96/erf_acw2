import bambi as bmb
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import arviz.labels as azl
import pingouin as pg
import os
import cloudpickle
from erf_acw2.src import p_direction, rope, plot_channel_effects, create_channel_mask
import arviz as az
from erf_acw2.src_importdata import get_template_bad
import mne
import matplotlib as mpl

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

resultpath = "/BICNAS2/ycatal/erf_acw2/results/hierarchical_model"
figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/supplementary_erf_rt"
figpath_supp = "/BICNAS2/ycatal/erf_acw2/figures/figs/supplementary_erf_rt/supplementary"

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
data2["ERF"] = zscore(data2["ERF"])
data2["RT"] = zscore(data2["RT"])
data3 = data2.copy()
data3["Channel"] = [str(i) for i in data2["Channel"]]
data3 = data3.dropna()

model_formula = (
    "RT ~ 1 + ERF + (1|Cluster) + (1|Channel) + (1|Trial)"
    " + (ERF|Cluster) + (ERF|Channel) + (ERF|Trial)"
)
model_priors = {
    "sigma": bmb.Prior("Exponential", lam=1),
    "ERF": bmb.Prior("Normal", mu=0, sigma=1),
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    "1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "nu": bmb.Prior("Exponential", lam=1),
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

results = az.from_netcdf(join(resultpath, "idata_rt_erf.nc"))
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

# Create combined effect variables
results.posterior["ERF + ERF|Channel"] = (
    results.posterior["ERF"] + results.posterior["ERF|Channel"]
)
results.posterior["ERF + ERF|Trial"] = (
    results.posterior["ERF"] + results.posterior["ERF|Trial"]
)
results.posterior["ERF + ERF|Cluster"] = (
    results.posterior["ERF"] + results.posterior["ERF|Cluster"]
)

summary = az.summary(results, hdi_prob=0.94)

# Add columns for pd and rope
varnames_for_pd_rope = [
    "ERF",
    "1|Channel",
    "1|Trial",
    "1|Cluster",
    "ERF|Channel",
    "ERF|Trial",
    "ERF|Cluster",
    "ERF + ERF|Channel",
    "ERF + ERF|Trial",
    "ERF + ERF|Cluster",
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


summary.to_csv(join(figpath_supp, "summary_rt_erf.csv"))

# Prior and posterior predictive check
prior_results = model.prior_predictive()
model.predict(results, kind="response")

f, ax = plt.subplots(2, 1, figsize=(7, 10), dpi=300)
az.plot_ppc(prior_results, ax=ax[0], num_pp_samples=100, group="prior")
ax[0].set_xlim(-15, 15)
ax[0].set_title("Prior predictive check")
ax[0].set_xlabel("")

az.plot_ppc(results, ax=ax[1], num_pp_samples=100, group="posterior")
ax[1].set_xlim(-5, 5)
ax[1].set_title("Posterior predictive check")
ax[1].set_xlabel("RT")
f.savefig(join(figpath_supp, "prior_and_posterior_rt_erf.png"), dpi=300)

# Trace plots
var_names = [
    "Intercept",
    "ERF",
    "nu",
    "sigma",
    "1|Trial",
    "1|Cluster",
    "1|Channel",
    "1|Trial_sigma",
    "1|Cluster_sigma",
    "1|Channel_sigma",
    "ERF|Trial",
    "ERF|Cluster",
    "ERF|Channel",
    "ERF|Trial_sigma",
    "ERF|Cluster_sigma",
    "ERF|Channel_sigma",
]
plt.close()
az.plot_trace(
    results,
    combined=True,
    compact=True,
    figsize=(12, 58),
    var_names=var_names,
)

plt.savefig(join(figpath_supp, "trace_rt_erf.png"), dpi=300)

# Forest plot of relevant parameters
plt.rcParams.update({"font.size": 24})
var_names_forest = ["ERF", "ERF|Trial", "ERF|Cluster", "ERF + ERF|Trial", "ERF + ERF|Cluster"]

f, ax = plt.subplots(1, 1, figsize=(22, 12), dpi=300)
az.plot_forest(results, 
            var_names=var_names_forest, combined=True, ax=ax, rope=[-0.1, 0.1], colors="black")
ax.set_title("RT ~ ERF + ERF|Trial + ERF|Cluster + ERF|Channel")
plt.xticks(fontsize=24)
ax.axvline(0, color="black", linestyle="--")
f.savefig(join(figpath_supp, "forest_rt_erf.png"), dpi=300)

# Topoplot of relevant parameters
template_good = get_template_bad(good=True)
chans = data.dropna().Channel.unique()
chan_order = np.argsort(chans)

info = template_good.pick(chans).info
channel_coords = data.groupby("Channel")[["xcoords", "ycoords"]].mean()
coords = channel_coords.values

plt.rcParams.update({"font.size": 24})

mask_erf = create_channel_mask(summary, "ERF|Channel")
data_topo = results.posterior["ERF|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="ERF|Channel", hdi_prob=0.94)["ERF|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="ERF|Channel",
    figpath=join(figpath, "channels_erf_rt.jpg"),
    mask=mask_erf
)

mask_erf_plus = create_channel_mask(summary, "ERF + ERF|Channel")
data_topo = results.posterior["ERF + ERF|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="ERF + ERF|Channel", hdi_prob=0.94)["ERF + ERF|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="ERF + ERF|Channel",
    figpath=join(figpath, "channels_erf_rt_plus.jpg"),
    mask=mask_erf_plus
)

