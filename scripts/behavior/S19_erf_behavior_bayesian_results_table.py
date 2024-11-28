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
from erf_acw2.src_importdata import get_template_bad
import mne

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
results = az.from_netcdf(join(resultpath, "erf_rt_hierarchical_take2.cdf"))
model = cloud_pklload(join(resultpath, "erf_rt_hierarchical_take2.pkl"))

with model:
    pm.sample_posterior_predictive(results, extend_inferencedata=True)

f, ax = plt.subplots(2)
az.plot_ppc(results, num_pp_samples=20, kind="kde", ax=ax[0], mean=False)
az.plot_ppc(results, num_pp_samples=20, kind="cumulative", ax=ax[1], mean=False)
ax[0].set_xlim((-3, 3))
ax[1].set_xlim((-3, 3))
ax[0].set_xlabel("Posterior (RT~ERF)")
ax[1].set_xlabel("Posterior (RT~ERF)")
f.savefig(join(figpath_supp, "posterior_predictive_erf_rt.jpg"), dpi=800)

# coords = {"beta_offset_cluster_erf": (["Encode Cluster #" + str(i) for i in range(1, 5)] +
#                        ["Probe Cluster #" + str(i) for i in range(1, 3)])}
labeller = azl.MapLabeller(
    var_name_map={
        "mu_beta_erf": r"$\beta$",
        "mu_alpha_erf": r"$\alpha$",
        "gamma_beta_cluster_t": r"$\gamma_{cluster}$",
        "gamma_beta_trial_t": r"$\gamma_{trial}$",
        # "gamma_beta_channel_t": r"$\gamma_{channel}$",
        "gamma_alpha_cluster_t": r"$\theta_{cluster}$",
        "gamma_alpha_trial_t": r"$\theta_{trial}$",
        # "gamma_alpha_channel_t": r"$\theta_{channel}$",
    }
)  # dim_map=coords)

results.posterior["gamma_alpha_cluster_t"] = (
    results.posterior["sigma_alpha_erf_cluster"]
    * results.posterior["alpha_offset_cluster_erf"]
)
results.posterior["gamma_alpha_trial_t"] = (
    results.posterior["sigma_alpha_erf_trial"]
    * results.posterior["alpha_offset_trial_erf"]
)
# results.posterior["gamma_alpha_channel_t"] = (
#     results.posterior["sigma_alpha_erf_channel"]
#     * results.posterior["channel prior alpha_erf"]
# )
results.posterior["gamma_beta_cluster_t"] = (
    results.posterior["sigma_beta_erf_cluster"]
    * results.posterior["beta_offset_cluster_erf"]
)
results.posterior["gamma_beta_trial_t"] = (
    results.posterior["sigma_beta_erf_trial"]
    * results.posterior["beta_offset_trial_erf"]
)
# results.posterior["gamma_beta_channel_t"] = (
#     results.posterior["sigma_beta_erf_channel"]
#     * results.posterior["channel prior beta_erf"]
# )
# Change the font size in matplotlib rc parameters
plt.rcParams["font.size"] = 16
############### Forest Plot ############################
f, ax = plt.subplots(2, 1, figsize=(16, 8))

az.plot_forest(
    results,
    var_names=["mu_beta_erf"],
    ax=ax[0],
    combined=True,
    colors="k",
    labeller=labeller,
    hdi_prob=0.94,
)
az.plot_forest(
    results,
    var_names=["gamma_beta_cluster_t", "gamma_beta_trial_t"],
    ax=ax[1],
    combined=True,
    colors="k",
    labeller=labeller,
    hdi_prob=0.94,
)
ax[0].axvline(0, color="k", linestyle="--")
ax[1].axvline(0, color="k", linestyle="--")
ax[1].set_title("")
ax[0].set_title(
    r"RT ~ $\alpha$ + $\theta_{cluster}$ + $\theta_{trial}$ + $\theta_{channel}$ + ($\beta$ + $\gamma_{cluster}$ + $\gamma_{trial}$ + $\gamma_{channel}$) $\times$ ERF"
)
f.savefig(join(figpath, "forest_mu_beta_erf_rt.jpg"))
###########  Now do the intercepts

f, ax = plt.subplots(2, 1, figsize=(16, 8))

az.plot_forest(
    results,
    var_names=["mu_alpha_erf"],
    ax=ax[0],
    combined=True,
    colors="k",
    labeller=labeller,
    hdi_prob=0.94,
)
az.plot_forest(
    results,
    var_names=["gamma_alpha_cluster_t", "gamma_alpha_trial_t"],
    ax=ax[1],
    combined=True,
    colors="k",
    labeller=labeller,
    hdi_prob=0.94,
)
ax[0].axvline(0, color="k", linestyle="--")
ax[1].axvline(0, color="k", linestyle="--")
ax[1].set_title("")
ax[0].set_title(
    r"RT ~ $\alpha$ + $\theta_{cluster}$ + $\theta_{trial}$ + $\theta_{channel}$ + ($\beta$ + $\gamma_{cluster}$ + $\gamma_{trial}$ + $\gamma_{channel}$) $\times$ ERF"
)
f.savefig(join(figpath, "forest_mu_alpha_erf_rt.jpg"))

########## Do a table of mean, HDI, rhat and effective sample size
labeller2 = azl.MapLabeller(
    var_name_map={
        "alpha_offset_trial_erf": r"$\alpha_{trial}$",
        "alpha_offset_cluster_erf": r"$\alpha_{cluster}$",
        "channel prior alpha_erf": r"$\alpha_{channel}$",
        "beta_offset_trial_erf": r"$\beta_{trial}$",
        "beta_offset_cluster_erf": r"$\beta_{cluster}$",
        "channel prior beta_erf": r"$\beta_{channel}$",
        "mu_alpha_erf": r"$\alpha$",
        "mu_beta_erf": r"$\beta$",
        "length_scale_alpha_erf": r"$\lambda_{alpha}$",
        "length_scale_beta_erf": r"$\lambda_{beta}$",
        "eta_sq_alpha_erf": r"$\eta^{2}_{alpha}$",
        "eta_sq_beta_erf": r"$\eta^{2}_{beta}$",
        "sigma_alpha_erf_cluster": r"$\sigma_{alpha, cluster}$",
        "sigma_alpha_erf_trial": r"$\sigma_{alpha, trial}$",
        "sigma_alpha_erf_channel": r"$\sigma_{alpha, channel}$",
        "sigma_beta_erf_cluster": r"$\sigma_{beta, cluster}$",
        "sigma_beta_erf_trial": r"$\sigma_{beta, trial}$",
        "sigma_beta_erf_channel": r"$\sigma_{beta, channel}$",
    }
)
var_names = [
        "alpha_offset_trial_erf",
        "alpha_offset_cluster_erf",
        "beta_offset_trial_erf",
        "beta_offset_cluster_erf",
        "mu_alpha_erf",
        "mu_beta_erf",
        "length_scale_alpha_erf",
        "length_scale_beta_erf",
        "eta_sq_alpha_erf",
        "eta_sq_beta_erf",
        "channel prior alpha_erf",
        "channel prior beta_erf",
        "sigma_alpha_erf_cluster",
        "sigma_alpha_erf_trial",
        # "sigma_alpha_erf_channel",
        "sigma_beta_erf_cluster",
        "sigma_beta_erf_trial",
        # "sigma_beta_erf_channel",
    ]
summary_df = az.summary(
    results,
    var_names=var_names,
    labeller=labeller2,
    hdi_prob=0.94,
)

summary_df.to_csv(join(figpath_supp, "erf_rt_diagnostic_table.csv"))

############# Now do trace plots ############
az.plot_trace(
    results,
    var_names=var_names,
    labeller=labeller2,
    figsize=(10, 70)
)
plt.savefig(join(figpath_supp, "trace_erf_rt.png"))
######## Prior vs posterior plot #############
# This is not very realistic, I'll reconsider this, keeping like this for now
"""with model:
    prior = pm.sample_prior_predictive(draws=100)

az.rcParams["plot.max_subplots"] = 600
# plt.rcParams["plot.max_subplots"] = 200
axs = az.plot_posterior(results, 
                 labeller=labeller2,
                 var_names=var_names, group="posterior", 
                 combine_dims={"chain", "draw", "Cluster", "Channel", "Trial"},
                 rope_color='C2', ref_val_color='C1',
                 figsize=(20,20),
                 hdi_prob=0.94)
az.plot_posterior(prior, 
                 labeller=labeller2,
                 var_names=var_names, group="prior", 
                 combine_dims={"chain", "draw", "Cluster", "Channel", "Trial"},
                 hdi_prob=0.94,
                 rope_color='black', ref_val_color='black',
                 ax=axs)
plt.savefig(join(figpath_supp, "prior_vs_post_erf_rt.png"))
"""

########### Channel gamma / theta plotting #############

gamma_beta_channel = results.posterior["channel prior beta_erf"].mean(dim=("draw", "chain")).values
gamma_alpha_channel = results.posterior["channel prior alpha_erf"].mean(dim=("draw", "chain")).values
gamma_beta_channel_hdi = az.hdi(results.posterior, var_names=["channel prior beta_erf"], hdi_prob=0.94)["channel prior beta_erf"].values
gamma_alpha_channel_hdi = az.hdi(results.posterior, var_names=["channel prior alpha_erf"], hdi_prob=0.94)["channel prior alpha_erf"].values


template_good = get_template_bad(good=True)
chans = data.dropna().channels.unique()
chan_order = np.argsort(chans)

info = template_good.pick(chans).info
channel_coords = data.groupby("channels")[["xcoords", "ycoords"]].mean()
coords = channel_coords.values


# -- ALPHA -- #
minval = gamma_alpha_channel_hdi.min()
maxval = gamma_alpha_channel_hdi.max()
fraction=0.05
pad=0.04
climval = np.max(np.abs([minval, maxval]))
vlim = (-climval, climval)

f, ax = plt.subplots(1,3, layout="constrained")
im0, cm0 = mne.viz.plot_topomap(
        data=gamma_alpha_channel_hdi[chan_order, 0],
        pos=info,
        axes=ax.ravel()[0],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )

ax[0].set_title("6% HDI")
im1, cm1 = mne.viz.plot_topomap(
        data=gamma_alpha_channel[chan_order],
        pos=info,
        axes=ax.ravel()[1],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )
ax[1].set_title(r"$\gamma_{channel}$" + "\n\nMean")

im2, cm2 = mne.viz.plot_topomap(
        data=gamma_alpha_channel_hdi[chan_order, 1],
        pos=info,
        axes=ax.ravel()[2],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )
ax[2].set_title("94% HDI")
f.colorbar(im2, fraction=fraction, pad=pad)
f.suptitle(r"$\theta_{channel}$")

f.savefig(join(figpath, "channels_theta_erf_rt.png"), dpi=500, transparent=True)


# -- BETA -- #
minval = gamma_beta_channel_hdi.min()
maxval = gamma_beta_channel_hdi.max()
fraction=0.05
pad=0.04
climval = np.max(np.abs([minval, maxval]))
vlim = (-climval, climval)

f, ax = plt.subplots(1,3, layout="constrained")
im0, cm0 = mne.viz.plot_topomap(
        data=gamma_beta_channel_hdi[chan_order, 0],
        pos=info,
        axes=ax.ravel()[0],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )

ax[0].set_title("6% HDI")
im1, cm1 = mne.viz.plot_topomap(
        data=gamma_beta_channel[chan_order],
        pos=info,
        axes=ax.ravel()[1],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )
ax[1].set_title(r"$\gamma_{channel}$" + "\n\nMean")

im2, cm2 = mne.viz.plot_topomap(
        data=gamma_beta_channel_hdi[chan_order, 1],
        pos=info,
        axes=ax.ravel()[2],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )
ax[2].set_title("94% HDI")
f.colorbar(im2, fraction=fraction, pad=pad)
f.suptitle(r"$\gamma_{channel}$")

f.savefig(join(figpath, "channels_gamma_erf_rt.png"), dpi=500, transparent=True)

#* Do channels and add beta to gamma ###############################

beta_channel_i = results.posterior["beta_erf"].mean(dim=("draw", "chain")).values
alpha_channel_i = results.posterior["alpha_erf"].mean(dim=("draw", "chain")).values
beta_channel_hdi_i = az.hdi(results.posterior, var_names=["beta_erf"], hdi_prob=0.94)["beta_erf"].values
alpha_channel_hdi_i = az.hdi(results.posterior, var_names=["alpha_erf"], hdi_prob=0.94)["alpha_erf"].values

channel_idx = results.constant_data.Channel.values
uniq_chan = np.unique(channel_idx)
beta_channel = np.zeros(len(uniq_chan))
alpha_channel = np.zeros(len(uniq_chan))
beta_channel_hdi = np.zeros((len(uniq_chan), 2))
alpha_channel_hdi = np.zeros((len(uniq_chan), 2))
for chan in uniq_chan:
    idx = np.where(channel_idx == chan)[0]
    beta_channel[chan] = beta_channel_i[idx].mean()
    alpha_channel[chan] = alpha_channel_i[idx].mean()
    beta_channel_hdi[chan, :] = beta_channel_hdi_i[idx].mean(axis=0)
    alpha_channel_hdi[chan, :] = alpha_channel_hdi_i[idx].mean(axis=0)

minval = (beta_channel_hdi + gamma_beta_channel_hdi).min()
maxval = (beta_channel_hdi + gamma_beta_channel_hdi).max()
fraction=0.05
pad=0.04
climval = np.max(np.abs([minval, maxval]))
vlim = (-climval, climval)


f, ax = plt.subplots(1,3, layout="constrained")
im0, cm0 = mne.viz.plot_topomap(
        data=beta_channel_hdi[chan_order, 0] + gamma_beta_channel_hdi[chan_order, 0],
        pos=info,
        axes=ax.ravel()[0],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )

ax[0].set_title("6% HDI")
im1, cm1 = mne.viz.plot_topomap(
        data=beta_channel[chan_order] + gamma_beta_channel[chan_order],
        pos=info,
        axes=ax.ravel()[1],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )
ax[1].set_title(r"$\beta$ + $\gamma_{channel}$" + "\n\nMean")

im2, cm2 = mne.viz.plot_topomap(
        data=beta_channel_hdi[chan_order, 0] + gamma_beta_channel_hdi[chan_order, 1],
        pos=info,
        axes=ax.ravel()[2],
        cmap="PiYG",
        vlim=vlim,
        show=False,
        size=15,
        res=800
    )
ax[2].set_title("94% HDI")
f.colorbar(im2, fraction=fraction, pad=pad)

f.savefig(join(figpath, "channels_beta_plus_gamma_erf_rt.png"), dpi=500, transparent=True)

############################# Posterior for distances (eta squared) ####################
# compute posterior median covariance
x_seq = np.linspace(0, 10, 100)
post = results.posterior.stack(sample=("chain", "draw"))
pmcov_mu = post["eta_sq_beta_erf"].median().values * np.exp(-post["length_scale_beta_erf"].median().values * (x_seq**2))

# plot functions sampled from posterior
f, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(
    x_seq,
    (
        post["eta_sq_beta_erf"][::50].values[:, None]
        * np.exp(-post["length_scale_beta_erf"][::50].values[:, None] * (x_seq**2))
    ).T,
    "k",
    alpha=0.08,
)

# plot median covariance function
ax.plot(x_seq, pmcov_mu, lw=3)

ax.set(
    title="Gaussian process posterior",
    ylim=(-0.1, 2),
    xlabel="distance (normalized)",
    ylabel="covariance",
)

f.savefig(join(figpath, "posterior_covariance_erf_rt.png"), dpi=800, transparent=True)