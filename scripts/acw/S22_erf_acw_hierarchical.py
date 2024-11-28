# cd /BICNAS2/ycatal/erf_acw2/scripts/acw
# nohup python S24_erf_acw_hierarchical.py > log/hierarchical_nuts_erf_acw.log &
# pid: 3707647
# pid: 3875097
import os
import numpy as np
import pandas as pd
import pymc as pm
from scipy.spatial.distance import pdist, squareform
import cloudpickle
from os.path import join
os.chdir("/BICNAS2/ycatal/erf_acw2/")

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure5"
def cloud_pklsave(filename, obj):
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)

data = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")


def zscore(x):
    return (x - x.mean()) / x.std()


def codify(df, varname):
    uniques = df[varname].unique()
    mapping = {ch: i for i, ch in enumerate(uniques)}
    code = df[varname].map(mapping)
    return code


data2 = data.copy()
data2["erfs"] = zscore(data2["erfs"])
data2["restacws"] = zscore(data2["restacws"])
data2["rts"] = zscore(data2["rts"])
data2["xcoords"] = zscore(data2["xcoords"])
data2["ycoords"] = zscore(data2["ycoords"])
data3 = data2.dropna()

subject_code = codify(data3, "subjects")
channel_code = codify(data3, "channels")
cluster_code = codify(data3, "clusternames")
trial_code = codify(data3, "erftype")

# Get distance matrix
channel_coords = data3.groupby("channels")[["xcoords", "ycoords"]].mean()
coords = channel_coords.values
D = squareform(pdist(coords))

cluster_names = ["Encode #1", "Encode #2", "Encode #3", "Encode #4", "Probe #1", "Probe #2"]
trial_names = ["efh", "efs", "es", "pfh", "pfs", "ps"]
with pm.Model(coords={"clusters": cluster_names,
                      "trials": trial_names}) as model:
    # Data
    erfs = pm.Data("erfs", data3["erfs"].values)
    acws = pm.Data("acws", data3["restacws"].values)

    cluster_c = pm.Data("Cluster", cluster_code)
    channel_c = pm.Data("Channel", channel_code)
    trial_c = pm.Data("Trial", trial_code)

    n_channels = len(np.unique(channel_code))
    n_clusters = len(np.unique(cluster_code))
    n_trial = len(np.unique(trial_code))

    # Hyperpriors
    mu_alpha_erf = pm.Normal("mu_alpha_erf", mu=0, sigma=1, shape=1)  # For ERF
    sigma_alpha_erf_cluster = pm.HalfNormal("sigma_alpha_erf_cluster", sigma=1, shape=1)
    sigma_alpha_erf_trial = pm.HalfNormal("sigma_alpha_erf_trial", sigma=1, shape=1)
    sigma_alpha_erf_channel = pm.HalfNormal("sigma_alpha_erf_channel", sigma=1, shape=1)

    mu_beta_erf = pm.Normal("mu_beta_erf", mu=0, sigma=1, shape=1)
    sigma_beta_erf_cluster = pm.HalfNormal("sigma_beta_erf_cluster", sigma=1, shape=1)
    sigma_beta_erf_trial = pm.HalfNormal("sigma_beta_erf_trial", sigma=1, shape=1)
    sigma_beta_erf_channel = pm.HalfNormal("sigma_beta_erf_channel", sigma=1, shape=1)

    # Non - centered parametrization for cluster and trial
    alpha_offset_cluster_erf = pm.Normal("alpha_offset_cluster_erf", mu=0, sigma=1, dims="clusters")
    beta_offset_cluster_erf = pm.Normal("beta_offset_cluster_erf", mu=0, sigma=1, dims="clusters")

    alpha_offset_trial_erf = pm.Normal("alpha_offset_trial_erf", mu=0, sigma=1, dims="trials")
    beta_offset_trial_erf = pm.Normal("beta_offset_trial_erf", mu=0, sigma=1, dims="trials")

    # GP for channel effects - Alpha ERF
    ls_inv_alpha_erf = pm.HalfNormal("length_scale_alpha_erf", 1.0)
    etasq_alpha_erf = pm.Exponential("eta_sq_alpha_erf", 1.0)
    cov_alpha_erf = etasq_alpha_erf * pm.gp.cov.ExpQuad(input_dim=len(D), ls=ls_inv_alpha_erf)
    gp_alpha_erf = pm.gp.Latent(cov_func=cov_alpha_erf)
    channel_prior_alpha_erf = gp_alpha_erf.prior("channel prior alpha_erf", X=D)
    
    # GP for channel effects - Beta ERF
    ls_inv_beta_erf = pm.HalfNormal("length_scale_beta_erf", 1.0)
    etasq_beta_erf = pm.Exponential("eta_sq_beta_erf", 1.0)
    cov_beta_erf = etasq_beta_erf * pm.gp.cov.ExpQuad(input_dim=len(D), ls=ls_inv_beta_erf)
    gp_beta_erf = pm.gp.Latent(cov_func=cov_beta_erf)
    channel_prior_beta_erf = gp_beta_erf.prior("channel prior beta_erf", X=D)

    gamma_alpha_cluster = pm.Deterministic("gamma_alpha_cluster", sigma_alpha_erf_cluster * alpha_offset_cluster_erf[cluster_c])
    gamma_alpha_trial = pm.Deterministic("gamma_alpha_trial", sigma_alpha_erf_trial * alpha_offset_trial_erf[trial_c])
    gamma_alpha_channel = pm.Deterministic("gamma_alpha_channel", sigma_alpha_erf_channel * channel_prior_alpha_erf[channel_c])

    gamma_beta_cluster = pm.Deterministic("gamma_beta_cluster", sigma_beta_erf_cluster * beta_offset_cluster_erf[cluster_c])
    gamma_beta_trial = pm.Deterministic("gamma_beta_trial", sigma_beta_erf_trial * beta_offset_trial_erf[trial_c])
    gamma_beta_channel = pm.Deterministic("gamma_beta_channel", sigma_beta_erf_channel * channel_prior_beta_erf[channel_c])

    alpha_erf = pm.Deterministic(
        "alpha_erf",
        mu_alpha_erf + gamma_alpha_cluster + gamma_alpha_trial + gamma_alpha_channel
    )
    beta_erf = pm.Deterministic(
        "beta_erf",
        mu_beta_erf + gamma_beta_cluster + gamma_beta_trial + gamma_beta_channel
    )
    
    # Model
    mu_acw = alpha_erf + beta_erf * erfs
    sigma_acw = pm.Exponential("sigma_acw", 1)
    nu_acw = pm.Exponential("nu_acw", 1)
    acw_lkl = pm.StudentT("acw_lkl", mu=mu_acw, sigma=sigma_acw, nu=nu_acw, observed=acws)

with model:
    trace = pm.sample(target_accept=0.9, draws=1000)

trace.to_netcdf("/BICNAS2/ycatal/erf_acw2/results/hierarchical_model/erf_acw_hierarchical.cdf")
cloud_pklsave("/BICNAS2/ycatal/erf_acw2/results/hierarchical_model/erf_acw_hierarchical.pkl", model)

# with model:
#     prior = pm.sample_prior_predictive(draws=10)

# x = np.linspace(-2, 2, 10)
# x_tiled = np.zeros((10, 2000))
# for i in range(2000):
#     x_tiled[:, i] = x
# prior_pred_check = np.zeros((10, 2000))
# for i in range(2000):
#     prior_pred_check[:, i] = (prior.prior["alpha_erf"][0, np.random.randint(10), i].to_numpy() + 
#         prior.prior["beta_erf"][0, np.random.randint(10), i].to_numpy() * x_tiled[:, i])

# f, ax = plt.subplots()
# for i in range(2000):
#     ax.plot(
#         x_tiled[:, i], 
#         prior_pred_check[:, i], 
#         c="k", alpha=0.1
#         )

# ax.set_xlabel("ERF (stdz)")
# ax.set_ylabel("RT (stdz)")
# ax.set_title("Prior predictive checks")
# f.savefig(join(figpath, "supplementary", "prior_predictive.png"))