# cd /BICNAS2/ycatal/erf_acw2/scripts/behavior
# nohup python S22_acw_behavior_hierarchical.py > log/hierarchical_nuts_acw_rt.log &
# pid: 3680444
import os
import numpy as np
import pandas as pd
import pymc as pm
from scipy.spatial.distance import pdist, squareform
import cloudpickle
import matplotlib.pyplot as plt
from os.path import join
import xarray as xr
os.chdir("/BICNAS2/ycatal/erf_acw2/")

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure5"
def cloud_pklsave(filename, obj):
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)

data = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")
data = data.drop_duplicates(subset=["channels", "subjects"]) # ACWs repeat across trials / clusters. Get rid of it. 

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
with pm.Model() as model:
    # Data
    acws = pm.Data("acws", data3["restacws"].values)
    rts = pm.Data("rts", data3["rts"].values)

    # cluster_c = pm.Data("Cluster", cluster_code)
    channel_c = pm.Data("Channel", channel_code)

    n_channels = len(np.unique(channel_code))
    # n_clusters = len(np.unique(cluster_code))

    # Hyperpriors
    mu_alpha_erf = pm.Normal("mu_alpha_erf", mu=0, sigma=1, shape=1)  # For ERF
    # sigma_alpha_erf_cluster = pm.HalfNormal("sigma_alpha_erf_cluster", sigma=1, shape=1)
    sigma_alpha_erf_channel = pm.HalfNormal("sigma_alpha_erf_channel", sigma=1, shape=1)

    mu_beta_erf = pm.Normal("mu_beta_erf", mu=0, sigma=1, shape=1)
    # sigma_beta_erf_cluster = pm.HalfNormal("sigma_beta_erf_cluster", sigma=1, shape=1)
    sigma_beta_erf_channel = pm.HalfNormal("sigma_beta_erf_channel", sigma=1, shape=1)

    # Non - centered parametrization for cluster and trial
    # alpha_offset_cluster_erf = pm.Normal("alpha_offset_cluster_erf", mu=0, sigma=1, dims="clusters")
    # beta_offset_cluster_erf = pm.Normal("beta_offset_cluster_erf", mu=0, sigma=1, dims="clusters")

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

    # gamma_alpha_cluster = pm.Deterministic("gamma_alpha_cluster", sigma_alpha_erf_cluster * alpha_offset_cluster_erf[cluster_c])
    gamma_alpha_channel = pm.Deterministic("gamma_alpha_channel", sigma_alpha_erf_channel * channel_prior_alpha_erf[channel_c])

    # gamma_beta_cluster = pm.Deterministic("gamma_beta_cluster", sigma_beta_erf_cluster * beta_offset_cluster_erf[cluster_c])
    gamma_beta_channel = pm.Deterministic("gamma_beta_channel", sigma_beta_erf_channel * channel_prior_beta_erf[channel_c])

    alpha_erf = pm.Deterministic(
        "alpha_erf",
        mu_alpha_erf + gamma_alpha_channel # + gamma_alpha_cluster + 
    )
    beta_erf = pm.Deterministic(
        "beta_erf",
        mu_beta_erf + gamma_beta_channel # + gamma_beta_cluster
    )
    
    # Model
    mu_rt = alpha_erf + beta_erf * acws
    sigma_rt = pm.Exponential("sigma_rt", 1)
    nu_rt = pm.Exponential("nu_rt", 1)
    rt_lkl = pm.StudentT("rt_lkl", mu=mu_rt, sigma=sigma_rt, nu=nu_rt, observed=rts)

with model:
    trace = pm.sample(target_accept=0.9, draws=1000)

trace.to_netcdf("/BICNAS2/ycatal/erf_acw2/results/hierarchical_model/acw_rt_hierarchical.cdf")
cloud_pklsave("/BICNAS2/ycatal/erf_acw2/results/hierarchical_model/acw_rt_hierarchical.pkl", model)

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