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
import os
from erf_acw2.src import (
    get_commonsubj,
    pklsave,
    pklload,
    import_exampleraw,
    acf_oscillatory_function
)

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

sns.set_theme()
sns.set_theme(style="whitegrid")
so.Plot.config.theme.update(axes_style("whitegrid"))

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_reviewer_comments"
if not os.path.exists(figpath):
    os.makedirs(figpath)


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
    "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/acfs_rest_10rois.jld2",
    "r",
)
# acfs = f.get("acfs")[...]
acw50s = f.get("acw50s")[...]
acw50s = np.nanmean(acw50s, axis=2)
f.close()

f = h5py.File(
    "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/erfs_task_10rois.jld2", "r"
)
erfs = f.get("erfs")[...]
rmss = f.get("rmss")[...]
activationflags = f.get("activationflags")[...]
f.close()

f = h5py.File("/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/rest_10rois.jld2", "r")
gamma_1 = f.get("gamma_1_values")[...]
A_B = f.get("A_B_values")[...]
A_L = f.get("A_L_values")[...]
A_F = f.get("A_F_values")[...]
f.close()

dr_data = pklload(os.path.join(figpath, "decay_rate_data.pkl"))
popt_all = dr_data["popt_all"]
acws_all = dr_data["acws_all"]
ints_all = dr_data["ints_all"]

nsims = 500
nrois = 10

all_acw50s = []
all_ints = []
all_erfs = []
all_gamma_1 = []
all_sim_idx = []
all_previous_L = []
all_previous_F = []
all_previous_B = []
all_next_L = []
all_next_F = []
all_next_B = []
all_roi_idx = []
for i in range(nsims):
    for j in range(nrois):
        all_acw50s.append(acw50s[j, i])
        all_ints.append(ints_all[i, j])
        if activationflags[i, j]:
            all_erfs.append(rmss[i, j])
        else:
            all_erfs.append(np.nan)
        
        all_gamma_1.append(gamma_1[i, j])
        if j == 0:
            all_previous_L.append(np.nan)
            all_previous_F.append(np.nan)
            all_previous_B.append(np.nan)
        else:
            all_previous_L.append(A_L[i, j - 1])
            all_previous_F.append(A_F[i, j - 1])
            all_previous_B.append(A_B[i, j - 1])

        if j == (nrois - 1):
            all_next_L.append(np.nan)
            all_next_F.append(np.nan)
            all_next_B.append(np.nan)
        else:
            all_next_L.append(A_L[i, j])
            all_next_F.append(A_F[i, j])
            all_next_B.append(A_B[i, j])

        all_sim_idx.append(i)
        all_roi_idx.append(j)


df = pd.DataFrame(
    {
        "acw50s": all_acw50s,
        "ints": all_ints,
        "erfs": all_erfs,
        "gamma_1": all_gamma_1,
        "previous_L": all_previous_L,
        "previous_F": all_previous_F,
        "previous_B": all_previous_B,
        "next_L": all_next_L,
        "next_F": all_next_F,
        "next_B": all_next_B,
        "roi_idx": all_roi_idx,
        "sim_idx": all_sim_idx,
    }
)
df.to_csv(os.path.join(figpath, "erf_acw_data.csv"))


# df_filtered = df[(df["erfs"] > 1.0) & (df["ints"] < 0.04)]
df_filtered = df

plt.close()
sns.scatterplot(data=df_filtered[df_filtered["roi_idx"] == 0], x="ints", y="erfs")
# plt.xlim((0, 0.04))
plt.savefig(os.path.join(figpath, "erf_acw_correlation_filtered.jpg"))

plt.close()
sns.scatterplot(data=df_filtered[df_filtered["roi_idx"] == 0], x="gamma_1", y="ints")
# plt.xlim((0, 0.04))
plt.savefig(os.path.join(figpath, "erf_acw_correlation_filtered_gamma_1.jpg"))


plt.close()
sns.regplot(data=df_filtered, x="gamma_1", y="ints")
# plt.xlim((0, 0.04))
plt.savefig(os.path.join(figpath, "erf_acw_correlation_filtered_gamma_1_allroi.jpg"))


plt.close()
sns.regplot(data=df_filtered, x="gamma_1", y="erfs")
# plt.xlim((0, 0.04))
plt.savefig(os.path.join(figpath, "erf_acw_correlation_filtered_gamma_1_allroi.jpg"))

plt.close()
sns.scatterplot(data=df_filtered[df_filtered["roi_idx"] == 0], x="gamma_1", y="acw50s")
# plt.xlim((0, 0.04))
plt.savefig(os.path.join(figpath, "erf_acw_correlation_filtered_gamma_1.jpg"))

df_filtered[df_filtered["roi_idx"] == 0][["erfs", "ints", "gamma_1", "acw50s"]].corr()

df_regression = df_filtered[df_filtered["roi_idx"] == 0][["ints", "gamma_1", "next_L", "next_F", "next_B"]].dropna()
df_regression["next_L_z"] = zscore(df_regression["next_L"])
df_regression["next_F_z"] = zscore(df_regression["next_F"])
df_regression["next_B_z"] = zscore(df_regression["next_B"])
df_regression["gamma_1_z"] = zscore(df_regression["gamma_1"])
df_regression["ints_z"] = zscore(df_regression["ints"])
y = df_regression.ints_z
X = df_regression[["gamma_1_z", "next_L_z", "next_F_z", "next_B_z"]]
pg.linear_regression(X, y)


import bambi as bmb
import pymc as pm
df_filtered["ints_z"] = zscore(df_filtered["ints"])
df_filtered["acw50s_z"] = zscore(df_filtered["acw50s"])
df_filtered["gamma_1_z"] = zscore(df_filtered["gamma_1"])
df_filtered["next_L_z"] = zscore(df_filtered["next_L"])
df_filtered["next_F_z"] = zscore(df_filtered["next_F"])
df_filtered["next_B_z"] = zscore(df_filtered["next_B"])
df_filtered["previous_L_z"] = zscore(df_filtered["previous_L"])
df_filtered["previous_F_z"] = zscore(df_filtered["previous_F"])
df_filtered["previous_B_z"] = zscore(df_filtered["previous_B"])

# Prepare data for modeling - handle missing connections properly
# Replace NaN values with 0 and create indicator variables
df_model = df_filtered.copy()

# Create indicator variables for which connections exist
df_model['has_previous'] = ~df_model[['previous_L_z', 'previous_F_z', 'previous_B_z']].isna().any(axis=1)
df_model['has_next'] = ~df_model[['next_L_z', 'next_F_z', 'next_B_z']].isna().any(axis=1)

# Fill NaN values with 0 (they won't contribute to the linear predictor)
df_model = df_model.fillna(0)

# Create ROI index for grouping
roi_idx, unique_rois = pd.factorize(df_model['roi_idx'])
n_rois = len(unique_rois)
n_obs = len(df_model)

# Set up coordinates for PyMC
coords = {
    "roi": unique_rois,
    "obs": range(n_obs)
}

with pm.Model(coords=coords) as hierarchical_model:
    # Data containers
    roi_idx_data = pm.Data("roi_idx", roi_idx, dims="obs")
    
    # Outcome variable
    ints_z_obs = pm.Data("ints_z", df_model['ints_z'].values, dims="obs")
    
    # Predictor variables (now with 0s instead of NaN)
    gamma_1_z_data = pm.Data("gamma_1_z", df_model['gamma_1_z'].values, dims="obs")
    next_L_z_data = pm.Data("next_L_z", df_model['next_L_z'].values, dims="obs")
    next_F_z_data = pm.Data("next_F_z", df_model['next_F_z'].values, dims="obs")
    next_B_z_data = pm.Data("next_B_z", df_model['next_B_z'].values, dims="obs")
    previous_L_z_data = pm.Data("previous_L_z", df_model['previous_L_z'].values, dims="obs")
    previous_F_z_data = pm.Data("previous_F_z", df_model['previous_F_z'].values, dims="obs")
    previous_B_z_data = pm.Data("previous_B_z", df_model['previous_B_z'].values, dims="obs")
    
    # Indicator variables for which connections exist
    has_previous = pm.Data("has_previous", df_model['has_previous'].values.astype(int), dims="obs")
    has_next = pm.Data("has_next", df_model['has_next'].values.astype(int), dims="obs")
    
    # Fixed effects (population-level intercept and slopes)
    intercept = pm.Normal("intercept", 0, 1)
    beta_gamma_1 = pm.Normal("beta_gamma_1", 0, 1)
    beta_next_L = pm.Normal("beta_next_L", 0, 1)
    beta_next_F = pm.Normal("beta_next_F", 0, 1)
    beta_next_B = pm.Normal("beta_next_B", 0, 1)
    beta_previous_L = pm.Normal("beta_previous_L", 0, 1)
    beta_previous_F = pm.Normal("beta_previous_F", 0, 1)
    beta_previous_B = pm.Normal("beta_previous_B", 0, 1)
    
    # Random effects: ROI-level intercepts and slopes
    # Intercept varying by ROI
    sigma_intercept = pm.HalfNormal("sigma_intercept", 1)
    intercept_roi = pm.Normal("intercept_roi", 0, sigma_intercept, dims="roi")
    
    # Slopes varying by ROI
    sigma_gamma_1 = pm.HalfNormal("sigma_gamma_1", 1)
    beta_gamma_1_roi = pm.Normal("beta_gamma_1_roi", 0, sigma_gamma_1, dims="roi")
    
    sigma_next_L = pm.HalfNormal("sigma_next_L", 1)
    beta_next_L_roi = pm.Normal("beta_next_L_roi", 0, sigma_next_L, dims="roi")
    
    sigma_next_F = pm.HalfNormal("sigma_next_F", 1)
    beta_next_F_roi = pm.Normal("beta_next_F_roi", 0, sigma_next_F, dims="roi")
    
    sigma_next_B = pm.HalfNormal("sigma_next_B", 1)
    beta_next_B_roi = pm.Normal("beta_next_B_roi", 0, sigma_next_B, dims="roi")
    
    sigma_previous_L = pm.HalfNormal("sigma_previous_L", 1)
    beta_previous_L_roi = pm.Normal("beta_previous_L_roi", 0, sigma_previous_L, dims="roi")
    
    sigma_previous_F = pm.HalfNormal("sigma_previous_F", 1)
    beta_previous_F_roi = pm.Normal("beta_previous_F_roi", 0, sigma_previous_F, dims="roi")
    
    sigma_previous_B = pm.HalfNormal("sigma_previous_B", 1)
    beta_previous_B_roi = pm.Normal("beta_previous_B_roi", 0, sigma_previous_B, dims="roi")
    
    # Linear predictor with proper handling of missing connections
    mu = pm.Deterministic(
        "mu",
        (intercept + intercept_roi[roi_idx_data]) +
        (beta_gamma_1 + beta_gamma_1_roi[roi_idx_data]) * gamma_1_z_data +
        # Next connections (only contribute when they exist)
        has_next * (beta_next_L + beta_next_L_roi[roi_idx_data]) * next_L_z_data +
        has_next * (beta_next_F + beta_next_F_roi[roi_idx_data]) * next_F_z_data +
        has_next * (beta_next_B + beta_next_B_roi[roi_idx_data]) * next_B_z_data +
        # Previous connections (only contribute when they exist)
        has_previous * (beta_previous_L + beta_previous_L_roi[roi_idx_data]) * previous_L_z_data +
        has_previous * (beta_previous_F + beta_previous_F_roi[roi_idx_data]) * previous_F_z_data +
        has_previous * (beta_previous_B + beta_previous_B_roi[roi_idx_data]) * previous_B_z_data,
        dims="obs"
    )
    
    # Observation noise
    sigma = pm.HalfNormal("sigma", 1)
    
    # Likelihood
    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=ints_z_obs, dims="obs")

# Sample from the model
with hierarchical_model:
    # Sample prior predictive to check model setup
    # prior_predictive = pm.sample_prior_predictive(samples=1000)
    
    # Sample posterior
    trace = pm.sample(draws=2000, tune=2000, chains=4, target_accept=0.95)
    
    # Sample posterior predictive
    posterior_predictive = pm.sample_posterior_predictive(trace)

# Save results
import arviz as az
inference_data = az.from_pymc(trace, prior=prior_predictive, posterior_predictive=posterior_predictive)
pklsave(inference_data, os.path.join(figpath, "hierarchical_model_results.pkl"))

# Model summary
print(az.summary(trace, var_names=["intercept", "beta_gamma_1", "beta_next_L", "beta_next_F", 
                                   "beta_next_B", "beta_previous_L", "beta_previous_F", "beta_previous_B",
                                   "sigma_intercept", "sigma_gamma_1", "sigma_next_L", "sigma_next_F",
                                   "sigma_next_B", "sigma_previous_L", "sigma_previous_F", "sigma_previous_B", "sigma"]))

