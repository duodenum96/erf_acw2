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
import arviz.labels as azl
from erf_acw2.src import rope, p_direction
sns.set_theme()
sns.set_theme(style="whitegrid")
so.Plot.config.theme.update(axes_style("whitegrid"))

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_f2"
figpath_supp = join(figpath, "supplementary")

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
        "gamma_1": zscore(np.tile(gamma1, [nsim, 1]).T.ravel()), 
        "Color:\n$\\gamma_1$": np.tile(gamma1_cat, [nsim, 1]).T.ravel(), 
    }
)

# Model 1: 
# ERF ~ ACW
# Model 2:
# ERF ~ ACW + gamma1

priors_1 = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    "ERF": bmb.Prior("Normal", mu=0, sigma=1),
    "sigma": bmb.Prior("Exponential", lam=1),
    "nu": bmb.Prior("Exponential", lam=1),
}
priors_2 = priors_1.copy()
priors_2["gamma_1"] = bmb.Prior("Normal", mu=0, sigma=1)

a2e_model = bmb.Model("ACW ~ ERF", data, family="t", priors=priors_1)
tr_a2e = a2e_model.fit()
a2e_g_model = bmb.Model("ACW ~ ERF + gamma_1", data, family="t", priors=priors_2)
tr_a2e_g = a2e_g_model.fit()

az.to_netcdf(tr_a2e, join(figpath_supp, "tr_a2e.nc"))
az.to_netcdf(tr_a2e_g, join(figpath_supp, "tr_a2e_g.nc"))

all_traces = [tr_a2e, tr_a2e_g]
all_model_names = ["ACW ~ ERF", "ACW ~ ERF + gamma_1"]
all_model_names_map = all_model_names.copy()
all_model_names_map[1] = "ACW ~ ERF + $\\gamma_1$"
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
ax1.set_yticklabels(["mERF"])
ax1.set_title("Model 1: ACW ~ mERF")
ax1.axvline(0, color="black", linestyle="--", linewidth=1)
ax1.set_xlim(-0.12, 0.95)

# Plot model 2: ERF ~ ACW + gamma1 
az.plot_forest([tr_a2e_g], 
               model_names=["ERF ~ ACW + $\\gamma_1$"], 
               combined=True, hdi_prob=0.94, var_names=["~sigma", "~Intercept", "~nu"],
               ax=ax2, colors="k", rope=[-0.1, 0.1])
ax2.set_yticklabels(["$\\gamma_1$", "mERF"]) 
ax2.set_title("Model 2: ACW ~ mERF + $\\gamma_1$")
ax2.axvline(0, color="black", linestyle="--", linewidth=1)
ax2.set_xticks(np.arange(-0.1, 1.1, 0.1))
ax2.set_xlim(-0.12, 0.95)

plt.tight_layout()
plt.savefig(join(figpath, "forests_linregress_a2e_combined.jpg"), dpi=800)
plt.close()

######### Do prior predictive checks, trace plots, and posterior predictive checks

# Prior predictive checks
prior_pred_a2e = a2e_model.prior_predictive()
prior_pred_a2e_g = a2e_g_model.prior_predictive()
a2e_model.predict(tr_a2e, kind="response")
a2e_g_model.predict(tr_a2e_g, kind="response")

plt.rcParams.update({"font.size": 12})
f, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
az.plot_ppc(prior_pred_a2e, ax=ax[0, 0], num_pp_samples=20, group="prior")
ax[0, 0].set_xlim(-15, 15)
ax[0, 0].set_title("Prior predictive check")
ax[0, 0].set_xlabel("")

az.plot_ppc(prior_pred_a2e, ax=ax[0, 1], num_pp_samples=20, group="prior", kind="cumulative")
ax[0, 1].set_xlim(-15, 15)
ax[0, 1].set_title("Prior predictive check")
ax[0, 1].set_xlabel("")
ax[0, 1].set_yticks([])

az.plot_ppc(tr_a2e, ax=ax[1, 0], num_pp_samples=20, group="posterior")
ax[1, 0].set_xlim(-5, 5)
ax[1, 0].set_title("Posterior predictive check")
ax[1, 0].set_xlabel("ACW")

az.plot_ppc(tr_a2e, ax=ax[1, 1], num_pp_samples=20, group="posterior", kind="cumulative")
ax[1, 1].set_xlim(-5, 5)
ax[1, 1].set_title("Posterior predictive check")
ax[1, 1].set_xlabel("ACW")
ax[1, 1].set_yticks([])
f.suptitle("Model 1: ACW ~ mERF")
f.savefig(join(figpath_supp, "prior_and_posterior_a2e.png"), dpi=300, transparent=True)

f, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
az.plot_ppc(prior_pred_a2e_g, ax=ax[0, 0], num_pp_samples=20, group="prior")
ax[0, 0].set_xlim(-15, 15)
ax[0, 0].set_title("Prior predictive check")
ax[0, 0].set_xlabel("")

az.plot_ppc(prior_pred_a2e_g, ax=ax[0, 1], num_pp_samples=20, group="prior", kind="cumulative")
ax[0, 1].set_xlim(-15, 15)
ax[0, 1].set_title("Prior predictive check")
ax[0, 1].set_xlabel("")
ax[0, 1].set_yticks([])

az.plot_ppc(tr_a2e_g, ax=ax[1, 0], num_pp_samples=20, group="posterior")
ax[1, 0].set_xlim(-5, 5)
ax[1, 0].set_title("Posterior predictive check")
ax[1, 0].set_xlabel("ACW")

az.plot_ppc(tr_a2e_g, ax=ax[1, 1], num_pp_samples=20, group="posterior", kind="cumulative")
ax[1, 1].set_xlim(-5, 5)
ax[1, 1].set_title("Posterior predictive check")
ax[1, 1].set_xlabel("ACW")
ax[1, 1].set_yticks([])

f.suptitle("Model 2: ACW ~ mERF + $\gamma_1$")
f.savefig(join(figpath_supp, "prior_and_posterior_a2e_g.png"), dpi=300, transparent=True)

# Trace plots

az.plot_trace(tr_a2e, figsize=(12, 18), var_names=["~mu"])
plt.suptitle("Model 1: ACW ~ mERF")
plt.savefig(join(figpath_supp, f"trace_a2e.png"))

az.plot_trace(tr_a2e_g, figsize=(12, 18), var_names=["~mu"])
plt.suptitle("Model 2: ACW ~ mERF + $\gamma_1$")
plt.savefig(join(figpath_supp, f"trace_a2e_g.png"))

# Add ROPE and pd to arviz summaries
summary_a2e = az.summary(tr_a2e, var_names=["~mu"])
summary_a2e_g = az.summary(tr_a2e_g, var_names=["~mu"])
varnames_a2e = ["Intercept", "ERF"]
varnames_a2e_g = ["Intercept", "ERF", "gamma_1"]
for varname in varnames_a2e:
    print(varname)
    pd = p_direction(tr_a2e, varname)
    r = rope(tr_a2e, varname)
    summary_a2e.loc[varname, "pd"] = pd
    summary_a2e.loc[varname, "rope"] = r

for varname in varnames_a2e_g:
    print(varname)
    pd = p_direction(tr_a2e_g, varname)
    r = rope(tr_a2e_g, varname)
    summary_a2e_g.loc[varname, "pd"] = pd
    summary_a2e_g.loc[varname, "rope"] = r

# Save Arviz Summaries

summary_a2e.to_csv(join(figpath, "supplementary", f"summary_a2e.csv"))
summary_a2e_g.to_csv(join(figpath, "supplementary", f"summary_a2e_g.csv"))
