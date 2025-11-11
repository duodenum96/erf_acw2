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

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_f2"


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
nsim = acw50s.shape[1]

vars = ["acw50s", "erfs"]
data = pd.DataFrame(
    {
        "ACW": acw50s.ravel(),
        "Color:\n$\\gamma_1$": np.tile(gamma1.astype(str), [nsim, 1]).T.ravel(),
        "ERF": erfs.ravel(),
    }
)

corr_results = pg.pairwise_corr(data, padjust="fdr_bh")

xs = ["ACW"]
ys = ["ERF"]
rho = corr_results["r"][0]
p = p2str(corr_results["p-unc"][0])

# plot = (
#     so.Plot(data)
#     .layout(size=(4, 4))
#     .pair(x=["ACW"], y=["ERF"])
#     .add(so.Dot(), color="Color:\n$\\gamma_1$")
#     .add(so.Line(color="black"), so.PolyFit(order=1))
#     .label(x="ACW (s)", y="mERF")
#     .theme({"axes.labelsize": 16})
#     .save(join(figpath, "variable_corrs.jpg"), dpi=800)
# )

###########################################################################################

# Create scatterplot with matplotlib and colorbar
fig, ax = plt.subplots(figsize=(6, 4))

# Create the scatter plot with gamma values as colors
gamma_values = np.tile(gamma1, [nsim, 1]).T.ravel()
acw_values = acw50s.ravel()
erf_values = erfs.ravel()

scatter = ax.scatter(acw_values, erf_values, c=gamma_values, 
                    cmap='viridis', alpha=0.7, s=20)
ax.grid(False)
# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('$\\gamma_1$', fontsize=16)

# Add regression line
z = np.polyfit(acw_values, erf_values, 1)
p = np.poly1d(z)
ax.plot(acw_values, p(acw_values), "k-", alpha=0.8, linewidth=2)

# Set labels and styling
ax.set_xlabel('ACW (s)', fontsize=16)
ax.set_ylabel('mERF', fontsize=16)
ax.grid(False)
ax.tick_params(labelsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adjust layout and save
plt.tight_layout()
plt.savefig(join(figpath, "variable_corrs_tmp.png"), dpi=300, bbox_inches='tight', transparent=True)

###########################################################################################

corr_results = pg.pairwise_corr(data, padjust="fdr_bh")
# ACW - ERF: r = 0.517***

################## Plot gamma / variables ####################
data = pd.DataFrame(
    {
        "ACW": acw50s.ravel(),
        "ERF": erfs.ravel(),
        "$\\gamma_1$": np.tile(gamma1, [nsim, 1]).T.ravel(),
        "Color:\n$\\gamma_1$": np.tile(gamma1_cat, [nsim, 1]).T.ravel(),
    }
)

data = data.rename(columns={"ACW": "ACW (s)", "ERF": "mERF"})

(
    so.Plot(data, x="$\\gamma_1$")
    .layout(size=(8, 4))
    .pair(y=["ACW (s)", "mERF"], wrap=1)
    .add(so.Dot(), color="Color:\n$\\gamma_1$")
    .add(so.Line(color="black"), so.PolyFit(order=1))
    .theme({"axes.labelsize": 16, "axes.grid": False, "axes.spines.right": False, "axes.spines.top": False})
    .save(join(figpath, "gamma_corrs.jpg"))
)
# (
#     so.Plot(data, x="$\\gamma_1$")
#     .layout(size=(8, 4))
#     .pair(y=["ACW (s)", "mERF"], wrap=1)
#     .add(so.Dot(), color="Color:\n$\\gamma_1$")
#     .add(so.Line(color="black"), so.PolyFit(order=1))
#     .theme({"axes.labelsize": 16})
#     .save(join(figpath, "gamma_corrs.jpg"), dpi=800)
# )

################################################################################################
# Create matplotlib version of gamma correlations plot with shared colorbar

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create matplotlib version of gamma correlations plot with shared colorbar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Extract data
gamma_values = np.tile(gamma1, [nsim, 1]).T.ravel()
acw_values = acw50s.ravel()
erf_values = erfs.ravel()

# First subplot: ACW vs gamma1
scatter1 = ax1.scatter(gamma_values, acw_values, c=gamma_values, 
                      cmap='viridis', alpha=0.7, s=20)

# Add regression line for ACW
z1 = np.polyfit(gamma_values, acw_values, 1)
p1 = np.poly1d(z1)
ax1.plot(gamma_values, p1(gamma_values), "k-", alpha=0.8, linewidth=2)

# Style first subplot
ax1.set_xlabel('$\\gamma_1$', fontsize=16)
ax1.set_ylabel('ACW (s)', fontsize=16)
ax1.grid(False)
ax1.tick_params(labelsize=12)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)


# Second subplot: mERF vs gamma1
scatter2 = ax2.scatter(gamma_values, erf_values, c=gamma_values, 
                      cmap='viridis', alpha=0.7, s=20)

# Add regression line for mERF
z2 = np.polyfit(gamma_values, erf_values, 1)
p2 = np.poly1d(z2)
ax2.plot(gamma_values, p2(gamma_values), "k-", alpha=0.8, linewidth=2)

# Style second subplot
ax2.set_xlabel('$\\gamma_1$', fontsize=16)
ax2.set_ylabel('mERF', fontsize=16)
ax2.grid(False)
ax2.tick_params(labelsize=12)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Create properly sized colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(scatter2, cax=cax)
cbar.set_label('$\\gamma_1$', fontsize=14)

# Adjust layout and save
plt.tight_layout()
plt.savefig(join(figpath, "gamma_corrs_tmp.png"), dpi=800, bbox_inches='tight', transparent=True)

#################################################################################################

pg.pairwise_corr(data, [["$\gamma_1$"], ["ACW (s)", "mERF"]], padjust="fdr_bh")

# ACW - gamma1: 0.578***
# ERF - gamma1: 0.904***
# n = 1240 in all correlations

######## Now, we'll do a 1 x 3 plot. In columns, we'll have the 3 different gamma1 values, and in rows, we'll have the scatterplot between 2 variables.
example_gamma1s = [40, 50, 60]
data_example = data[data["$\\gamma_1$"].isin(example_gamma1s)]

g = (
    so.Plot(data_example, x="ACW (s)", y="mERF", color="Color:\n$\\gamma_1$")
    .facet(col="$\\gamma_1$")
    .add(so.Dot(color="black"))
    .add(so.Line(color="black"), so.PolyFit(order=1))
    .label(title="$\\gamma_1$ = {}".format)
    .layout(size=(12, 4))
    .theme({"axes.grid": False, "axes.spines.right": False, "axes.spines.top": False})
)
plt.legend("off")
g.save(join(figpath, "gamma1_scatterplots.jpg"), dpi=800)

# To calculate correlations, make each gamma1 value a seperate column, then put it into pg.pairwise_corr
data_example_corr = []
for gamma in example_gamma1s:
    subset = data_example[data_example["$\\gamma_1$"] == gamma]
    corr = pg.corr(subset["ACW (s)"], subset["mERF"])
    data_example_corr.append({
        'gamma1': gamma,
        'r': corr['r'].iloc[0],
        'p-val': corr['p-val'].iloc[0]
    })

data_example_corr = pd.DataFrame(data_example_corr)
# Do correction for multiple comparisons
data_example_corr["p-cor"] = pg.multicomp(data_example_corr["p-val"], alpha=0.05, method="fdr_bh")[1]
print(data_example_corr)
# Correlations:
# gamma1 = 40: r = -0.0.058, p = 0.719
# gamma1 = 50: r = 0.096, p = 0.719
# gamma1 = 60: r = -0.278, p = 0.248
