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

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_f1"


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
    "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/sensitivity_control_pythonic.jld2",
    "r",
)
acws_A_F = f.get("acws_A_F")[...]
acws_A_B = f.get("acws_A_B")[...]
acws_A_L = f.get("acws_A_L")[...]
acws_gamma_1 = f.get("acws_gamma_1")[...]
erfs_A_F = f.get("erfs_A_F")[...]
erfs_A_B = f.get("erfs_A_B")[...]
erfs_A_L = f.get("erfs_A_L")[...]
erfs_gamma_1 = f.get("erfs_gamma_1")[...]
A_F_values = f.get("A_F_values")[...]
A_B_values = f.get("A_B_values")[...]
A_L_values = f.get("A_L_values")[...]
gamma_1_values = f.get("gamma_1_values")[...]
f.close()

areas = ["Area 1", "Area 2"]
params = ["A_F", "A_B", "A_L", "gamma_1"]

narea, nvalue, nsim = acws_A_F.shape

A_F_values2 = np.transpose(np.tile(A_F_values, (narea, nsim, 1)), (0, 2, 1))
assert np.all(A_F_values2[0, :, 0] == A_F_values)
A_B_values2 = np.transpose(np.tile(A_B_values, (narea, nsim, 1)), (0, 2, 1))
A_L_values2 = np.transpose(np.tile(A_L_values, (narea, nsim, 1)), (0, 2, 1))
gamma_1_values2 = np.transpose(np.tile(gamma_1_values, (narea, nsim, 1)), (0, 2, 1))

measures = ["ACW", "ERF"]

area_x_param = np.array(
    [[i + " " + j + " " + k] for i in areas for j in params for k in measures]
).flatten()
# %%
df_results = pd.DataFrame(
    {
        "area_x_param": area_x_param,
        "pval": np.zeros(len(area_x_param)),
        "rho": np.zeros(len(area_x_param)),
    }
)

all_acws = [acws_A_F, acws_A_B, acws_A_L, acws_gamma_1]
all_erfs = [erfs_A_F, erfs_A_B, erfs_A_L, erfs_gamma_1]
all_values = [A_F_values2, A_B_values2, A_L_values2, gamma_1_values2]

for i_area, area in enumerate(areas):
    for i_param, param in enumerate(params):
        df_idx = area + " " + param
        res_acw = pg.corr(
            all_acws[i_param][i_area, :, :].flatten(),
            all_values[i_param][i_area, :, :].flatten(),
            method="spearman",
        )
        df_results.loc[df_results["area_x_param"] == df_idx + " ACW", "pval"] = res_acw[
            "p-val"
        ].values[0]
        df_results.loc[df_results["area_x_param"] == df_idx + " ACW", "rho"] = res_acw[
            "r"
        ].values[0]

        res_erf = pg.corr(
            all_erfs[i_param][i_area, :, 0:nsim].flatten(),
            all_values[i_param][i_area, :, :].flatten(),
            method="spearman",
        )
        df_results.loc[df_results["area_x_param"] == df_idx + " ERF", "pval"] = res_erf[
            "p-val"
        ].values[0]
        df_results.loc[df_results["area_x_param"] == df_idx + " ERF", "rho"] = res_erf[
            "r"
        ].values[0]

# %%
df_results["pval_corrected"] = pg.multicomp(
    df_results["pval"].values, method="bonferroni"
)[1]

df_results["pval_str"] = df_results["pval_corrected"].apply(p2str)

# print df_results to 3 decimal places
print(df_results.round(3))

# %% Scatter plot
# columns: ACW area 1, acw area 2, erf area 1, erf area 2
# Rows: A_F, A_B, A_L, gamma_1

params_ltx = ["$A_F$", "$A_B$", "$A_L$", "$\\gamma_1$"]
measure_names = ["ACW (s)", "mERF"]
full_names = ["Feedforward", "Feedback", "Lateral", "Intracolumnar"]

cm = 1 / 2.54  # centimeters in inches

all_measures = [all_acws, all_erfs]


f, ax = plt.subplots(
    narea * len(measures),
    len(params),
    figsize=(21 * cm, 12.4 * cm),
    layout="constrained",
)
f.subplots_adjust(hspace=0.9, wspace=0.9)

for i_area, area in enumerate(areas):
    for i_param, param in enumerate(params):
        for i_measure, measure in enumerate(measures):
            subtitle = area + " " + params_ltx[i_param] + " " + measure
            df_idx = area + " " + param + " " + measure
            row_idx = i_area * len(measures) + i_measure
            x = all_values[i_param][i_area, :, :].flatten()
            y = all_measures[i_measure][i_param][i_area, :, 0:nsim].flatten()
            ax[row_idx, i_param].scatter(
                x,
                y,
                alpha=0.2,
                color="black",
                s=3,
            )
            tmp = pd.DataFrame({"x": x, "y": y}).dropna()
            p = np.polyfit(tmp["x"], tmp["y"], 1)
            ax[row_idx, i_param].plot(
                tmp["x"],
                np.polyval(p, tmp["x"]),
                color="tab:red",
                linewidth=1,
            )

            if row_idx == 0:
                ax[row_idx, i_param].set_title(full_names[i_param])

            if i_param == 0:
                ax[row_idx, i_param].set_ylabel(area + "\n" + measure_names[i_measure])

            xticklabels = (
                np.arange(
                    np.min(all_values[i_param][i_area, :, :].flatten()),
                    np.max(all_values[i_param][i_area, :, :].flatten()),
                    5,
                )
            )

            if row_idx == 3:
                ax[row_idx, i_param].set_xlabel(params_ltx[i_param])
                ax[row_idx, i_param].set_xticks(xticklabels, [f"{i:.0f}" for i in xticklabels], rotation=45)
            else:
                ax[row_idx, i_param].set_xticks(xticklabels, [])

            star = df_results.loc[
                df_results["area_x_param"] == df_idx, "pval_str"
            ].values[0]
            r = df_results.loc[df_results["area_x_param"] == df_idx, "rho"].values[0]

            statstring = f"r = {r:.2f}{star}"
            ax[row_idx, i_param].text(
                0.05,
                0.95,
                statstring,
                transform=ax[row_idx, i_param].transAxes,
                fontsize=8,
                verticalalignment="top",
                color="darkolivegreen",
                fontweight="bold",
            )
            ax[row_idx, i_param].spines["right"].set_visible(False)
            ax[row_idx, i_param].spines["top"].set_visible(False)

f.savefig(join(figpath, "supplementary_sensitivity_control.jpg"), dpi=300)
