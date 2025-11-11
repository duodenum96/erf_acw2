import os
import pickle
import numpy as np
from erf_acw2.src import pick_megchans, re_epoch, loop_acw_acf, pklsave, pklload
import h5py
from scipy import stats
import pingouin as pg
import matplotlib.pyplot as plt
import mne
import os.path as op
from matplotlib.colors import LogNorm

# Import example data
raw_path = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing/sub-ON02747/rest/rejection_ica/sub-ON02747_rest_preprocessed-epo.fif.gz"
epochs = mne.read_epochs(raw_path)
epochs = pick_megchans(epochs)

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

acw_results = pklload(
    os.path.join(
        savepath, "supplementary_sensitivity_control_rest_acws_channel_space.pkl"
    )
)["acw_results"]
rms_results = pklload(
    os.path.join(savepath, "supplementary_sensitivity_control_task_rms_channel_space.pkl")
)["rms_results"] # "key": (nchan, n_param, n_sim), keys: "A_F", "A_B", "A_L", "gamma_1"

parameters = h5py.File(os.path.join(savepath, "parameters.jld2"), "r") 
A_F_values = parameters.get("A_F_values")[...]
A_B_values = parameters.get("A_B_values")[...]
A_L_values = parameters.get("A_L_values")[...]
gamma_1_values = parameters.get("gamma_1_values")[...]
parameters.close()

parameters_dict = {
    "A_F": A_F_values,
    "A_B": A_B_values,
    "A_L": A_L_values,
    "gamma_1": gamma_1_values,
}

parameter_names = ["A_F", "A_B", "A_L", "gamma_1"]

# Inside *_results, we have a dict with keys "A_F", "A_B", "A_L", "gamma_1"
# Inside each keys, there is  a list of length 20 denoting simulations
# Inside each element of the list there is a 1d array of shape 272 (n_channels, )

fontsize = 18

fancy_parameter_names = ["$ A_F $", "$ A_B $", "$ A_L $", r"$ \gamma_1 $"]

for i, i_parameter in enumerate(parameter_names):
    rms_values = np.array(rms_results[i_parameter])
    acw_values = np.array(acw_results[i_parameter]) # (nchan, n_param, n_sim)
    parameter_values = parameters_dict[i_parameter] # n_param

    nchan = rms_values.shape[0]
    n_param = rms_values.shape[1]
    n_sim = rms_values.shape[2]

    i_parameters = np.repeat(parameter_values[:, np.newaxis], n_sim, axis=1)

    r_values_acw_rms = np.zeros(nchan)
    p_values_acw_rms = np.zeros(nchan)

    r_values_parameter_acw = np.zeros(nchan)
    p_values_parameter_acw = np.zeros(nchan)

    r_values_parameter_rms = np.zeros(nchan)
    p_values_parameter_rms = np.zeros(nchan)

    for i_chan in range(nchan):
        rms_channel = rms_values[i_chan, :, :]
        acw_channel = acw_values[i_chan, :, :]

        corr_acw_rms = pg.corr(acw_channel.ravel(), rms_channel.ravel(), method="spearman")
        corr_parameter_acw = pg.corr(acw_channel.ravel(), i_parameters.ravel(), method="spearman")
        corr_parameter_rms = pg.corr(rms_channel.ravel(), i_parameters.ravel(), method="spearman")

        r_values_acw_rms[i_chan] = corr_acw_rms["r"]
        p_values_acw_rms[i_chan] = corr_acw_rms["p-val"]

        r_values_parameter_acw[i_chan] = corr_parameter_acw["r"]
        p_values_parameter_acw[i_chan] = corr_parameter_acw["p-val"]

        r_values_parameter_rms[i_chan] = corr_parameter_rms["r"]
        p_values_parameter_rms[i_chan] = corr_parameter_rms["p-val"]


    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ACW-RMS
    vmin = np.min(r_values_acw_rms)
    vmax = np.max(r_values_acw_rms)
    vminmax = np.max(np.abs([vmin, vmax]))
    im1, cn1 = mne.viz.plot_topomap(r_values_acw_rms, epochs.info, vlim=(-vminmax, vminmax), axes=ax[0], cmap="PiYG")
    ax[0].set_title("ACW-mERF Correlation", fontsize=fontsize)
    plt.colorbar(im1, ax=ax[0])

    p_values_corrected = pg.multicomp(p_values_acw_rms, method="fdr_bh")[1]
    mask = p_values_corrected < 0.05
    vmin = np.min(p_values_corrected)
    vmax = np.max(p_values_corrected)
    # im2, cn2 = mne.viz.plot_topomap(p_values_corrected, epochs.info, axes=ax[1], mask=mask, cnorm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    im2, cn2 = mne.viz.plot_topomap(p_values_corrected, epochs.info, axes=ax[1], mask=mask, vlim=(0, vmax), cmap="viridis")
    ax[1].set_title("p-values", fontsize=fontsize)
    cb = plt.colorbar(im2, ax=ax[1])
    ticks = cb.get_ticks()
    cb.set_ticks(np.append(ticks, 0.05))

    # Parameter-ACW
    # vmin = np.min(r_values_parameter_acw)
    # vmax = np.max(r_values_parameter_acw)
    # vminmax = np.max(np.abs([vmin, vmax]))
    # im3, cn3 = mne.viz.plot_topomap(r_values_parameter_acw, epochs.info, vlim=(-vminmax, vminmax), axes=ax[1, 0], cmap="PiYG")
    # ax[1, 0].set_title(f"{i_parameter}-ACW Correlation", fontsize=fontsize)
    # plt.colorbar(im3, ax=ax[1, 0])

    # p_values_corrected = pg.multicomp(p_values_parameter_acw, method="fdr_bh")[1]
    # mask = p_values_corrected < 0.05
    # vmin = np.min(p_values_corrected)
    # vmax = np.max(p_values_corrected)
    # im4, cn4 = mne.viz.plot_topomap(p_values_corrected, epochs.info, axes=ax[1, 1], mask=mask, cnorm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    # ax[1, 1].set_title("p-values", fontsize=fontsize)
    # plt.colorbar(im4, ax=ax[1, 1])

    # # Parameter-RMS
    # vmin = np.min(r_values_parameter_rms)
    # vmax = np.max(r_values_parameter_rms)
    # vminmax = np.max(np.abs([vmin, vmax]))
    # im5, cn5 = mne.viz.plot_topomap(r_values_parameter_rms, epochs.info, vlim=(-vminmax, vminmax), axes=ax[2, 0], cmap="PiYG")
    # ax[2, 0].set_title(f"{i_parameter}-RMS Correlation", fontsize=fontsize)
    # plt.colorbar(im5, ax=ax[2, 0])

    # p_values_corrected = pg.multicomp(p_values_parameter_rms, method="fdr_bh")[1]
    # mask = p_values_corrected < 0.05
    # vmin = np.min(p_values_corrected)
    # vmax = np.max(p_values_corrected)
    # im6, cn6 = mne.viz.plot_topomap(p_values_corrected, epochs.info, axes=ax[2, 1], mask=mask, cnorm=LogNorm(vmin=vmin+1e-30, vmax=vmax), cmap="viridis")
    # ax[2, 1].set_title("p-values", fontsize=fontsize)
    # plt.colorbar(im6, ax=ax[2, 1])

    plt.suptitle(f"{fancy_parameter_names[i]}", fontsize=fontsize)

    plt.savefig(op.join(savepath, f"correlations_on_channels_{i_parameter}_plain.png"), dpi=300, transparent=True)


