import os
import pickle
import numpy as np
from erf_acw2.src import pick_megchans, re_epoch, loop_acw_acf, pklsave, pklload
import h5py

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

acw_results = pklload(
    os.path.join(
        savepath, "supplementary_sensitivity_control_rest_acws_channel_space.pkl"
    )
)["acw_results"]
rms_results = pklload(
    os.path.join(savepath, "supplementary_sensitivity_control_task_rms.pkl")
)["rms_results"]

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

# Inside *_results, we have a dict with keys "A_F", "A_B", "A_L", "gamma_1"
# Inside each keys, there is  a list of length 20 denoting simulations
# Inside each element of the list there is a 1d array of shape 272 (n_channels, )

i_parameter = "A_F"
rms_values = np.array(rms_results[i_parameter])
acw_values = np.array(acw_results[i_parameter])
parameter_values = parameters_dict[i_parameter]
