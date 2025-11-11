from erf_acw2.src import pklload
import numpy as np

taskname = "haririhammer"

induced_r2 = pklload(f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}/induced_r2_fooof.pkl")
n_tasks = len(induced_r2.keys())
r2_values = np.zeros((2, n_tasks))
for i_k, k in enumerate(list(induced_r2.keys())):
    for i in range(2):
        r2_values[i, i_k] = np.nanmedian(induced_r2[k][:, :, i])



