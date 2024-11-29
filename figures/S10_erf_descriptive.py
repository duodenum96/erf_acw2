##### Calculate grand average
import numpy as np
import mne
import os
from os.path import join as pathjoin
import matplotlib.pyplot as plt
import matplotlib
from erf_acw2.src import subjs, pklsave, pklload, get_commonsubj

figfolder = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure2"
os.chdir("/BICNAS2/ycatal/erf_acw2/")

taskname = "haririhammer"
subjs_common = get_commonsubj()

all_encode_face_happy = []
all_probe_face_happy = []
all_encode_face_sad = []
all_probe_face_sad = []
all_encode_shape = []
all_probe_shape = []
for i, i_subj in enumerate(subjs_common):
    filename = pathjoin(
        f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}",
        f"{i_subj}_{taskname}_erf_emo.pkl",
    )
    epochsdict = pklload(filename)
    all_encode_face_happy.append(epochsdict["encode_face_happy"])
    all_probe_face_happy.append(epochsdict["probe_face_happy"])
    all_encode_face_sad.append(epochsdict["encode_face_sad"])
    all_probe_face_sad.append(epochsdict["probe_face_sad"])
    all_encode_shape.append(epochsdict["encode_shape"])
    all_probe_shape.append(epochsdict["probe_shape"])

encode_face_happy_ga = mne.grand_average(all_encode_face_happy)
probe_face_happy_ga = mne.grand_average(all_probe_face_happy)
encode_face_sad_ga = mne.grand_average(all_encode_face_sad)
probe_face_sad_ga = mne.grand_average(all_probe_face_sad)
encode_shape_ga = mne.grand_average(all_encode_shape)
probe_shape_ga = mne.grand_average(all_probe_shape)

erfs = [encode_face_happy_ga, probe_face_happy_ga,
        encode_face_sad_ga, probe_face_sad_ga,
        encode_shape_ga, probe_shape_ga]
erfnames = ["Encode Face Happy", "Probe Face Happy", 
            "Encode Face Sad", "Probe Face Sad", 
            "Encode Shape", "Probe Shape"]

erf_savenames = ["encode_face_happy", "probe_face_happy", 
                 "encode_face_sad", "probe_face_sad", 
                 "encode_shape", "probe_shape"]

matplotlib.rcParams.update({"font.size": 14})
for i, i_erf in enumerate(erfs):
    plt.close()
    plot = i_erf.plot_joint(topomap_args={"cmap": "PiYG"})
    [j.set_fontsize(8) for j in plot.get_children()[-2].get_yticklabels()]
    plt.suptitle(erfnames[i])
    plt.savefig(pathjoin(figfolder, f"{taskname}_fig2a_{erf_savenames[i]}.jpg"), dpi=800)

# Create Schematic Figure
plt.rcParams.update({"font.size": 28, "axes.labelweight": "bold"})
i_erf = erfs[0]
# erf_np = np.mean(i_erf.data, axis=0)
erf_np = i_erf.data[15, :]

f, ax = plt.subplots(figsize=(8,8))

ax.plot(erf_np, alpha=1.0, linewidth=4, color="k")
ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel("Time")
ax.set_ylabel("Event-Related Activity")
# ax.set_xlim(0, 0.25)
# hide x and y ticks
ax.set_xticks([])
ax.set_yticks([])


f.savefig(pathjoin(figfolder, "erf_1d.png"), dpi=800, transparent=True)

