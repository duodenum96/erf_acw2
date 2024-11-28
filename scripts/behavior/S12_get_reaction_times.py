import os
from os.path import join as pathjoin
import numpy as np
from erf_acw2.src import pklsave, get_commonsubj
import mne

os.chdir("/BICNAS2/ycatal/erf_acw2/")

######################### Prelude  ##############################

taskname = "haririhammer"

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

#################################################################
correct_or_wrong = [] # 1 or 0
reaction_time = [] # how long it took
accuracy = [] # percentage of correct hits
speed = [] # this is only for correct hits
durations_all_subj = [] # durations of every subject
for i in range(nsubj):
    i_subj = subjs_common[i]

    preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
    subj_preprocpath = pathjoin(preprocpath, i_subj)
    outputpath = pathjoin(subj_preprocpath, "haririhammer", "rejection_ica")

    datadir = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
    subjdir = os.path.join(datadir, i_subj, "ses-01", "meg")
    rawdata = mne.io.read_raw_ctf(
        pathjoin(subjdir, i_subj + "_ses-01_task-haririhammer_run-01_meg.ds"), preload=False
    )
    events = mne.events_from_annotations(rawdata)
    eventdict = events[1]
    i_event = events[0]
    
    markers = i_event[:, 2]
    times = i_event[:, 0]

    hit_m = eventdict["response_hit"]
    hit_left = eventdict["response_l"]
    hit_right = eventdict["response_r"]

    accuracy.append(
        np.sum(markers == hit_m) / np.sum((markers == hit_left) | (markers == hit_right))
    )
    
    # An inefficient algorithm to calculate average speed of correct hits
    face_m = eventdict["probe_face"]
    shape_m = eventdict["probe_shape"]
    durations = []
    for j, j_marker in enumerate(markers):
        if (j_marker == face_m) | (j_marker == shape_m):
            k = 1
            while True:
                if (j+k) == markers.shape[0]:
                    break
                elif markers[j+k] == hit_m:
                    durations.append(times[j+k] - times[j])
                    break
                elif (markers[j+k] == face_m) | (markers[j+k] == shape_m):
                    break
                else:
                    k += 1
                    
    durations_all_subj.append(durations)
    speed.append(np.mean(durations) / rawdata.info["sfreq"])

filename = "/BICNAS2/ycatal/erf_acw2/results/behavior/haririhammer_beh_measures.pkl"
pklsave(filename, {"speed": speed, "accuracy": accuracy, "durations_all_subj": durations_all_subj})

