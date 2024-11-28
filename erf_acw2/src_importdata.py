# Helper functions for data wrangling / wrestling
import numpy as np
from os.path import join as pathjoin
from erf_acw2.src import (
    pklload,
    get_commonsubj,
    badchan_padnan_2d,
    badchan_padnan
)
import pandas as pd
from pathlib import Path
import mne

def get_template_bad(good=False):
    taskname = "haririhammer"
    subjs_common = get_commonsubj()
    tasknames = ["encode_face_happy", "probe_face_happy", 
             "encode_face_sad", "probe_face_sad", 
             "encode_shape", "probe_shape"]

    tasks = {i: [] for i in tasknames}

    for i, i_subj in enumerate(subjs_common):
        filename = pathjoin(
            f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}",
            f"{i_subj}_{taskname}_erf_emo.pkl",
        )
        epochsdict = pklload(filename)
        for j in tasknames:
            tasks[j].append(epochsdict[j])

    if not good:
        # Equalize channels, there is one bad channel in some subjects that is messing up things everything. Unfortunately we need to 
        # take it out from all subjects.
        template_bad = tasks["encode_face_happy"][17]
        for i, i_subj in enumerate(subjs_common):
            for j in tasknames:
                tasks[j][i] = mne.channels.equalize_channels([template_bad, tasks[j][i]])[1]

        template_bad = tasks["encode_face_happy"][17]
        return template_bad
    else:
        return tasks["encode_face_happy"][0]

    
def erf_pick_all_hh(
    erfdata, timerange
):
    """
    Sloppy code to get all ERFs regardless of task activation
    """
    
    times = erfdata["encode_happy"][0].times
    times_after_stimulation = (times > timerange[0]) & (times < timerange[1])
    keys = erfdata.keys()

    all_erf = np.zeros((56, 272, len(keys)))
    for i, key in enumerate(keys):
        for j in range(56):
            i_chlist = erfdata[key][j].info["ch_names"]
            data = erfdata[key][j].get_data()[:, times_after_stimulation]

            erf = badchan_padnan(i_chlist, 
                np.sqrt(
                    np.nanmean(
                        data ** 2, axis=1
                    )
                )
            )
            all_erf[j, :, i] = erf

    return all_erf

