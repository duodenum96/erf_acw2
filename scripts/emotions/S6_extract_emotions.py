# pid: 6381
from erf_acw2.src import get_commonsubj, get_subjdatapath, get_subjpath, pklsave
import mne
from os.path import join as pathjoin
import pandas as pd
import numpy as np

taskname = "haririhammer"
subjs = get_commonsubj()

for i_subj in subjs:
    subjpath = get_subjpath(i_subj)
    datapath = get_subjdatapath(i_subj)
    data = mne.read_epochs(datapath)
    rawdata = mne.io.read_raw_ctf(get_subjdatapath(i_subj, preproc=False))

    events = mne.events_from_annotations(rawdata)
    events_ft = events[0]
    event_dict = events[1]

    annot = data.annotations.to_data_frame(time_format=None)
    nrow = len(annot)

    # I hate life
    c = 0
    trial = np.zeros(nrow, dtype="<U10")
    emotion = np.zeros(nrow, dtype="<U10")
    emo = "shape"
    for i in range(nrow):
        descr = annot["description"][i]
        if descr == "encode_face":
            c += 1
            i_onset = annot.iloc[i]["onset"]
            i_trial = annot[annot["onset"] == i_onset]
            if np.any(["happy" in i for i in i_trial["description"].to_numpy()]):
                emotion[annot["onset"] == i_onset] = "happy"
                emo = "happy"
            elif np.any(["sad" in i for i in i_trial["description"].to_numpy()]):
                emotion[annot["onset"] == i_onset] = "sad"
                emo = "sad"
        elif descr == "encode_shape":
            c += 1
            i_onset = annot.iloc[i]["onset"]
            i_trial = annot[annot["onset"] == i_onset]
            emotion[(annot["onset"] == i_onset).to_numpy()] = "shape"
            emo = "shape"
        elif descr == "probe_face":
            i_onset = annot.iloc[i]["onset"]
            emotion[(annot["onset"] == i_onset).to_numpy()] = emo
        elif descr == "probe_shape":
            i_onset = annot.iloc[i]["onset"]
            emotion[(annot["onset"] == i_onset).to_numpy()] = emo
        else:
            i_onset = annot.iloc[i]["onset"]
            emotion[(annot["onset"] == i_onset).to_numpy()] = emo
        
        print(c)
        print(descr)
        print(emo)
        

    annot["emotion"] = emotion
    annot["trial"] = trial
    
    new_descr = []
    for i in range(nrow):
        if "shape" not in annot.iloc[i]["emotion"]:
            i_new_descr = f"{annot.iloc[i]["description"]}_{annot.iloc[i]["emotion"]}"
            new_descr.append(i_new_descr)
        else:
            new_descr.append(f"{annot.iloc[i]["description"]}")
    

    uniqnames = np.unique(new_descr)
    new_descr_dict = {i: i_name for i, i_name in enumerate(uniqnames)}
    new_descr_dict_inv = {i_name: i for i, i_name in enumerate(uniqnames)}

    events_ft_new = events_ft.copy()
    for i in range(nrow):
        events_ft_new[i, 2] = new_descr_dict_inv[new_descr[i]]

    new_annotations = mne.annotations_from_events(events_ft_new, data.info["sfreq"], event_desc=new_descr_dict)

    data_new = data.copy()
    data_new.set_annotations(new_annotations)

    data_new.save(pathjoin(subjpath, f"{i_subj}_{taskname}_preprocessed_emo-epo.fif"), 
              overwrite=True)
    pklsave(pathjoin(subjpath, f"{i_subj}_{taskname}_events_emo.pkl"),
            {"events_ft": events_ft_new, "event_desc": new_descr_dict})

print("DONE")
