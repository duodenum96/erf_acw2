# run as python haririhammer_segmentblocks_eventrelated $i
# $i goes from 0 to 57
import numpy as np
import mne
import sys
import os
from os.path import join as pathjoin
import time
from erf_acw2.src import subjs, pick_megchans, get_commonsubj, pklload

taskname = "haririhammer"

subjs_common = get_commonsubj()
i = int(sys.argv[1])
i_subj = subjs_common[i]
nsubj = len(subjs_common)

print(f"Starting subj: {i_subj}")
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "haririhammer", "rejection_ica")
pp_filename = pathjoin(
    outputpath, i_subj + "_haririhammer_preprocessed_emo-epo.fif"
)

tmp = mne.read_epochs(pp_filename)
epochs = pick_megchans(tmp)
#######################
# start f'n
ntrial = len(epochs.drop_log)
nchan = len(epochs.info.get_channel_types())
n_timepoints = epochs.get_data().shape[2]
all_data = np.zeros((ntrial, nchan, n_timepoints))

all_data[epochs.selection, :, :] = epochs.get_data()
all_data[np.setdiff1d(np.arange(ntrial), epochs.selection), :, :] = np.nan

pseudo_raw_data = np.hstack(all_data)
re_raw = mne.io.RawArray(pseudo_raw_data, epochs.info)

events = pklload(pathjoin(outputpath, f"{i_subj}_{taskname}_events_emo.pkl"))
event_dict = events["event_desc"]

# Create the new epoch object based on events
epoch_start_names = ["encode_face_happy", "probe_face_happy", 
                     "encode_face_sad", "probe_face_sad",
                     "encode_shape", "probe_shape"]
epoch_start_codes = []
for i in epoch_start_names:
    for key, value in event_dict.items():
        if value == i:
            epoch_start_codes.append(key)

# ugly ass code
idx = (
    (events["events_ft"][:, 2] == epoch_start_codes[0])
    | (events["events_ft"][:, 2] == epoch_start_codes[1])
    | (events["events_ft"][:, 2] == epoch_start_codes[2])
    | (events["events_ft"][:, 2] == epoch_start_codes[3])
    | (events["events_ft"][:, 2] == epoch_start_codes[4])
    | (events["events_ft"][:, 2] == epoch_start_codes[5])
)

events_for_epoch = events["events_ft"][idx, :]

event_id = {epoch_start_names[i]: epoch_start_codes[i] for i in range(len(epoch_start_codes))}

# Segregate responses from events
re_epoch = mne.Epochs(
    re_raw, events_for_epoch, tmin=-0.3, tmax=0.7, event_id=event_id, preload=True, baseline=(None, 0)
)

ntrial = re_epoch.events.shape[0]
# Mark the nans as not selected
nanidx = np.isnan(re_epoch.get_data())
selection = np.where(~np.any(nanidx, axis=(1, 2)))[0]

re_epoch = re_epoch[selection]

final_filename = pathjoin(
    outputpath, i_subj + "_haririhammer_preprocessed_erf_emo-epo.fif"
)

re_epoch.save(final_filename, fmt="double", overwrite=True)

event_filename = pathjoin(
    outputpath, i_subj + "_haririhammer_events.eve"
)

mne.write_events(event_filename, events["events_ft"], overwrite=True)
print("DONE")
