# cd /BICNAS2/ycatal/erf_acw2/scripts/modeling/
# nohup python SMR3_project_to_MEG_task.py > log/SMR3_project_to_MEG_task.log 2>&1 &
# echo $! > log/SMR3_project_to_MEG_task.pid
import mne
import numpy as np
import h5py
import os
import multiprocessing as mp
from erf_acw2.src import pick_megchans, re_epoch, loop_acw_acf, pklsave

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_reviewer_comments"
savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

# Load resting state ys
task_ys = h5py.File(os.path.join(savepath, "supplementary_sensitivity_control_task_ys.jld2"), "r")
A_F = task_ys.get("A_F")[...]
A_B = task_ys.get("A_B")[...]
A_L = task_ys.get("A_L")[...]
gamma_1 = task_ys.get("gamma_1")[...]
task_ys.close()

ngamma, narea, ntime, nsim = A_F.shape

fs = 1200.0
subject = "sub-ON02747"

raw_path = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing/sub-ON02747/rest/rejection_ica/sub-ON02747_rest_preprocessed-epo.fif.gz"
epochs = mne.read_epochs(raw_path)
epochs = pick_megchans(epochs)
info = epochs.info

tstops = np.arange(5, 100, 5)
event_onsets = tstops * int(fs)
events = np.zeros((len(event_onsets), 3), dtype=int)
events[:, 0] = event_onsets
events[:, 2] = 1
events = events.astype(int)

tstart = -0.3
tend = 0.7

forward_path = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing/sub-ON02747/rest/forward-fwd.fif"
fwd = mne.read_forward_solution(forward_path)
src = fwd["src"]

selected_label_1 = mne.read_labels_from_annot(
    subject, regexp="parsopercularis", subjects_dir=subjects_dir
)[0]
selected_label_2 = mne.read_labels_from_annot(
    subject, regexp="parstriangularis", subjects_dir=subjects_dir
)[0]
other_labels = mne.read_labels_from_annot(
    subject, subjects_dir=subjects_dir
)

location = "center"  # Use the center of the region as a seed.
extent = 10.0  # Extent in mm of the region.
label1 = mne.label.select_sources(
    subject, selected_label_1, location=location, extent=extent, subjects_dir=subjects_dir
)
label2 = mne.label.select_sources(
    subject, selected_label_2, location=location, extent=extent, subjects_dir=subjects_dir
)
remaining_labels = []
for i in other_labels:
    if (i.name is not label1.name) and (i.name is not label2.name):
        remaining_labels.append(
            mne.label.select_sources(
               subject, i, location=location, extent=extent, subjects_dir=subjects_dir
            )
        )

tstep = 1 / fs
nchan = info["nchan"]


def process_parameter(args):
    """Worker function to process a single parameter in parallel."""
    k_parameter, dataset, j_sim, src, tstep, label1, label2, remaining_labels, info, events, fwd, fs, tstart, tend = args
    
    source_time_series_1 = dataset[k_parameter, 0, :, j_sim]
    source_time_series_2 = dataset[k_parameter, 1, :, j_sim]
    
    epoch_length_samples = int((tend-tstart) * fs) 
    minus_sample = int((0 - tstart) * fs)
    plus_sample = int((tend - 0) * fs)
    onsets = events[:, 0]

    n_epochs = events.shape[0]

    # Reshape to epochs (n_epochs, epoch_length_samples)
    source_time_series_epochs_1 = np.array([source_time_series_1[(i-minus_sample):(i+plus_sample)] for i in onsets])
    source_time_series_epochs_2 = np.array([source_time_series_2[(i-minus_sample):(i+plus_sample)] for i in onsets])

    prestim_times = np.arange(tstart, 0, tstep)
    poststim_times = np.arange(0, tend, tstep)

    source_evoked_1 = np.mean(source_time_series_epochs_1, axis=0)
    source_evoked_2 = np.mean(source_time_series_epochs_2, axis=0)
    noise_variance_1 = np.var(source_evoked_1[:len(prestim_times)])
    noise_variance_2 = np.var(source_evoked_2[:len(prestim_times)])

    source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
    source_simulator.add_data(label1, source_time_series_epochs_1, events)
    source_simulator.add_data(label2, source_time_series_epochs_2, events)

    for j in remaining_labels:
        source_simulator.add_data(j, noise_variance_1 * np.random.randn(*source_time_series_epochs_2.shape), events)

    raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd, n_jobs=1)
    cov = mne.make_ad_hoc_cov(raw.info)

    mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])

    sim_epochs = mne.Epochs(raw, events, 1, tmin=0.3, tmax=tend, baseline=(0.3,0.3))
    evoked = sim_epochs.average()
    rms = np.mean(np.sqrt(evoked.get_data()**2), axis=1)
    
    return k_parameter, rms


############################## For loops start here ##############################

# Initialize dictionary to store results
rms_results = {}
dataset_names = ['A_F', 'A_B', 'A_L', 'gamma_1']
datasets = [A_F, A_B, A_L, gamma_1]

# Determine number of processes (conservative approach)
n_processes = ngamma

# Loop over each dataset
for dataset_idx, (dataset_name, dataset) in enumerate(zip(dataset_names, datasets)):
    print(f"Processing {dataset_name}...")
    rms_results[dataset_name] = np.zeros((nchan, ngamma, nsim))
    
    # Loop over simulations
    for j_sim in range(nsim):
        print(f"  Simulation {j_sim+1}/{nsim}")
        
        # Prepare arguments for parallel processing
        args_list = []
        for k_parameter in range(ngamma):
            args_list.append((k_parameter, dataset, j_sim, src, tstep, label1, label2, remaining_labels, info, events, fwd, fs, tstart, tend))
        
        # Process parameters in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(process_parameter, args_list)
        
        # Store results
        for k_parameter, rms in results:
            rms_results[dataset_name][:, k_parameter, j_sim] = rms
            print(f"    Parameter {k_parameter+1}/{ngamma} completed")
        
        print(f"j_sim: {j_sim} done")
    print(f"dataset: {dataset_name} done")

        
pklsave(os.path.join(savepath, "supplementary_sensitivity_control_task_rms_channel_space.pkl"), 
        {"rms_results": rms_results})

print("done")