# cd /BICNAS2/ycatal/erf_acw2/scripts/modeling/
# nohup python SMR2_project_to_MEG_rest_parallel.py > log/SMR2_project_to_MEG_rest_parallel.log 2>&1 &
# echo $! > log/SMR2_project_to_MEG_rest_parallel.pid
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
rest_ys = h5py.File(os.path.join(savepath, "supplementary_sensitivity_control_rest_ys.jld2"), "r")
A_F = rest_ys.get("A_F")[...]
A_B = rest_ys.get("A_B")[...]
A_L = rest_ys.get("A_L")[...]
gamma_1 = rest_ys.get("gamma_1")[...]
rest_ys.close()

ngamma, narea, ntime, nsim = A_F.shape

fs = 1200.0
subject = "sub-ON02747"

raw_path = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing/sub-ON02747/rest/rejection_ica/sub-ON02747_rest_preprocessed-epo.fif.gz"
epochs = mne.read_epochs(raw_path)
epochs = pick_megchans(epochs)
info = epochs.info
events = epochs.events

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


def process_parameter(args):
    """Worker function to process a single parameter in parallel."""
    k_parameter, dataset, j_sim, src, tstep, label1, label2, remaining_labels, info, events, fwd, fs = args
    
    source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)

    source_time_series_1 = dataset[k_parameter, 0, :, j_sim]
    source_time_series_2 = dataset[k_parameter, 1, :, j_sim]
    
    # Chunk source_time_series to 3 second epochs
    epoch_length_samples = int(3.0 * fs)  # 3 seconds * sampling rate
    n_epochs = len(source_time_series_1) // epoch_length_samples

    # Reshape to epochs (n_epochs, epoch_length_samples)
    source_time_series_epochs_1 = source_time_series_1[:n_epochs * epoch_length_samples].reshape(n_epochs, epoch_length_samples)
    source_time_series_epochs_2 = source_time_series_2[:n_epochs * epoch_length_samples].reshape(n_epochs, epoch_length_samples)

    events_trimmed = events[:n_epochs]
    noise_variance_1 = np.mean(np.var(source_time_series_epochs_1))
    noise_variance_2 = np.mean(np.var(source_time_series_epochs_2))

    source_simulator.add_data(label1, source_time_series_epochs_1, events_trimmed)
    source_simulator.add_data(label2, source_time_series_epochs_2, events_trimmed)

    for j in remaining_labels:
        source_simulator.add_data(j, noise_variance_1 * np.random.randn(*source_time_series_epochs_2.shape), events_trimmed)

    raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd, n_jobs=1)
    cov = mne.make_ad_hoc_cov(raw.info)

    mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])

    sim_epochs = mne.Epochs(raw, events_trimmed, 1, tmin=0.0, tmax=3.0, baseline=(0,0))
    sim_epochs_10sec = re_epoch(sim_epochs, 10.0)
    ntrial = sim_epochs_10sec.get_data().shape[0]

    acws, _ = loop_acw_acf(sim_epochs_10sec, ntrial=ntrial, nlags=2000)
    
    return k_parameter, acws


############################## For loops start here ##############################

# Initialize dictionary to store results
acw_results = {}
dataset_names = ['A_F', 'A_B', 'A_L', 'gamma_1']
datasets = [A_F, A_B, A_L, gamma_1]
nchan = info["nchan"]

# Determine number of processes (conservative approach)
n_processes = ngamma

# Loop over each dataset
for dataset_idx, (dataset_name, dataset) in enumerate(zip(dataset_names, datasets)):
    print(f"Processing {dataset_name}...")
    acw_results[dataset_name] = np.zeros((nchan, ngamma, nsim))
    
    # Loop over simulations
    for j_sim in range(nsim):
        print(f"  Simulation {j_sim+1}/{nsim}")
        
        # Prepare arguments for parallel processing
        # See: https://stackoverflow.com/questions/47424315/python-using-list-multiple-arguments-in-pool-map
        args_list = []
        for k_parameter in range(ngamma):
            args_list.append((k_parameter, dataset, j_sim, src, tstep, label1, label2, remaining_labels, info, events, fwd, fs))
        
        # Process parameters in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(process_parameter, args_list)
        
        # Store results
        for k_parameter, acws in results:
            acw_results[dataset_name][:, k_parameter, j_sim] = np.nanmean(acws, axis=0)
            print(f"    Parameter {k_parameter+1}/{ngamma} completed")

pklsave(os.path.join(savepath, "supplementary_sensitivity_control_rest_acws_channel_space.pkl"), 
        {"acw_results": acw_results})

print("done")