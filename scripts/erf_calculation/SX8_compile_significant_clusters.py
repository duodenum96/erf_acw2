from os.path import join as pathjoin
from erf_acw2.src import pklsave, pklload, get_commonsubj
import numpy as np
import mne

taskname = "haririhammer"

all_effects = ["A", "B"]
effect_names = ["factor_encprob", "factor_emo"]

source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"

def load_cluster_results(taskname, effect_names, source_results_dir=source_results_dir):
    """Load permutation test results for all effects"""
    cluster_results = {}
    output_dir = pathjoin(source_results_dir, taskname)
    
    for effect_name in effect_names:
        loadname = pathjoin(output_dir, f"{taskname}_erp_permutationtest_st_{effect_name}.pkl")
        cluster_results[effect_name] = pklload(loadname)
    
    return cluster_results

def find_significant_clusters(cluster_results, p_thresh=0.01):
    """Find significant clusters across all effects"""
    significant_info = {}
    
    for effect_name, results in cluster_results.items():
        good_cluster_inds = np.where(results["cluster_p"] < p_thresh)[0]
        significant_info[effect_name] = {
            'cluster_indices': good_cluster_inds,
            'cluster_p_values': results["cluster_p"][good_cluster_inds],
            'n_significant': len(good_cluster_inds)
        }
    
    return significant_info

def load_behavioral_data(taskname, subjs_common, source_results_dir=source_results_dir):
    """Load ERF data needed for visualization (sensor space equivalent)"""
    tasknames = ["encode_face_happy", "encode_face_sad", "encode_shape",
                 "probe_face_happy", "probe_face_sad", "probe_shape"]
    
    tasks = {i: [] for i in tasknames}
    
    for i, i_subj in enumerate(subjs_common):
        filename = pathjoin(
            f"/BICNAS2/ycatal/erf_acw2/results/erf/{taskname}",
            f"{i_subj}_{taskname}_erf_emo.pkl",
        )
        epochsdict = pklload(filename)
        for j in tasknames:
            tasks[j].append(epochsdict[j])
    
    return tasks, tasknames

def prepare_comparison_configs():
    """Define comparison configurations for visualization"""
    comparisons = [
        ["encode_face_happy", "encode_face_sad", "encode_shape"], 
        ["probe_face_happy", "probe_face_sad", "probe_shape"],
        ["encode_face_happy", "probe_face_happy"],
        ["encode_face_sad", "probe_face_sad"],
        ["encode_shape", "probe_shape"]
    ]
    
    comparisons_nicer = [
        ["encode face happy", "encode face sad", "encode shape"], 
        ["probe face happy", "probe face sad", "probe shape"],
        ["encode face happy", "probe face happy"],
        ["encode face sad", "probe face sad"],
        ["encode shape", "probe shape"]
    ]
    
    colors_list = [
        ["crimson", "steelblue", "darkorchid"],
        ["maroon", "turquoise", "forestgreen"],
        ["crimson", "maroon"],
        ["steelblue", "turquoise"],
        ["darkorchid", "forestgreen"]
    ]
    
    suptitles = ["Encode", "Probe", "Happy", "Sad", "Shape"]
    
    return {
        'comparisons': comparisons,
        'comparisons_nicer': comparisons_nicer,
        'colors_list': colors_list,
        'suptitles': suptitles
    }

def extract_cluster_data(cluster_results, effect_name, cluster_idx):
    """Extract spatial and temporal information for a specific cluster"""
    clusters = cluster_results[effect_name]["clusters"]
    stats = cluster_results[effect_name]["stats"]
    
    # Source space cluster
    time_inds, vertex_inds = clusters[cluster_idx]
    spatial_extent = len(np.unique(vertex_inds))
    f_map = stats[time_inds, ...].mean(axis=0)
    
    return {
        'time_inds': np.unique(time_inds),
        'spatial_extent': spatial_extent,
        'f_map': f_map,
        'cluster_p': cluster_results[effect_name]["cluster_p"][cluster_idx]
    }

def create_summary_report(cluster_summary, output_dir, taskname):
    """Create a text summary of significant clusters"""
    report_file = pathjoin(output_dir, f"{taskname}_cluster_summary.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"Significant Clusters Summary for {taskname}\n")
        f.write("="*50 + "\n\n")
        
        for effect_name, clusters in cluster_summary.items():
            f.write(f"Effect: {effect_name}\n")
            f.write(f"Number of significant clusters: {len(clusters)}\n")
            
            for i, cluster in enumerate(clusters):
                f.write(f"  Cluster {i+1}:\n")
                f.write(f"    p-value: {cluster['cluster_p']:.6f}\n")
                f.write(f"    Spatial extent: {cluster['spatial_extent']} vertices\n")
                f.write(f"    Temporal extent: {cluster['time_extent']} time points\n")
            
            f.write("\n")

output_compile_dir = "/BICNAS2/ycatal/erf_acw2/results/visualization_packages"

# Load all necessary data
source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"
effect_names = ["factor_encprob", "factor_emo"]
subjs_common = get_commonsubj()

# Load cluster results
cluster_results = load_cluster_results(taskname, effect_names, source_results_dir)

# Find significant clusters
significant_info = find_significant_clusters(cluster_results, p_thresh=0.01)

# Load behavioral data (if needed for validation/comparison)
tasks, tasknames = load_behavioral_data(taskname, subjs_common)

# Prepare comparison configurations
comparison_configs = prepare_comparison_configs()

# Create summary of significant clusters
cluster_summary = {}
for effect_name in effect_names:
    cluster_summary[effect_name] = []
    for cluster_idx in significant_info[effect_name]['cluster_indices']:
        cluster_data = extract_cluster_data(cluster_results, effect_name, cluster_idx)
        cluster_summary[effect_name].append({
            'cluster_idx': cluster_idx,
            'cluster_p': cluster_data['cluster_p'],
            'spatial_extent': cluster_data['spatial_extent'],
            'time_extent': len(cluster_data['time_inds'])
        })

# Package everything for download
visualization_package = {
    'cluster_results': cluster_results,
    'significant_info': significant_info,
    'cluster_summary': cluster_summary,
    'comparison_configs': comparison_configs,
    'tasknames': tasknames,
    'subjs_common': subjs_common,
    'tasks': tasks  # Include if needed for comparison
}

# Save the compiled package
output_file = pathjoin(output_compile_dir, f"{taskname}_visualization_package.pkl")
pklsave(output_file, visualization_package)

# Create a summary report
create_summary_report(cluster_summary, output_compile_dir, taskname)

print(f"Files saved to: {output_compile_dir}")

