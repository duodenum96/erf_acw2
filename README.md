# Code to replicate the analyses in the paper

This repo contains the code to replicate the analyses in the paper 
"How Intrinsic Neural Timescales Relate To Event-Related Activity – Key Role For Intracolumnar Connections"

Preprint available at https://www.biorxiv.org/content/10.1101/2025.01.10.632350v1

Scripts are located in the folder scripts. 
Clone this github repo. Create a conda environment. Install this repo with 

`pip install -e .`

after cd'ing into the repo. 

Getting the paths right might need a little bit of tinkering. A better way to do this would 
be to make a config file at the erf_acw2 folder. 

## MODELING

The simulations are done in Julia. Visualization is done in Python. 
Run scripts with the following order:

### Relationships between variables (2-column model)
scripts/modeling/SM0_jr_relationships.jl

scripts/modeling/SM1_jr_relationships_task.jl

scripts/modeling/SM2_jr_compile_results.jl

### Visualize relationships
scripts/modeling/SM3_visualize_relationships.py

### Plots for correlations / bayesian model
scripts/modeling/SM4_jansenrit_rest.jl

scripts/modeling/SM5_jansenrit_task.jl

scripts/modeling/SM6_visualize_relationships.py

scripts/modeling/SM7_bayesian_models.py

## EMPIRICAL DATA
### Preprocessing
scripts/preprocessing/S0_haririhammer_preproc_1.py

scripts/preprocessing/S1_haririhammer_preproc_2.py

scripts/preprocessing/S2_haririhammer_preproc_3.py

scripts/preprocessing/S3_rest_preproc_1.py

scripts/preprocessing/S4_rest_preproc_2.py

scripts/preprocessing/S5_rest_preproc_3.py


Between 2 and 3, I eyeball the independent components. This generates the two 
bad IC files rest_badICs.py and haririhammer_badICs.py.

### Extract the blocks of emotions (happy, sad, shape) from MEG data
scripts/emotions/S6_extract_emotions.py

### Calculate ERF and do spatiotemporal permutation test between happy, sad and shape and encode vs probe
scripts/erf_calculation/S7_haririhammer_segmentblocks.py

scripts/erf_calculation/S8_haririhammer_calc_erf.py

scripts/erf_calculation/S9_haririhammer_st_permutation.py

### Visualize grand average ERFs and spatiotemporal permutation test results
figures/S10_haririhammer_plot_erf.py

figures/S11_erf_st_test.py

### Get reaction times
scripts/behavior/S12_get_reaction_times.py

### Calculate the ACW-50s, compile results in one pickle file
scripts/acw/S13_calculate_acw.py

scripts/acw/S14_compile_int.py

### Draw topomap for the ACW-50s and nice figure of ACFs
figures/S15_acw_topomap.py

figures/S16_acw_acffigure.py

### Now that we have calculated everything, we can compile all the data in a neat csv file
scripts/S17_compile_all_data_st.py

### Hierarchical Model for ERF - ACW correlation
scripts/acw/S18_erf_acw_hierarchical.py

scripts/acw/S19_erf_acw_bayesian_results.py

### Do Hierarchical Model for ERF - RT correlation using the csv we created at S17 and plot
scripts/behavior/S20_erf_behavior_hierarchical.py

scripts/behavior/S21_erf_behavior_bayesian_results_table.py

### Hierarchical Model for ACW - RT correlation
scripts/behavior/S22_acw_behavior_hierarchical.py

scripts/behavior/S23_acw_behavior_bayesian_results.py
