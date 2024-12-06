# Code to replicate the analyses in the paper

Scripts are located in the folder scripts. 
Clone this github repo. Create a conda environment. Install this repo with 

`pip install -e .`

after cd'ing into the repo. 

Getting the paths right might need a little bit of tinkering. A better way to do this would 
be to make a config file at the erf_acw2 folder. But I don't have time to refactor all the code
according to that. If somebody is volunteering, send a pull request. 

## MODELING

The simulations are done in Julia. Visualization is done in Python. 
Run scripts with the following order:

### Sensitivity analysis
scripts/modeling/SM0_rest_sensitivity_analysis.jl

scripts/modeling/SM1_task_sensitivity_analysis.jl

scripts/modeling/SM2_turn_sensitivity_to_python.jl

### Visualize sensitivity analysis
scripts/modeling/SM3_visualize_sensitivity.py

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
