using JLD2

restsens = load("/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/sobol_sensitivity_rest_final.jld2")["sensitivityresults"]
tasksens = load("/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/sobol_sensitivity_task_final.jld2")["sensitivityresults"]

restsens.S1
restsens.ST
tasksens.S1
tasksens.ST

rest_s1 = restsens.S1
rest_st = restsens.ST
task_s1 = tasksens.S1
task_st = tasksens.ST

rest_s1_ci = restsens.S1_Conf_Int
rest_st_ci = restsens.ST_Conf_Int
task_s1_ci = tasksens.S1_Conf_Int
task_st_ci = tasksens.ST_Conf_Int

restvars = ["ACW (ROI 1)", "ACW (ROI 2)"]
taskvars = ["ERF (ROI 1)", "ERF (ROI 2)"]

param = ["A_F", "A_L", "A_B", "gamma_1"]

jldsave(
    "/BICNAS2/ycatal/erf_acw2/results/modeling/results/sensitivity_results_pythonic.jld2";
    rest_s1=rest_s1,
    rest_st=rest_st,
    task_s1=task_s1,
    task_st=task_st,
    rest_s1_ci=rest_s1_ci,
    rest_st_ci=rest_st_ci,
    task_s1_ci=task_s1_ci,
    task_st_ci=task_st_ci,
    restvars=restvars,
    taskvars=taskvars,
    param=param)