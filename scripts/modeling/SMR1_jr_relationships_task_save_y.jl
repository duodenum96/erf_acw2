# cd /BICNAS2/ycatal/erf_acw2/scripts/modeling
# nohup julia SMR1_jr_relationships_task_save_y.jl > log/SMR1_jr_relationships_task_save_y.log 2>&1 &
# echo $! > log/SMR1_jr_relationships_task_save_y.pid

cd("/BICNAS2/ycatal/erf_acw2/scripts/modeling")
using Pkg
Pkg.activate(".")

using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

gamma_1_values = collect(40:1:70)
ngamma = length(gamma_1_values)

narea = 2

A_F_values = collect(LinRange(0.0, 20.0, ngamma))
nA_F = length(A_F_values)
A_B_values = collect(LinRange(0.0, 20.0, ngamma))
nA_B = length(A_B_values)
A_L_values = collect(LinRange(0.0, 20.0, ngamma))
nA_L = length(A_L_values)
p, x0, tspan, tsteps = get_default_param("task", 2)
p.A_F = 2.5 # Arbitrary value
c = p.c
tstops = p.tstops

# prob = SDEProblem(jansenrit_2d_noLA_task!, jansenrit_2d_noise_noLA!, x0, tspan, p_noLA)
# ensembleprob = EnsembleProblem(prob)

testvals = Dict("A_F" => A_F_values, "A_B" => A_B_values, "A_L" => A_L_values,
                "gamma_1" => gamma_1_values)
testvals_names = ["A_F", "A_B", "A_L", "gamma_1"]

nsim = 20
all_erfs = Dict("A_F" => zeros((nsim, nA_F, narea)), "A_B" => zeros((nsim, nA_B, narea)),
                  "A_L" => zeros((nsim, nA_L, narea)), "gamma_1" => zeros((nsim, ngamma, narea)))

n_timepoints = length(tsteps)
all_ys = Dict()
for testval_name in testvals_names
    all_ys[testval_name] = zeros((nsim, n_timepoints, narea, ngamma))
end

for (k, testval_name) in enumerate(testvals_names)
    testval_values = testvals[testval_name]
    ntestval = length(testval_values)
    erfs = zeros((nsim, ntestval, narea))
    for (i, testval) in enumerate(testval_values)
        p2 = [p.A_F, p.A_L, p.A_B, p.gamma_1]
        p2[k] = testval
        global prob = SDEProblem(jansenrit_2d_noLA_task!, jansenrit_2d_noise_noLA!, x0, tspan, p2)
        global ensembleprob = EnsembleProblem(prob)

        sol = solve(
            ensembleprob,
            SKenCarp(),
            EnsembleThreads(),
            trajectories = nsim,
            saveat = tsteps,
            callback=callback_function(c, tstops), 
            tstops=tstops
        )


        for j in 1:nsim
            if sol[j].retcode !== ReturnCode.Success
                erfs[j, i, :] .= NaN
            else
                y1 = sol[j](tsteps)[2, :] .- sol[j](tsteps)[3, :]
                y2 = sol[j](tsteps)[10, :] .- sol[j](tsteps)[11, :]
                # _, erf_rms1, _, _ = calc_erf(y1, p.tstops)
                # _, erf_rms2, _, _ = calc_erf(y2, p.tstops)
                # erfs[j, i, 1] = erf_rms1
                # erfs[j, i, 2] = erf_rms2
                all_ys[testval_name][j, :, 1, i] = y1
                all_ys[testval_name][j, :, 2, i] = y2
            end
        end
        println(i)
    end
    all_erfs[testval_name] = copy(erfs)
end

jldsave(joinpath(savepath, "supplementary_sensitivity_control_task_ys.jld2"); 
    A_F = all_ys["A_F"], 
    A_B = all_ys["A_B"], 
    A_L = all_ys["A_L"], 
    gamma_1 = all_ys["gamma_1"])
println("DONE")

jldsave(joinpath(savepath, "parameters.jld2"); 
    A_F_values = A_F_values,
    A_B_values = A_B_values,
    A_L_values = A_L_values,
    gamma_1_values = gamma_1_values)