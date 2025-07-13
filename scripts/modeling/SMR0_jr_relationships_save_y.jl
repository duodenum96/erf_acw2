# cd /BICNAS2/ycatal/erf_acw2/scripts/modeling
# nohup julia SMR0_jr_relationships_save_y.jl > log/SMR0_jr_relationships_save_y.log &
# echo $! > log/SMR0_jr_relationships_save_y.pid

using Pkg
cd("/BICNAS2/ycatal/erf_acw2/scripts/modeling")
Pkg.activate(".")

using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

narea = 2

gamma_1_values = collect(40:1:70)
ngamma = length(gamma_1_values)

A_F_values = collect(LinRange(0.0, 20.0, ngamma))
nA_F = length(A_F_values)
A_B_values = collect(LinRange(0.0, 20.0, ngamma))
nA_B = length(A_B_values)
A_L_values = collect(LinRange(0.0, 20.0, ngamma))
nA_L = length(A_L_values)
p, x0, tspan, tsteps = get_default_param("rest", 2)
p.A_F = 2.5 # Arbitrary value

prob = SDEProblem(jansenrit_2d!, jansenrit_2d_noise!, x0, tspan, p)
ensembleprob = EnsembleProblem(prob)

testvals = Dict("A_F" => A_F_values, "A_B" => A_B_values, "A_L" => A_L_values,
                "gamma_1" => gamma_1_values)
testvals_names = keys(testvals)

nsim = 20
all_acw50s = Dict("A_F" => zeros((nsim, nA_F, narea)), "A_B" => zeros((nsim, nA_B, narea)),
                  "A_L" => zeros((nsim, nA_L, narea)), "gamma_1" => zeros((nsim, ngamma, narea)))

n_time_points = length(tsteps)
all_ys = Dict()
for testval_name in testvals_names
    all_ys[testval_name] = zeros((nsim, n_time_points, narea, ngamma))
end

for testval_name in testvals_names
    testval_values = testvals[testval_name]
    ntestval = length(testval_values)
    # acw50s = zeros((nsim, ntestval, narea))
    for (i, testval) in enumerate(testval_values)
        p2 = copy(p)
        p2[Symbol(testval_name)] = testval
        global prob = SDEProblem(jansenrit_2d!, jansenrit_2d_noise!, x0, tspan, p2)
        global ensembleprob = EnsembleProblem(prob)

        sol = solve(ensembleprob, SKenCarp(), trajectories=nsim, saveat=tsteps)
        for j in 1:nsim
            if sol[j].retcode !== ReturnCode.Success
                acw50s[j, i, :] .= NaN
            else
                y1 = sol[j][2, :] .- sol[j][3, :]
                y2 = sol[j][10, :] .- sol[j][11, :]
                # _, mean_acw501, _, _, _ = dynamic_acw(y1; fs=1200, windowsize=10, simple=true)
                # _, mean_acw502, _, _, _ = dynamic_acw(y2; fs=1200, windowsize=10, simple=true)
                # acw50s[j, i, 1] = mean_acw501
                # acw50s[j, i, 2] = mean_acw502
                all_ys[testval_name][j, :, 1, i] = y1
                all_ys[testval_name][j, :, 2, i] = y2
            end
        end
        println(i)
    end
    # all_acw50s[testval_name] = copy(acw50s)
end

jldsave(joinpath(savepath, "supplementary_sensitivity_control_rest_ys.jld2"); 
    A_F = all_ys["A_F"], 
    A_B = all_ys["A_B"], 
    A_L = all_ys["A_L"], 
    gamma_1 = all_ys["gamma_1"])
println("DONE")

