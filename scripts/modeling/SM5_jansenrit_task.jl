using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
using Statistics
include("/BICNAS2/ycatal/erf_acw/scripts/model/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw/scripts/model"

gamma_1_values = collect(40:1:70)
ngamma = length(gamma_1_values)
p, x0, tspan, tsteps = get_default_param("task", 1)

prob = SDEProblem(jansenrit_1d!, jansenrit_1d_noise!, x0, tspan, p)
ensembleprob = EnsembleProblem(prob)

nsim = 20

acw50s = zeros((nsim, ngamma))
erfs = zeros((nsim, ngamma))
for (i, gamma) in enumerate(gamma_1_values)
    p2 = copy(p)
    p2.gamma_1 = gamma
    p2.gamma_2 = 0.8 * gamma
    p2.gamma_3 = 0.25 * gamma
    p2.gamma_4 = 0.25 * gamma

    prob = SDEProblem(jansenrit_1d!, jansenrit_1d_noise!, x0, tspan, p2)
    ensembleprob = EnsembleProblem(prob)

    sol = solve(ensembleprob, SKenCarp(), trajectories=nsim, saveat=tsteps,
                callback=callback_function(p.c, p.tstops), tstops=p.tstops)
    for j in 1:nsim
        y = sol[j][2, :] .- sol[j][3, :]
        _, erf_rms, _ = calc_erf(y, p.tstops)

        erfs[j, i] = erf_rms
    end
    println(i)
end

jldsave(joinpath(savepath, "task.jld2"); gamma_1_values=gamma_1_values, erfs=erfs)