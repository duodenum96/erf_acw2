# cd /BICNAS2/ycatal/erf_acw2/scripts/modeling
# nohup julia SM4_jansenrit_rest.jl > log/SM4_jansenrit_rest.log &
# echo $! > log/SM4_jansenrit_rest.pid
using Pkg
cd("/BICNAS2/ycatal/erf_acw2/scripts/modeling")
Pkg.activate(".")

using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
using DSP
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

gamma_1_values = collect(40:1:70)
ngamma = length(gamma_1_values)
p, x0, tspan, tsteps = get_default_param("rest", 1)

prob = SDEProblem(jansenrit_1d!, jansenrit_1d_noise!, x0, tspan, p)
ensembleprob = EnsembleProblem(prob)

nsim = 40
lags = 0:2000
nlags = length(lags)
nfreqs = 6001
acw50s = zeros((nsim, ngamma))
acfs = zeros((nsim, ngamma, nlags))
psds = zeros((nsim, ngamma, nfreqs))
for (i, gamma) in enumerate(gamma_1_values)
    p2 = copy(p)
    p2.gamma_1 = gamma
    p2.gamma_2 = 0.8*gamma
    p2.gamma_3 = 0.25*gamma
    p2.gamma_4 = 0.25*gamma
    prob = SDEProblem(jansenrit_1d!, jansenrit_1d_noise!, x0, tspan, p2)
    ensembleprob = EnsembleProblem(prob)

    sol = solve(ensembleprob, SKenCarp(), trajectories=nsim, saveat=tsteps)
    for j in 1:nsim
        y = sol[j][2, :] .- sol[j][3, :]
        _, mean_acw50, _, _, _, acf = dynamic_acw(y; fs = 1200, windowsize=10, simple=true)
        acw50s[j, i] = mean_acw50
        acfs[j, i, :] = mean(acf)
        psds[j, i, :] = welch_pgram(y, div(length(y), length(acf)), 0, fs=1200).power
    end
    println(i)
end

jldsave(joinpath(savepath, "rest.jld2"); acw50s=acw50s, gamma_1_values=gamma_1_values)
jldsave(joinpath(savepath, "rest_acf_and_psd.jld2"); acfs=acfs, psds=psds,
        gamma_1_values=gamma_1_values, lags=lags, freqs=freqs)
println("DONE")

