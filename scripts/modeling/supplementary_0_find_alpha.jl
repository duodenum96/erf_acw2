using DSP
using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

narea = 2

gamma_1_values = collect(40:1:70)
ngamma = length(gamma_1_values)

A_F_values = collect(LinRange(0.0, 5.0, ngamma))
nA_F = length(A_F_values)
A_B_values = collect(LinRange(0.0, 5.0, ngamma))
nA_B = length(A_B_values)
A_L_values = collect(LinRange(0.0, 5.0, ngamma))
nA_L = length(A_L_values)
p, x0, tspan, tsteps = get_default_param("rest", 2)

prob = SDEProblem(jansenrit_2d!, jansenrit_2d_noise!, x0, tspan, p)
ensembleprob = EnsembleProblem(prob)

testvals = Dict("A_F" => A_F_values, "A_B" => A_B_values, "A_L" => A_L_values,
                "gamma_1" => gamma_1_values)
testvals_names = keys(testvals)

nsim = 10
# Find alpha oscillations
p.A_F = 0.0 # Arbitrary value
p.gamma_1 = 20.0
p.A_L = 0.0
p.A_B = 0.0

p.tau_E = 0.03

prob = SDEProblem(jansenrit_2d!, jansenrit_2d_noise!, x0, tspan, p)
ensembleprob = EnsembleProblem(prob)

sol = solve(ensembleprob, SKenCarp(), trajectories=nsim, saveat=tsteps)
ts = Array(sol) # nroi x ntime x nsim

y1 = ts[2, :, :] .- ts[3, :, :]
y2 = ts[10, :, :] .- ts[11, :, :]

plot(tsteps, y1[:, 1])
xlims!(0, 20)

psd = [periodogram(y1[:, i], fs=1200).power for i in 1:nsim]
psd_ave = mean(hcat(psd...), dims=2)

psd = [periodogram(y2[:, i], fs=1200).power for i in 1:nsim]
psd_ave2 = mean(hcat(psd...), dims=2)

freq = periodogram(y1[:, 1], fs=1200).freq

plot(freq[2:end], psd_ave[2:end], scale=:log10, label="y1")
vline!([10], label="alpha")
plot!(freq[2:end], psd_ave2[2:end], label="y2")

