# cd /BICNAS3/ycatal/erf_acw2/scripts/modeling
# nohup julia supplementary_sensitivity_control.jl > log/supplementary_sensitivity_control.log &
using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"
narea = 10
nsims = 500

task_data = load(joinpath(savepath, "task_10rois.jld2"))

ys = task_data["ys"]

narea, ntime, nsims = size(ys)

p, x0, tspan, tsteps = get_default_param("task", narea)

dt = tsteps[2] - tsteps[1]
fs = 1 / dt
n_erf_timepoints = 1201

erfs = zeros(narea, n_erf_timepoints, nsims)
rmss = zeros(narea, nsims)
activationflags = zeros(Bool, narea, nsims)

for i in 1:narea
    for j in 1:nsims
        y = ys[i, :, j]
        if any(isnan.(y))
            erfs[i, :, j] .= NaN
            rmss[i, j] = NaN
        else
            erf, erf_rms1, _, activationflag = calc_erf(y, p.tstops)
            erfs[i, :, j] = erf[:]
            rmss[i, j] = erf_rms1
            activationflags[i, j] = activationflag
        end
    end
    println(i)
end

println(sum(activationflags, dims=2))

save(joinpath(savepath, "erfs_task_10rois.jld2"), "erfs", erfs, "rmss", rmss, "activationflags", activationflags)