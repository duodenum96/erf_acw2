# cd /BICNAS3/ycatal/erf_acw2/scripts/modeling
# nohup julia supplementary_sensitivity_control.jl > log/supplementary_sensitivity_control.log &
using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
using NaNStatistics
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"
narea = 10
nsims = 500

rest_data = load(joinpath(savepath, "rest_10rois.jld2"))

ys = rest_data["ys"]

narea, ntime, nsims = size(ys)

p, x0, tspan, tsteps = get_default_param("rest", narea)

dt = tsteps[2] - tsteps[1]

fs = 1 / dt

lags = 0:10000

windowsize = 10
n_timewindows = 10

acfs = zeros(narea, length(lags), n_timewindows, nsims)

for i in 1:narea
    for j in 1:nsims
        
        x = ys[i, :, j]
        if any(isnan.(x))
            acfs[i, :, :, j] .= NaN
            continue
        end

        ntp = length(x)
        ws = windowsize * fs
        nwindow = Int(floor(ntp / ws))
        swindows = [Int.([(i-1)*ws+1, i*ws]) for i in 1:nwindow]

        for k in eachindex(swindows)
            window = x[swindows[k][1]:swindows[k][2]]
            acf = autocor(window, lags)
            acfs[i, :, k, j] = acf
        end
        println("$(j) / $(nsims)")
    end
    println("$(i) / $(narea)")
end

acw50s = zeros(narea, nsims, n_timewindows)
for i in 1:narea
    for j in 1:nsims
        for k in 1:n_timewindows
            acf = acfs[i, :, k, j]
            if any(isnan.(acf))
                acw50s[i, j, k] = NaN
                continue
            end
            max_value, max_index = findmax(acf .<= 0.5)
            if max_value
                acw50s[i, j, k] = max_index / fs
            else
                error("ANAN")
            end
        end
    end
end

save(joinpath(savepath, "acfs_rest_10rois.jld2"), "acfs", acfs, "acw50s", acw50s)


########################################################
acfs_data = load(joinpath(savepath, "acfs_rest_10rois.jld2"))
acfs = acfs_data["acfs"]
acw50s = acfs_data["acw50s"]

acfs_ave = dropdims(nanmean(acfs, dims=3), dims=3)
acw50s_ave = dropdims(nanmean(acw50s, dims=3), dims=3)

A_B = rest_data["A_B_values"]
A_F = rest_data["A_F_values"]
A_L = rest_data["A_L_values"]
gamma_1 = rest_data["gamma_1_values"]

scatter(A_B[3, :], acw50s_ave[1, :])
scatter(gamma_1[9, :], acw50s_ave[9, :])

