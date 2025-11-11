# cd /BICNAS2/ycatal/erf_acw2/scripts/modeling
# nohup julia supplementary_sensitivity_control.jl > log/supplementary_sensitivity_control.log &
using DifferentialEquations
using JLD2
using BenchmarkTools
using Plots
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

narea = 10
nsims = 500 

gamma_1_range = [40.0, 70.0]
A_F_range = [0.0, 10.0]
A_L_range = [0.0, 10.0]
A_B_range = [0.0, 10.0]

gamma_1_values = zeros((narea, nsims))
A_F_values = zeros((narea-1, nsims))
A_B_values = zeros((narea-1, nsims))
A_L_values = zeros((narea-1, nsims))

A_F_matrices = zeros((narea, narea, nsims))
A_L_matrices = zeros((narea, narea, nsims))
A_B_matrices = zeros((narea, narea, nsims))

lower_off_diag_indices = [[i, i-1] for i in 2:narea]
upper_off_diag_indices = [[i-1, i] for i in 2:narea]

for i in 1:nsims
    gamma_1_values[:, i] = gamma_1_range[1] .+ rand(narea) * (gamma_1_range[2] - gamma_1_range[1])
    A_F_values[:, i] = A_F_range[1] .+ rand(narea-1) * (A_F_range[2] - A_F_range[1])
    A_B_values[:, i] = A_B_range[1] .+ rand(narea-1) * (A_B_range[2] - A_B_range[1])
    A_L_values[:, i] = A_L_range[1] .+ rand(narea-1) * (A_L_range[2] - A_L_range[1])

    # Fill A_*A_B_matrices
    # A_F: lower off-diagonal (1->2, 2->3, ...)
    # A_B: upper off-diagonal (2->1, 3->2, ...)
    # A_L: upper and lower off-diagonal (1<->2, 2<->3, ...)
    for j in 1:(narea-1)
        A_F_matrices[lower_off_diag_indices[j][1], lower_off_diag_indices[j][2], i] = A_F_values[j, i]
        A_B_matrices[upper_off_diag_indices[j][1], upper_off_diag_indices[j][2], i] = A_B_values[j, i]

        A_L_matrices[lower_off_diag_indices[j][1], lower_off_diag_indices[j][2], i] = A_L_values[j, i]
        A_L_matrices[upper_off_diag_indices[j][1], upper_off_diag_indices[j][2], i] = A_L_values[j, i]
    end
end

p, x0, tspan, tsteps = get_default_param("rest", narea)

prob = SDEProblem(jansenrit_nd!, jansenrit_nd_noise!, x0, tspan, p)

ys = zeros((narea, length(tsteps), nsims))

for i in 1:nsims
    p.gamma_1 = gamma_1_values[:, i]
    p.A_F = A_F_matrices[:, :, i]
    p.A_B = A_B_matrices[:, :, i]
    p.A_L = A_L_matrices[:, :, i]

    prob_remaked = remake(prob, p=p)
    sol = solve(prob_remaked, SKenCarp(), saveat=tsteps)
    if sol.retcode != ReturnCode.Success
        ys[:, :, i] .= NaN
        continue
    end

    sol_array = Array(sol)

    y = sol_array[2, :, :] .- sol_array[3, :, :]
    ys[:, :, i] = y
    
    println("$(i) / $(nsims)")
end

jldsave(joinpath(savepath, "rest_10rois.jld2"); 
    ys=ys, 
    gamma_1_values=gamma_1_values, 
    A_F_values=A_F_values,
    A_B_values=A_B_values,
    A_L_values=A_L_values
)

println("DONE")
