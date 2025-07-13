using Plots
include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")

p, x0, tspan, tsteps = get_default_param("task", 2)
tstops = p.tstops
c = 1e4

p.A_F = 2.5
p2 = [p.A_F, 20, p.A_B, p.gamma_1]

prob = SDEProblem(jansenrit_2d_noLA_task!, jansenrit_2d_noise_noLA!, x0, tspan, p2)
sol = solve(prob, SKenCarp(), saveat=tsteps, tstops=tstops)

plot(sol, idxs=1)
plot!(sol, idxs=2)

# Compare with ND