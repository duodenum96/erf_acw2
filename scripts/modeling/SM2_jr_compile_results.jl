using Plots
using JLD2

savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"
figpath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results/supplementary"

gamma_1_values = collect(40:1:70)
ngamma = length(gamma_1_values)

A_F_values = collect(LinRange(0.0, 20.0, ngamma))
nA_F = length(A_F_values)
A_B_values = collect(LinRange(0.0, 20.0, ngamma))
nA_B = length(A_B_values)
A_L_values = collect(LinRange(0.0, 20.0, ngamma))
nA_L = length(A_L_values)

testvals = Dict("A_F" => A_F_values, "A_B" => A_B_values, "A_L" => A_L_values,
                "gamma_1" => gamma_1_values)

all_acws = jldopen(joinpath(savepath, "supplementary_sensitivity_control_rest.jld2"), "r")["all_acw50s"]
all_erfs = jldopen(joinpath(savepath, "supplementary_sensitivity_control_task.jld2"), "r")["all_erfs"]

nsim, ntestval, narea = size(all_acws["A_F"])

testvals_names = ["A_F", "A_B", "A_L", "gamma_1"]

for testval_name in testvals_names
    fig = Plots.plot(layout=(2,2), size=(800,800))
    testval_values = testvals[testval_name]
    
    # Top row - ACW plots
    for area in 1:2
        for i in 1:ntestval
            scatter!(fig[area], repeat([testval_values[i]], nsim), 
                all_acws[testval_name][:, i, area],
                label="", alpha=0.3, color=:blue)
            
            title = area == 1 ? "Area 1 - ACW" : "Area 2 - ACW"
            plot!(fig[area], title=title, xlabel=testval_name, ylabel="ACW")
        end
    end
    
    # Bottom row - ERF plots
    for area in 1:2
        for i in 1:ntestval
            scatter!(fig[area + 2], repeat([testval_values[i]], nsim), 
                all_erfs[testval_name][:, i, area],
                label="", alpha=0.3, color=:red)
        
        title = area == 1 ? "Area 1 - ERF" : "Area 2 - ERF"
            plot!(fig[area + 2], title=title, xlabel=testval_name, ylabel="ERF")
        end
    end
    
    # Save the figure
    Plots.savefig(joinpath(figpath, "sensitivity_$(testval_name).png"))
end

all_acws_A_F = all_acws["A_F"]
all_acws_A_B = all_acws["A_B"]
all_acws_A_L = all_acws["A_L"]
all_acws_gamma_1 = all_acws["gamma_1"]

all_erfs_A_F = all_erfs["A_F"]
all_erfs_A_B = all_erfs["A_B"]
all_erfs_A_L = all_erfs["A_L"]
all_erfs_gamma_1 = all_erfs["gamma_1"]

jldsave(joinpath(savepath, "sensitivity_control_pythonic.jld2"), 
        acws_A_F=all_acws_A_F,
        acws_A_B=all_acws_A_B,
        acws_A_L=all_acws_A_L,
        acws_gamma_1=all_acws_gamma_1,
        erfs_A_F=all_erfs_A_F,
        erfs_A_B=all_erfs_A_B,
        erfs_A_L=all_erfs_A_L,
        erfs_gamma_1=all_erfs_gamma_1,
        A_F_values=A_F_values,
        A_B_values=A_B_values,
        A_L_values=A_L_values,
        gamma_1_values=gamma_1_values)