using Plots
using JLD2
using DSP
using Measures

# load the ACFs for figure 2
savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/results"

data = load(joinpath(savepath, "supplementary_sensitivity_control_rest_acf_and_psd.jld2"))

all_acfs = data["all_acfs"]
all_psds = data["all_psds"]
freqs = data["freqs"]
lags = data["lags"]
A_F_values = data["A_F_values"]
A_B_values = data["A_B_values"]
A_L_values = data["A_L_values"]
gamma_1_values = data["gamma_1_values"]
fs = 1200
lags_seconds = lags ./ fs

freq = load(joinpath(savepath, "freq.jld2"))["freq"]

parameters = ["A_F", "A_B", "A_L", "gamma_1"]

figsize=(1200, 800)

for (i, param) in enumerate(parameters)
    parameter_values = data[param * "_values"]
    
    # ACFs
    acfs = data["all_acfs"][param]
    n_parameter, n_roi, n_lags, n_sims = size(acfs)
    midpoint = n_parameter ÷ 2
    values_to_plot = [1, midpoint, n_parameter]
    
    p1s = []
    p2s = []
    for (j, value_idx) in enumerate(values_to_plot)
        j_acf_1 = acfs[value_idx, 1, :, :]
        j_acf_2 = acfs[value_idx, 2, :, :]
        p1 = plot(lags_seconds, j_acf_1, label="", title=param * " = " * string(round(parameter_values[value_idx], digits=2)),
            xlims=(0.0, 0.5), ylims=(-0.5, 1.1), yticks=-0.4:0.2:1.0)
        
        p2 = plot(lags_seconds, j_acf_2, label="", title=param * " = " * string(round(parameter_values[value_idx], digits=2)),
            xlims=(0.0, 0.5), ylims=(-0.5, 1.1), yticks=-0.4:0.2:1.0, xlabel="Lags (s)")

        if j == 1
            ylabel!(p1, "Area 1\nAutocorrelation")
            ylabel!(p2, "Area 2\nAutocorrelation")
        end
        
        push!(p1s, p1)
        push!(p2s, p2)
    end
    plot(p1s..., p2s..., size=figsize, margins=10mm,
            xtickfontsize=12, ytickfontsize=12, xguidefontsize=16, yguidefontsize=16, titlefontsize=18)
    savefig(joinpath(savepath, "rest_acfs_$(param).png"))

    # PSDs

    psds = data["all_psds"][param]
    n_parameter, n_roi, n_freqs, n_sims = size(psds)
    midpoint = n_parameter ÷ 2
    values_to_plot = [1, midpoint, n_parameter]

    p1s = []
    p2s = []
    for (j, value_idx) in enumerate(values_to_plot)
        j_psd_1 = psds[value_idx, 1, :, :]
        j_psd_2 = psds[value_idx, 2, :, :]
        p1 = plot(freq[2:end], j_psd_1[2:end, :], label="", title=param * " = " * string(round(parameter_values[value_idx], digits=2)),
            scale=:log10, xlims=(1.0, 70.0))
        
        p2 = plot(freq[2:end], j_psd_2[2:end, :], label="", title=param * " = " * string(round(parameter_values[value_idx], digits=2)),
            scale=:log10, xlims=(1.0, 70.0), xlabel="Frequency (Hz)")

        if j == 1
            ylabel!(p1, "Area 1\nPower")
            ylabel!(p2, "Area 2\nPower")
        end
        
        push!(p1s, p1)
        push!(p2s, p2)
    end
    plot(p1s..., p2s..., size=figsize, margins=10mm, 
            xtickfontsize=12, ytickfontsize=12, xguidefontsize=16, yguidefontsize=16, titlefontsize=18)
    savefig(joinpath(savepath, "rest_psds_$(param).png"))
end


########################### Figure 3

data = load(joinpath(savepath, "rest_acf_and_psd.jld2"))

acfs = data["acfs"]
psds = data["psds"]
gamma_1_values = data["gamma_1_values"]
lags = data["lags"]
freqs = data["freqs"]
 
n_sims, n_parameter, n_lags = size(acfs)
midpoint = n_parameter ÷ 2
values_to_plot = [1, midpoint, n_parameter]

figsize=(1200, 400)
p1s = []
for (j, value_idx) in enumerate(values_to_plot)
    j_acf = acfs[:, value_idx, :]
    p1 = plot(lags_seconds, j_acf', label="", title="gamma_1" * " = " * string(round(parameter_values[value_idx], digits=2)),
                xlims=(0.0, 0.5), ylims=(-0.5, 1.1), yticks=-0.4:0.2:1.0)
    
    if j == 1
        ylabel!(p1, "Autocorrelation")
    end
    
    push!(p1s, p1)
end
plot(p1s..., size=figsize, margins=10mm, layout=(1, 3),
        xtickfontsize=12, ytickfontsize=12, xguidefontsize=16, yguidefontsize=16, titlefontsize=18)

savefig(joinpath(savepath, "rest_acfs_figure3.png"))

# PSDs

n_sims, n_parameter, n_freqs = size(psds)
midpoint = n_parameter ÷ 2
values_to_plot = [1, midpoint, n_parameter]

p1s = []
for (j, value_idx) in enumerate(values_to_plot)
    j_psd = psds[:, value_idx, :]
    p1 = plot(freq[2:end], j_psd[:, 2:end]', label="", title="gamma_1" * " = " * string(round(gamma_1_values[value_idx], digits=2)),
                scale=:log10, xlims=(1.0, 70.0))

    if j == 1
        ylabel!(p1, "Power")
    end
    
    push!(p1s, p1)
end
plot(p1s..., size=figsize, margins=10mm, layout=(1, 3),
        xtickfontsize=12, ytickfontsize=12, xguidefontsize=16, yguidefontsize=16, titlefontsize=18)
savefig(joinpath(savepath, "rest_psds_figure3.png"))

