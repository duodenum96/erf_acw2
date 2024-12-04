# cd /BICNAS2/ycatal/erf_acw/scripts/model
# nohup julia -t 12 sensitivity_analysis_rest_final.jl > log/gsa_rest_final.log &
# pid: 3969012
using LinearAlgebra
BLAS.set_num_threads(12)


using DifferentialEquations
using Plots
using GlobalSensitivity
using JLD2
using QuasiMonteCarlo
using Missings

include("/BICNAS2/ycatal/erf_acw2/scripts/modeling/src_jansenrit.jl")
savepath = "/BICNAS2/ycatal/erf_acw2/scripts/modeling/"

p, x0, tspan, tsteps = get_default_param("rest", 2)

######## Define ranges for parameters
A_F_range = [0, 5]
A_L_range = [0, 5]
A_B_range = [0, 5]
gamma_1_range = [40, 70]

p_target_ranges = [A_F_range, A_L_range, A_B_range, gamma_1_range]


prob = SDEProblem(jansenrit_2d_noLA!, jansenrit_2d_noise_noLA!, x0, tspan, p)

function sensitivity_parallel(p)

    Np::Int64 = size(p, 2)
    println("Running ", Np, " cases")
    chunksize::Int64 = Np / chunks
    println("Using ", chunks, " chunk(s) of size ", chunksize)

    acwresults = zeros(Union{Missing, Float64}, 2, Np)
    
    for k in 1:chunks
        offset = Int((k-1) * Np / chunks)
        startindx = Int(offset + 1)
        endindx = Int(k * Np / chunks)
        println("Starting chunk ", k, ", from ", startindx, " to ", endindx, " with offset ", offset, ".")

        pchunk = p[:,startindx:endindx]

        
        function prob_func(prob, j, repeat)
            println(j)
            flush(stdout)
            remake(prob; p = pchunk[:, j])
        end
        
        ensembleprob = EnsembleProblem(prob, prob_func=prob_func)

        sol = solve(
            ensembleprob,
            SKenCarp(),
            EnsembleThreads(),
            trajectories = chunksize,
            saveat = tsteps,
        )

        for i in 1:chunksize
            if sol[i].retcode !== ReturnCode.Success
                println("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                acwresults[1, i + offset] = missing
                acwresults[2, i + offset] = missing
            else
                x21 = sol[i][2, :]
                x31 = sol[i][3, :]

                x22 = sol[i][10, :]
                x32 = sol[i][11, :]

                y1 = x21 .- x31
                y2 = x22 .- x32
                _, mean_acw1, _, _, _ = dynamic_acw(y1; fs = 1200, windowsize=10, simple=true)
                _, mean_acw2, _, _, _ = dynamic_acw(y2; fs = 1200, windowsize=10, simple=true)

                acwresults[1, i + offset] = mean_acw1
                acwresults[2, i + offset] = mean_acw2
            end
        end
    end

    # Replace missing values with the median of the row
    acwresults_clean = copy(acwresults)
    for i in 1:size(acwresults, 1)
        row = acwresults[i, :]
        idx_missing = ismissing.(row)
        row[idx_missing] .= median(skipmissing(row))
        acwresults_clean[i, :] = row
    end

    return disallowmissing(acwresults_clean) # convert from Union{Missing, Float64} to Float64
end

N = 20000

chunks = 5000
lb = [A_F_range[1], A_L_range[1], A_B_range[1], gamma_1_range[1]]
ub = [A_F_range[2], A_L_range[2], A_B_range[2], gamma_1_range[2]]
bounds = tuple.(lb, ub)
sampler = SobolSample(R = Shift())
A, B = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, sampler)

m = gsa(sensitivity_parallel, Sobol(nboot=40), A, B; batch=true)

save(joinpath(savepath, "sobol_sensitivity_rest.jld2"), Dict("sensitivityresults" => m))
println("done")
