# cd /BICNAS2/ycatal/erf_acw/scripts/model
# nohup julia -t 24 sensitivity_analysis_task_final.jl > log/gsa_task_final.log &
# pid: 3683793
using DifferentialEquations
using Plots
using GlobalSensitivity
using JLD2
using QuasiMonteCarlo
using LinearAlgebra
using Missing
BLAS.set_num_threads(24)

include("/BICNAS2/ycatal/erf_acw/scripts/model/src_jansenrit.jl")
savepath = "/BICNAS2/ycatal/erf_acw/scripts/model/"

p, x0, tspan, tsteps = get_default_param("task", 2)
tstops = p.tstops
c = 1e4

######## Define ranges for parameters
A_F_range = [0, 15]
A_L_range = [0, 15]
A_B_range = [0, 15]
gamma_1_range = [40, 70]

p_target_ranges = [A_F_range, A_L_range, A_B_range, gamma_1_range]
p = [p.A_F, p.A_L, p.A_B, p.gamma_1]
prob = SDEProblem(jansenrit_2d_noLA_task!, jansenrit_2d_noise_noLA!, x0, tspan, p)

function sensitivity_parallel(p)
    Np::Int64 = size(p, 2)
    println("Running ", Np, " cases")
    chunksize::Int64 = Np / chunks
    println("Using ", chunks, " chunk(s) of size ", chunksize)

    sensresults = zeros(4, Np) # 1, 2: ACW (roi1, roi2); 3, 4: ERF (roi1, roi2); 5, 6: RTs (roi1, roi2)

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
            callback=callback_function(c, tstops), 
            tstops=tstops
        )
        
        
        for i in 1:chunksize
            if sol[i].retcode !== ReturnCode.Success
                println("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                sensresults[:, i + offset] .= missing
            else
                x21 = sol[i][2, :]
                x31 = sol[i][3, :]

                x22 = sol[i][10, :]
                x32 = sol[i][11, :]

                y1 = x21 .- x31
                y2 = x22 .- x32

                _, erf_rms1, _, rts1 = calc_erf(y1, tstops, threshold=1.0)
                _, erf_rms2, _, rts2 = calc_erf(y2, tstops, threshold=0.1)

                sensresults[1, i + offset] = erf_rms1
                sensresults[2, i + offset] = erf_rms2
            end
        end
    end

    sensresults_clean = copy(sensresults)
    for i in 1:size(sensresults, 1)
        row = sensresults[i, :]
        idx_missing = isnan.(row)
        row[idx_missing] .= median(skipmissing(row))
        sensresults_clean[i, :] = row
    end

    return disallowmissing(sensresults_clean)
end

N = 20000
chunks = 5000
lb = [A_F_range[1], A_L_range[1], A_B_range[1], gamma_1_range[1]]
ub = [A_F_range[2], A_L_range[2], A_B_range[2], gamma_1_range[2]]
bounds = tuple.(lb, ub)
sampler = SobolSample(R = Shift())
A, B = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, sampler)

m = gsa(sensitivity_parallel, Sobol(nboot=40), A, B; batch=true)

save(joinpath(savepath, "sobol_sensitivity_task_final.jld2"), Dict("sensitivityresults" => m))
println("done")
