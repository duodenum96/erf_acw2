##### Helper functions for Jansen-Rit simulations
using DifferentialEquations
using PyCall
using LsqFit
using StatsBase
using TimeseriesSurrogates
using LabelledArrays

#################### Model Functions ########################

function S_general(v, E0, r)
    out = (2 * E0) / (1 + exp(-r * v)) - E0
    return out
end

#### 1D ####
function jansenrit_1d!(dx, x, p, t)
    """
    Differential Eq's for Jansen-Rit model as implemented in 
    David, Harrison and Friston 2005 NeuroImage 
    Modelling event-related responses in the brain
    x is a vector comprised of x1, x2, x3, x4, x5, x6, x7, x8

    y is tricky, it is not updated as in dx. y = x2 - x3.
    A workaround is in the function, define y. After we get output, get y as x2-x3

    u is the input time series. It will be updated with a shock using event callback functions
    This is only one dimensional
    """

    S(x) = S_general(x, p.E0, p.r)

    x1, x2, x3, x4, x5, x6, x7, x8 = x
    y = x2 - x3

    dx[1] = x4
    dx[2] = x5
    dx[3] = x6
    dx[7] = x8

    dx[4] = (p.H_E / p.tau_E) * (p.gamma_1 * S(y) + p.u) - (2 / p.tau_E) * x4 - (1 / (p.tau_E^2)) * x1
    dx[5] = (p.H_E / p.tau_E) * p.gamma_2 * S(x1) - (2 / p.tau_E) * x5 - (1 / (p.tau_E^2)) * x2
    dx[6] = (p.H_I / p.tau_I) * p.gamma_4 * S(x7) - (2 / p.tau_I) * x6 - (1 / (p.tau_I^2)) * x3
    dx[8] = (p.H_E / p.tau_E) * p.gamma_3 * S(y) - (2 / p.tau_E) * x8 - (1 / (p.tau_E^2)) * x7

    return Nothing
end

function jansenrit_1d_noise!(dx, x, p, t)
    dx[4] = (p.c_noise * p.H_E) / p.tau_E
end

function callback_function(c, tstops)
    condition(u, t, integrator) = any(t .== tstops)
    affect!(integrator) = integrator.u[4] += c # give a shock to x4 when t = 0
    cb = DiscreteCallback(condition, affect!)
    return cb
end

#### 2D ####

function jansenrit_2d!(dx, x, p, t)
    """
    Differential Eq's for Jansen-Rit model as implemented in 
    David, Harrison and Friston 2005 NeuroImage 
    Modelling event-related responses in the brain

    x is a vector comprised of x11, x21, x31, x41, x51, x61, x71, x81, x12, x22, x32, x42, x52, x62, x72, x82

    y1 = x21 - x31
    y2 = x22 - x32

    u is the input time series. It will be updated with a shock using event callback functions
    This is two dimensional

    Only takes the parameters A_F, A_L, A_B
    """

    S(x) = S_general(x, p.E0, p.r)

    x11, x21, x31, x41, x51, x61, x71, x81, x12, x22, x32, x42, x52, x62, x72, x82 = x
    y1 = x21 - x31
    y2 = x22 - x32

    dx[1] = dx11 = x41
    dx[2] = dx21 = x51
    dx[3] = dx31 = x61
    dx[7] = dx71 = x81

    dx[4] =
        dx41 =
            (p.H_E / p.tau_E) * (p.A_L * S(y2) + p.gamma_1 * S(y1) + p.u) -
            (2 / p.tau_E) * x41 - (1 / (p.tau_E^2)) * x11
    dx[5] =
        dx51 =
            (p.H_E / p.tau_E) * (p.A_B * S(y2) + p.A_L * S(y2) + p.gamma_2 * S(x11)) -
            (2 / p.tau_E) * x51 - (1 / (p.tau_E^2)) * x21
    dx[6] =
        dx61 = (p.H_I / p.tau_I) * p.gamma_4 * S(x71) - (2 / p.tau_I) * x61 - (1 / (p.tau_I^2)) * x31
    dx[8] =
        dx81 =
            (p.H_E / p.tau_E) * (p.A_B * S(y2) + p.A_L * S(y2) + p.gamma_3 * S(y1)) -
            (2 / p.tau_E) * x81 - (1 / (p.tau_E^2)) * x71

    dx[9] = dx12 = x42
    dx[10] = dx22 = x52
    dx[11] = dx32 = x62
    dx[15] = dx72 = x82

    dx[12] =
        dx42 =
            (p.H_E / p.tau_E) * (p.A_F * S(y1) + p.A_L * S(y1) + p.gamma_1 * S(y2) + p.u) -
            (2 / p.tau_E) * x42 - (1 / (p.tau_E^2)) * x12
    dx[13] =
        dx52 =
            (p.H_E / p.tau_E) * (p.A_L * S(y1) + p.gamma_2 * S(x12)) -
            (2 / p.tau_E) * x52 - (1 / (p.tau_E^2)) * x22
    dx[14] =
        dx62 = (p.H_I / p.tau_I) * p.gamma_4 * S(x72) - (2 / p.tau_I) * x62 - (1 / (p.tau_I^2)) * x32
    dx[16] =
        dx82 =
            (p.H_E / p.tau_E) * (p.A_L * S(y1) + p.gamma_3 * S(y2)) -
            (2 / p.tau_E) * x82 - (1 / (p.tau_E^2)) * x72

    return Nothing
end

function jansenrit_2d_noise!(dx, x, p, t)
    dx[4] = (p.c_noise * p.H_E) / p.tau_E
    dx[12] = (p.c_noise * p.H_E) / p.tau_E
end

################## 2D stuff but not using labeled arrays because parallelized GSA doesn't like them (https://www.youtube.com/watch?v=UJYT-adBCfk) ############

function jansenrit_2d_noLA!(dx, x, p, t)
    # A_F, A_L, A_B, gamma_1, gamma_4 = p
    A_F, A_L, A_B, gamma_1 = p

    H_E = 3.25
    H_I = 29.3
    tau_E = 10 / 1000
    tau_I = 15 / 1000
    E0 = 2.5
    r = 0.56

    gamma_2 = 0.8 * gamma_1
    gamma_3 = 0.25 * gamma_1
    gamma_4 = 0.25 * gamma_1
    
    c = 1e4
    c_noise = 0.05

    S(x) = S_general(x, E0, r)

    x11, x21, x31, x41, x51, x61, x71, x81, x12, x22, x32, x42, x52, x62, x72, x82 = x
    y1 = x21 - x31
    y2 = x22 - x32

    dx[1] = dx11 = x41
    dx[2] = dx21 = x51
    dx[3] = dx31 = x61
    dx[7] = dx71 = x81

    dx[4] =
        dx41 =
            (H_E / tau_E) * (A_L * S(y2) + gamma_1 * S(y1)) -
            (2 / tau_E) * x41 - (1 / (tau_E^2)) * x11
    dx[5] =
        dx51 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_2 * S(x11)) -
            (2 / tau_E) * x51 - (1 / (tau_E^2)) * x21
    dx[6] =
        dx61 = (H_I / tau_I) * gamma_4 * S(x71) - (2 / tau_I) * x61 - (1 / (tau_I^2)) * x31
    dx[8] =
        dx81 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_3 * S(y1)) -
            (2 / tau_E) * x81 - (1 / (tau_E^2)) * x71

    dx[9] = dx12 = x42
    dx[10] = dx22 = x52
    dx[11] = dx32 = x62
    dx[15] = dx72 = x82

    dx[12] =
        dx42 =
            (H_E / tau_E) * (A_F * S(y1) + A_L * S(y1) + gamma_1 * S(y2)) -
            (2 / tau_E) * x42 - (1 / (tau_E^2)) * x12
    dx[13] =
        dx52 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_2 * S(x12)) -
            (2 / tau_E) * x52 - (1 / (tau_E^2)) * x22
    dx[14] =
        dx62 = (H_I / tau_I) * gamma_4 * S(x72) - (2 / tau_I) * x62 - (1 / (tau_I^2)) * x32
    dx[16] =
        dx82 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_3 * S(y2)) -
            (2 / tau_E) * x82 - (1 / (tau_E^2)) * x72

    return Nothing
end

function jansenrit_2d_noLA_incl_inh!(dx, x, p, t)
    A_F, A_L, A_B, gamma_1, gamma_4 = p

    H_E = 3.25
    H_I = 29.3
    tau_E = 10 / 1000
    tau_I = 15 / 1000
    E0 = 2.5
    r = 0.56

    gamma_2 = 0.8 * gamma_1
    gamma_3 = 0.25 * gamma_1
    # gamma_4 = 0.25 * gamma_1
    
    c = 1e4
    c_noise = 0.05

    S(x) = S_general(x, E0, r)

    x11, x21, x31, x41, x51, x61, x71, x81, x12, x22, x32, x42, x52, x62, x72, x82 = x
    y1 = x21 - x31
    y2 = x22 - x32

    dx[1] = dx11 = x41
    dx[2] = dx21 = x51
    dx[3] = dx31 = x61
    dx[7] = dx71 = x81

    dx[4] =
        dx41 =
            (H_E / tau_E) * (A_L * S(y2) + gamma_1 * S(y1)) -
            (2 / tau_E) * x41 - (1 / (tau_E^2)) * x11
    dx[5] =
        dx51 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_2 * S(x11)) -
            (2 / tau_E) * x51 - (1 / (tau_E^2)) * x21
    dx[6] =
        dx61 = (H_I / tau_I) * gamma_4 * S(x71) - (2 / tau_I) * x61 - (1 / (tau_I^2)) * x31
    dx[8] =
        dx81 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_3 * S(y1)) -
            (2 / tau_E) * x81 - (1 / (tau_E^2)) * x71

    dx[9] = dx12 = x42
    dx[10] = dx22 = x52
    dx[11] = dx32 = x62
    dx[15] = dx72 = x82

    dx[12] =
        dx42 =
            (H_E / tau_E) * (A_F * S(y1) + A_L * S(y1) + gamma_1 * S(y2)) -
            (2 / tau_E) * x42 - (1 / (tau_E^2)) * x12
    dx[13] =
        dx52 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_2 * S(x12)) -
            (2 / tau_E) * x52 - (1 / (tau_E^2)) * x22
    dx[14] =
        dx62 = (H_I / tau_I) * gamma_4 * S(x72) - (2 / tau_I) * x62 - (1 / (tau_I^2)) * x32
    dx[16] =
        dx82 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_3 * S(y2)) -
            (2 / tau_E) * x82 - (1 / (tau_E^2)) * x72

    return Nothing
end

function jansenrit_2d_noLA_task!(dx, x, p, t)
    # Add constant input u, which is given to x4 (dx[4, 12])
    # A_F, A_L, A_B, gamma_1, gamma_4 = p
    A_F, A_L, A_B, gamma_1 = p

    H_E = 3.25
    H_I = 29.3
    tau_E = 10 / 1000
    tau_I = 15 / 1000
    E0 = 2.5
    r = 0.56

    gamma_2 = 0.8 * gamma_1
    gamma_3 = 0.25 * gamma_1
    gamma_4 = 0.25 * gamma_1
    
    c = 1e4
    c_noise = 0.05

    u = 0.0

    S(x) = S_general(x, E0, r)

    x11, x21, x31, x41, x51, x61, x71, x81, x12, x22, x32, x42, x52, x62, x72, x82 = x
    y1 = x21 - x31
    y2 = x22 - x32

    dx[1] = dx11 = x41
    dx[2] = dx21 = x51
    dx[3] = dx31 = x61
    dx[7] = dx71 = x81

    dx[4] =
        dx41 =
            (H_E / tau_E) * (A_L * S(y2) + gamma_1 * S(y1) + u) -
            (2 / tau_E) * x41 - (1 / (tau_E^2)) * x11
    dx[5] =
        dx51 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_2 * S(x11)) -
            (2 / tau_E) * x51 - (1 / (tau_E^2)) * x21
    dx[6] =
        dx61 = (H_I / tau_I) * gamma_4 * S(x71) - (2 / tau_I) * x61 - (1 / (tau_I^2)) * x31
    dx[8] =
        dx81 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_3 * S(y1)) -
            (2 / tau_E) * x81 - (1 / (tau_E^2)) * x71

    dx[9] = dx12 = x42
    dx[10] = dx22 = x52
    dx[11] = dx32 = x62
    dx[15] = dx72 = x82

    dx[12] =
        dx42 =
            (H_E / tau_E) * (A_F * S(y1) + A_L * S(y1) + gamma_1 * S(y2) + u) -
            (2 / tau_E) * x42 - (1 / (tau_E^2)) * x12
    dx[13] =
        dx52 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_2 * S(x12)) -
            (2 / tau_E) * x52 - (1 / (tau_E^2)) * x22
    dx[14] =
        dx62 = (H_I / tau_I) * gamma_4 * S(x72) - (2 / tau_I) * x62 - (1 / (tau_I^2)) * x32
    dx[16] =
        dx82 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_3 * S(y2)) -
            (2 / tau_E) * x82 - (1 / (tau_E^2)) * x72

    return Nothing
end
function jansenrit_2d_noLA_task_incl_inh!(dx, x, p, t)
    # Add constant input u, which is given to x4 (dx[4, 12])
    A_F, A_L, A_B, gamma_1, gamma_4 = p
    # A_F, A_L, A_B, gamma_1 = p

    H_E = 3.25
    H_I = 29.3
    tau_E = 10 / 1000
    tau_I = 15 / 1000
    E0 = 2.5
    r = 0.56

    gamma_2 = 0.8 * gamma_1
    gamma_3 = 0.25 * gamma_1
    
    c = 1e4
    c_noise = 0.05

    u = 0.0

    S(x) = S_general(x, E0, r)

    x11, x21, x31, x41, x51, x61, x71, x81, x12, x22, x32, x42, x52, x62, x72, x82 = x
    y1 = x21 - x31
    y2 = x22 - x32

    dx[1] = dx11 = x41
    dx[2] = dx21 = x51
    dx[3] = dx31 = x61
    dx[7] = dx71 = x81

    dx[4] =
        dx41 =
            (H_E / tau_E) * (A_L * S(y2) + gamma_1 * S(y1) + u) -
            (2 / tau_E) * x41 - (1 / (tau_E^2)) * x11
    dx[5] =
        dx51 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_2 * S(x11)) -
            (2 / tau_E) * x51 - (1 / (tau_E^2)) * x21
    dx[6] =
        dx61 = (H_I / tau_I) * gamma_4 * S(x71) - (2 / tau_I) * x61 - (1 / (tau_I^2)) * x31
    dx[8] =
        dx81 =
            (H_E / tau_E) * (A_B * S(y2) + A_L * S(y2) + gamma_3 * S(y1)) -
            (2 / tau_E) * x81 - (1 / (tau_E^2)) * x71

    dx[9] = dx12 = x42
    dx[10] = dx22 = x52
    dx[11] = dx32 = x62
    dx[15] = dx72 = x82

    dx[12] =
        dx42 =
            (H_E / tau_E) * (A_F * S(y1) + A_L * S(y1) + gamma_1 * S(y2) + u) -
            (2 / tau_E) * x42 - (1 / (tau_E^2)) * x12
    dx[13] =
        dx52 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_2 * S(x12)) -
            (2 / tau_E) * x52 - (1 / (tau_E^2)) * x22
    dx[14] =
        dx62 = (H_I / tau_I) * gamma_4 * S(x72) - (2 / tau_I) * x62 - (1 / (tau_I^2)) * x32
    dx[16] =
        dx82 =
            (H_E / tau_E) * (A_L * S(y1) + gamma_3 * S(y2)) -
            (2 / tau_E) * x82 - (1 / (tau_E^2)) * x72

    return Nothing
end

function jansenrit_2d_noise_noLA!(dx, x, p, t)
    H_E = 3.25
    tau_E = 10 / 1000
    c_noise = 0.05

    dx[4] = (c_noise * H_E) / tau_E
    dx[12] = (c_noise * H_E) / tau_E
end


#################### Parameters ########################

function get_default_param(restortask = "task", dim=1)

    if dim == 1
        p = LVector((
            H_E = 3.25,
            H_I = 29.3,
            tau_E = 10 / 1000,
            tau_I = 15 / 1000,
            gamma_1 = 50,
            gamma_2 = 0.8 * 50,
            gamma_3 = 0.25 * 50,
            gamma_4 = 0.25 * 50,
            E0 = 2.5,
            r = 0.56,
            c = 1e4,
            c_noise = 0.05,
            u = 0,
            tstops = 5:5:95
        ))
        x0 = zeros(8)
    elseif dim == 2
        p = LVector((
            H_E = 3.25,
            H_I = 29.3,
            tau_E = 10 / 1000,
            tau_I = 15 / 1000,
            gamma_1 = 50,
            gamma_2 = 0.8 * 50,
            gamma_3 = 0.25 * 50,
            gamma_4 = 0.25 * 50,
            E0 = 2.5,
            r = 0.56,
            c = 1e4,
            c_noise = 0.05,
            A_F = 0,
            A_B = 0,
            A_L = 0,
            u = 0,
            tstops = 5:5:95
        ))
        x0 = zeros(16)
    end

    tspan = (0.0, 100.0)

    if restortask == "task"
        p.u = 0.0
    end

    tsteps = tspan[1]:(1/1200):tspan[2]

    return p, x0, tspan, tsteps
end

###################### ACW Functions ########################

function acw_simple(x, fs=1200)
    lags = 0:2000
    acf = autocor(x, lags)
    acw50 = findmax(acf .<= 0.5)[2] / fs
    acw0 = findmax(acf .<= 0.0)[2] / fs

    return acw0, acw50
end

function dynamic_acw(x; fs = 1200, windowsize=10, simple=false)
    ntp = length(x)
    ws = windowsize * fs
    nwindow = Int(floor(ntp / ws))
    swindows = [Int.([(i-1)*ws+1, i*ws]) for i in 1:nwindow]

    acw0s = zeros(nwindow)
    acw50s = zeros(nwindow)
    acwdrs = zeros(nwindow)
    if !simple
        for i in 1:nwindow
            acw0s[i], acw50s[i], acwdrs[i] = acw(x[swindows[i][1]:swindows[i][2]])
        end
    else
        for i in 1:nwindow
            acw0s[i], acw50s[i] = acw_simple(x[swindows[i][1]:swindows[i][2]])
        end
    end

    if !simple
        return mean(acw0s), mean(acw50s), mean(acwdrs), std(acw0s), std(acw50s), std(acwdrs), acw0s
    else
        return mean(acw0s), mean(acw50s), std(acw0s), std(acw50s), acw0s
    end
end

############### Helpers #####################

function ind2sub(matrix, i)
    i2s = CartesianIndices(matrix)
    return i2s[i]
end

function sub2ind(matrix, i...)
    s2i = LinearIndices(matrix)
    s2i[i...]
end

function calc_erf(y, tstops; fs=1200, tlim=(-0.3, 0.7), threshold=1.0)
    tlim_samples = convert.(Int, tlim .* fs)
    erfs = hcat([y[((stop*fs) .+ tlim_samples[1]):((stop*fs) .+ tlim_samples[2])] for stop in tstops]...)
    erf = mean(erfs, dims=2)
    time = tlim[1]:(1/fs):tlim[2]
    erf_postonset = erf[time .>= 0]
    erf_rms = sqrt(mean(erf_postonset .^ 2))

    erf_preonset = erf[time .<= 0]
    sd = std(erf_preonset)
    erf_mean = mean(erf_preonset)
    activationflag = erf_rms > (erf_mean+sd) ? true : false
    
    return erf, erf_rms, erfs, activationflag
end
