@kwdef struct Params # Huggett DIFFUSION
    # Grid construction
    l::Int64               = 2
    surplus::Vector{Int64} = [3, 0]
    l_dense::Vector{Int64} = [7, 2] # vector of "surplus" for dense grid
    d::Int64               = 2 # total dimension
    d_idio::Int64          = 2
    d_agg::Int64           = 0
    kmin::Float64          = 0.0
    kmax::Float64          = 50.0
    zmin::Float64          = 0.3
    zmax::Float64          = 1.5
    min::Matrix{Float64}   = [kmin zmin]
    max::Matrix{Float64}   = [kmax zmax]

    # Grid adaptation:
    add_rule::Symbol      = :tol
    add_tol::Float64      = 1e-5
    keep_tol::Float64     = 1e-6 # keep_tol should be smaller than add_tol
    max_adapt_iter::Int64 = 20

    # PDE tuning parameters
    Δ::Int64              = 1000
    maxit::Int64          = 100
    crit::Float64         = 1e-8
    Δ_KF::Int64           = 1000
    maxit_KF::Int64       = 100
    crit_KF::Float64      = 1e-8

    ## ECONOMIC PARAMETERS
    # Household parameters
    ρ::Float64 = 0.05
    γ::Float64 = 2.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    u     = x -> x ^ (1 - γ) / (1 - γ)
    u1    = x -> x ^ (-γ)
    u1inv = x -> x ^ (-1/γ)

    # Earning parameters
    zmean::Float64            = (zmax + zmin)/2
    θz::Float64               = 0.25
    σz::Float64               = 0.02
    L::Float64                = copy(zmean)

    range::Matrix{Float64}    = max .- min
    dxx_dims::Vector{Int64}   = [2]
    dxy_dims::Vector{Int64}   = Int[]
    names::Vector{Symbol}     = [:k, :z]
    named_dims::Vector{Int64} = [1, 2]
end
