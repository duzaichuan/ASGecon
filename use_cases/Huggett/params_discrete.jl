@kwdef struct Params # Huggett_discrete
    # Grid construction
    l::Int64             = 5
    surplus::Vector{Int64} = [0]
    l_dense::Int64       = 8
    d::Int64             = 1 # total dimension
    d_idio::Int64        = 1
    d_agg::Int64         = 0
    amin::Float64        = -1.0
    amax::Float64        = 20.0
    min::Matrix{Float64} = [amin;;]
    max::Matrix{Float64} = [amax;;]

    # Grid adaptation:
    add_rule::Symbol      = :tol
    add_tol::Float64      = 1e-5
    keep_tol::Float64     = 1e-6
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
    ρ::Float64 = 0.02
    γ::Float64 = 2.0
    u     = x -> x ^ (1 - γ) / (1 - γ)
    u1    = x -> x ^ (-γ)
    u1inv = x -> x ^ (-1/γ)

    # Earning parameters
    zz::Matrix{Float64}            = [0.8 1.2]
    λ1::Float64                    = 1/3
    λ2::Float64                    = 1/3
    L::Float64                     = λ1/(λ1 + λ2) * zz[1] + λ2/(λ1 + λ2) * zz[2]
    discrete_types::Vector{Symbol} = [:y1, :y2]

    range::Matrix{Float64}  = max .- min
    dxx_dims::Vector{Int64} = Int[]
    dxy_dims::Vector{Int64} = Int[]
    names::Vector{Symbol} = [:a]
    named_dims::Vector{Int64} = [1]
end
