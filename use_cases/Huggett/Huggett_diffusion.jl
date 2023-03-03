using LinearAlgebra, SparseArrays
using Combinatorics, NLsolve, LinearSolve

include("../../lib/grid_setup.jl")
include("../../lib/grid_hierarchical.jl")
include("../../lib/grid_projection.jl")
include("../../lib/grid_FD.jl")
include("../../lib/grid_adaption.jl")

# Probably can use @view upon matrices slicing to speed up
#!! when itr is empty in reduce(vcat,[]), errors appear
#!! Indexing operations, especially assignment, are expensive, when carried out one element at a time.
@kwdef struct Params # Huggett DIFFUSION
    # Grid construction
    l::Int64               = 2
    surplus::Vector{Int64} = [2, 0]
    l_dense::Vector{Int64} = [7, 4] # vector of "surplus" for dense grid
    d::Int64               = 2 # total dimension
    d_idio::Int64          = 2
    d_agg::Int64           = 0
    amin::Float64          = -1.0
    amax::Float64          = 20.0
    zmin::Float64          = 0.8
    zmax::Float64          = 1.2
    min::Matrix{Float64}   = [amin zmin]
    max::Matrix{Float64}   = [amax zmax]

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
    ρ::Float64 = 0.02
    γ::Float64 = 2.0
    u     = x -> x ^ (1 - γ) / (1 - γ)
    u1    = x -> x ^ (-γ)
    u1inv = x -> x ^ (-1/γ)

    # Earning parameters
    zmean::Float64            = 1.0
    θz::Float64               = 0.25
    σz::Float64               = 0.01
    L::Float64                = 1.0

    range::Matrix{Float64}    = max .- min
    dxx_dims::Vector{Int64}   = [2]
    dxy_dims::Vector{Int64}   = Int[]
    names::Vector{Symbol}     = [:a, :z]
    named_dims::Vector{Int64} = [1, 2]
end

### HOUSEHOLD VARIABLES
mutable struct Household
    V::Vector{Float64}
    cpol::Vector{Float64}
    spol::Vector{Float64}
    s_dense::Vector{Float64}
    income::Vector{Float64}
    u::Vector{Float64} # utilities
    g::Vector{Float64} # distribution
    B::Float64
    C::Float64
    r::Float64
    w::Float64
    Y::Float64
    A::SparseMatrixCSC
    V_adapt::Vector{Vector{Float64}}
end

function Household(pa::Params)
    V = zeros(2)
    cpol = zeros(2)
    spol = zeros(2)
    s_dense = zeros(2)
    u  = zeros(2)
    g  = zeros(2)
    A = spzeros(10,10)
    r = pa.ρ/3
    w = 1.0
    Y = pa.L
    B = 1.0
    C = 1.0
    income = [pa.zmean]
    V_adapt = Vector{Vector{Float64}}(undef, pa.max_adapt_iter)
    return Household(V, cpol, spol, s_dense, income, u, g, B, C, r, w, Y, A, V_adapt)
end


### ITERATION FUNCTIONS
function VFI!(hh::Household, G::Grid, pa::Params)

    Az, const_z = FD_operator(G, μ = pa.θz .* (pa.zmean .- G.value[:, G.names_dict[:z]]), σ = pa.σz * ones(G.J), dims = 2)

    for iter = 1:pa.maxit
        HJB!(hh, G, pa)
        Aa, const_a = FD_operator(G, μ = hh.spol, σ = zeros(G.J), dims = 1)
        hh.A = Aa .+ Az

        B = (1/pa.Δ + pa.ρ) .* sparse(I, G.J, G.J) .- hh.A
        b = hh.u .+ hh.V ./ pa.Δ .+ const_a .+ const_z
        probB = LinearProblem(B, b)
        V_new = solve(probB).u # KLUFactorization() < 0.004s
        # V_new = B\b # 0.008s x 150 times per main loop
        V_change = V_new .- hh.V
        hh.V .= V_new

        dist = maximum(abs.(V_change))
        if dist < pa.crit
            # println(" Iteration = ", iter, " Max Diff = ", dist)
            # println(" Converge!")
            break
        elseif !isreal(hh.V)
            println("Complex values in VFI: terminating process.")
            break
        elseif iter == pa.maxit && dist[iter] >= dist
            error("Did not converge within $it rounds")
	end
    end
end


function HJB!(hh::Household, G::Grid, pa::Params)

    VaF = deriv_sparse(G, hh.V, operator = :D1F, dims = 1)
    VaB = deriv_sparse(G, hh.V, operator = :D1B, dims = 1)

    cF = pa.u1inv.(VaF)
    cB = pa.u1inv.(VaB)
    c0 = hh.income

    sF = hh.income .- cF
    sB = hh.income .- cB
    IF = (sF .> 1e-6) .* (G.grid[:, 1] .< 1) # do not set to 0, floating point error
    IB = (sB .< -1e-6) .* (IF .== 0) .* (G.grid[:, 1] .> 0)
    I0 = 1 .- IF .- IB

    hh.spol = sF .* IF .+ sB .* IB
    hh.cpol = cF .* IF .+ cB .* IB .+ c0 .* I0
    hh.u = pa.u.(hh.cpol)
end

function KF!(hh::Household, G_dense::Grid, pa::Params) # use G_dense, c.f. HJB!

    Az,_ = FD_operator(G_dense, μ = pa.θz .* (pa.zmean .- G_dense.value[:, G_dense.names_dict[:z]]), σ = pa.σz .* ones(G_dense.J), dims = 2)
    Aa,_ = FD_operator(G_dense, μ = hh.s_dense, σ = zeros(G_dense.J), dims = 1)
    AT = (Aa .+ Az)'

    # KF 1
    b = zeros(G_dense.J)
    i_fix = 1
    b[i_fix] = 0.1
    row = hcat(zeros(1, i_fix-1), 1.0, zeros(1, G_dense.J - i_fix))
    AT[i_fix, :] = row

    gg = AT \ b
    g_sum = sum(gg .* prod(G_dense.dx))
    hh.g = gg ./ g_sum

    # Some tests
    mass = sum(hh.g .* prod(G_dense.dx))
    if abs(mass - 1) > 1e-5
        println("Distribution not normalized!")
    end
end


### MAIN SECTION
mutable struct Problem # setup problem container for NonlinearSolve.jl
    const pa::Params
    hh::Household
    G::Grid
    const G_dense::Grid
end

function setup_p()
    pa = Params();
    hh = Household(pa);
    # Sparse Grid
    G = setup_grid(pa, level = pa.l, surplus = pa.surplus);
    # Dense grid
    G_dense = setup_grid(pa, level = 0, surplus = pa.l_dense);
    # Projection matrix from sparse to dense: this is for KF! and consistent aggregation
    G.BH_dense = get_projection_matrix(G, G_dense.grid, G_dense.lvl);
    hh.income = hh.r .* G.value[:, G.names_dict[:a]] .+ hh.w .* G.value[:, G.names_dict[:z]]
    hh.V = pa.u.(hh.income) ./ pa.ρ # V0
    return Problem(pa, hh, G, G_dense)
end

function stationary!(r, p::Problem) # p as parameter, has to be the second position

    # @assert hh.r < pa.ρ || hh.r > -0.1 "init r is too large or small"
    (; pa, hh, G, G_dense) = p
    # for iter = 1:pa.maxit
        hh.income = r .* G.value[:, G.names_dict[:a]] .+ hh.w .* G.value[:, G.names_dict[:z]]
        # State-constrained boundary conditions
        left_bound = pa.u1.(hh.income)
        right_bound = pa.u1.(hh.income)
        BC = Vector{Dict}(undef, G.d)
        BC[1] = Dict(
            :lefttype => :VNB, :righttype => :VNF,
            :leftfn => (x -> sparse_project(G, x, left_bound)),
            :rightfn => (x -> sparse_project(G, x, right_bound))
        )
        BC[2] = Dict(
            :lefttype => :zero, :righttype => :zero
        )
        gen_FD!(G, BC)
        gen_FD!(G_dense, BC) # Note this is not actually necessary for Huggett

        # VALUE FUNCTION ITERATION
        VFI!(hh, G, pa)
        # KOLMOGOROV FORWARD
        hh.s_dense = G.BH_dense * hh.spol
        KF!(hh, G_dense, pa)
        # MARKET CLEARING
        a = G_dense.value[:, G.names_dict[:a]]
        hh.B = sum(a .* hh.g .* G_dense.dx[1] .* G_dense.dx[2])
        # hh.ssS = sum(G.BH_dense * hh.spol .* hh.g .* da)

        # # UPDATE INTEREST RATE
        # if hh.B > pa.crit
        #     # println("Excess Supply")
        #     rmax = copy(hh.r)
        #     hh.r = 0.5*hh.r + 0.5*rmin
        # elseif hh.B < -pa.crit
        #     # println("Excess Demand")
        #     rmin = copy(hh.r)
        #     hh.r = 0.5*hh.r + 0.5*rmax
        # elseif abs(hh.B) < pa.crit
        #     println("Equilibrium Found, Interest rate =", hh.r)
        #     break
        # end
    # end
    return hh.B
end

function main!(p::Problem)

    (; pa, hh, G, G_dense) = p;

    for iter = 1:pa.max_adapt_iter
        println(" MainIteration = ", iter)
        hh.r = nlsolve(f!, [hh.r], method = :trust_region).zero[1]
        hh.B = stationary!(hh.r, p)

        println("Stationary Equilibrium: (r = $(hh.r), B = $(hh.B))")
        hh.V_adapt[iter] = hh.V
        G.G_adapt[iter] = G.grid
        adapt_grid!( # generate BH_adapt projection and update grid
            G, hh.V,
            AddRule = :tol, # Expand nodes with hierarchical coefficient greater than 'AddTol'
            AddTol = 1e-5,
            KeepTol = 1e-6
        )
        if G.stats_dict[:n_change] == 0
            break
        end
        # update value function crt. the new grid
        hh.V = G.BH_adapt * hh.V
        # update the matrix projection to the dense grid
        G.BH_dense = get_projection_matrix(G, G_dense.grid, G_dense.lvl)
    end
end

p = setup_p();

function f!(F, x)
    F[1] = stationary!(x[1], p) # excess_bond
end

@time main!(p) # 17s for the first run (compiling time included), still 2x slower than original matlab code
# show(stdout, "text/plain", setdiff(G.G_adapt[4], G.G_adapt[5]))
