using LinearAlgebra, SparseArrays
using Combinatorics, NonlinearSolve, LinearSolve

include("../lib/grid_setup.jl")
include("../lib/hierarchical.jl")
include("../lib/grid_projection.jl")
include("../lib/grid_FD.jl")
include("../lib/grid_adaption.jl")

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

### SETUP GRID

function setup_grid(pa::Params; surplus, dense = false)
    if dense == true
        l = pa.l_dense
    else
        l = pa.l
    end
    names_dict = Dict(pa.names[i] => pa.named_dims[i] for i = 1:pa.d)
    grid, lvl_grid = gen_sparse_grid(pa.d, l, surplus)
    J = size(grid, 1)
    h = 2.0 .^ (-lvl_grid)
    value = grid .* pa.range .+ pa.min
    dx = pa.range[names_dict[:a]] * minimum(h[:, names_dict[:a]])
    _, H_comp = gen_H_mat(grid, lvl_grid)

    DS_boundary_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}()
    DSijs_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}()
    DS_const_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}()
    DFull_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}()

    for i in 1:length(pa.discrete_types)
        name = pa.discrete_types[i]
        DS_boundary_dict[Symbol(name, :D1F)] = Vector{SparseMatrixCSC}(undef, pa.d)
        DS_boundary_dict[Symbol(name, :D1B)] = Vector{SparseMatrixCSC}(undef, pa.d)
        DS_boundary_dict[Symbol(name, :D1C)] = Vector{SparseMatrixCSC}(undef, pa.d)
        DS_boundary_dict[Symbol(name, :D2)] = Vector{SparseMatrixCSC}(undef, pa.d)
        DSijs_dict[Symbol(name, :D1F)] = Vector{Matrix{Float64}}(undef, pa.d)
        DSijs_dict[Symbol(name, :D1B)] = Vector{Matrix{Float64}}(undef, pa.d)
        DSijs_dict[Symbol(name, :D1C)] = Vector{Matrix{Float64}}(undef, pa.d)
        DSijs_dict[Symbol(name, :D2)] = Vector{Matrix{Float64}}(undef, pa.d)
        DS_const_dict[Symbol(:DCH_, name)] = Vector{SparseMatrixCSC}(undef, pa.d)
        DS_const_dict[Symbol(name, :D1F)] = Vector{Vector{Float64}}(undef, pa.d)
        DS_const_dict[Symbol(name, :D1B)] = Vector{Vector{Float64}}(undef, pa.d)
        DS_const_dict[Symbol(name, :D1C)] = Vector{Vector{Float64}}(undef, pa.d)
        DS_const_dict[Symbol(name, :D2)] = Vector{Vector{Float64}}(undef, pa.d)
        DFull_dict[Symbol(name, :D1F)] = Vector{Matrix{Float64}}(undef, pa.d)
        DFull_dict[Symbol(name, :D1B)] = Vector{Matrix{Float64}}(undef, pa.d)
        DFull_dict[Symbol(name, :D1C)] = Vector{Matrix{Float64}}(undef, pa.d)
        DFull_dict[Symbol(name, :D2)] = Vector{Matrix{Float64}}(undef, pa.d)
    end
    G = Grid(
        pa.d,
        pa.min,
        pa.range,
        names_dict,
        pa.dxx_dims,
        pa.dxy_dims,
        grid,
        lvl_grid,
        h, # hierarchical nodal distance
        value, # 0-1 to economic values
        dx, # mesh grid distance
        J, # number of points in the grid
        J > 100, # sparse or not
        H_comp, # hierarchical gains
        spzeros(J,J), # BH_dense
        spzeros(J,J), # BH_adapt
        Dict{Symbol, Union{Array, SparseMatrixCSC}}( # bound_grid_dict
            :grid_to_bound => Vector{Vector{Int64}}(undef, pa.d),
            :grid => Vector{Matrix{Float64}}(undef, pa.d),
            :lvl => Vector{Matrix{Int64}}(undef, pa.d),
            # :ids => Vector{Vector{Int64}}(undef, pa.d),
            :BH_grid_to_bound_comp => Vector{SparseMatrixCSC}(undef, pa.d),
            :bound_Hk => Vector{SparseMatrixCSC}(undef, pa.d),
            :bound_Ek => Vector{SparseMatrixCSC}(undef, pa.d),
            :left_neighbor_bound => Vector{Vector{Int64}}(undef, pa.d),
            :right_neighbor_bound => Vector{Vector{Int64}}(undef, pa.d)
        ),
        Dict{Symbol, Vector{Dict}}(), # BC_dict
        Dict{Symbol, Union{Array, SparseMatrixCSC}}( # DS_interior_dict
            :D1F => Vector{SparseMatrixCSC}(undef, pa.d),
            :D1B => Vector{SparseMatrixCSC}(undef, pa.d),
            :D1C => Vector{SparseMatrixCSC}(undef, pa.d),
            :D2  => Vector{SparseMatrixCSC}(undef, pa.d)
        ),
        DS_boundary_dict,
        DS_const_dict,
        DSijs_dict,
        DFull_dict,
        Dict{Symbol, Union{Int64, Float64}}(), # stats_dict
        Matrix{Float64}(undef,0,0), # blacklist
        Vector{Matrix{Float64}}(undef, pa.max_adapt_iter) # G_adapt
    )
    gen_bound_grid!(G)
    gen_FD_interior!(G)

    return G
end

### HOUSEHOLD VARIABLES
mutable struct Household
    V::Matrix{Float64}
    μ::Vector{Array}
    μ_dense::Vector{Array}
    σ::Vector{Array}
    σ_dense::Vector{Array}
    cpol::Matrix{Float64}
    spol::Matrix{Float64}
    income::Matrix{Float64}
    u::Matrix{Float64} # utilities
    g::Matrix{Float64} # distribution
    B::Float64
    C::Float64
    r::Float64
    w::Float64
    Y::Float64
    A::SparseMatrixCSC
    V_adapt::Vector{SparseMatrixCSC}
end

function Household(pa::Params)
    V = zeros(2,2)
    μ = Vector{Array}(undef, length(pa.discrete_types))
    μ_dense = Vector{Array}(undef, length(pa.discrete_types))
    σ = Vector{Array}(undef, length(pa.discrete_types))
    σ_dense = Vector{Array}(undef, length(pa.discrete_types))
    cpol = zeros(2,2)
    spol = zeros(2,2)
    u  = zeros(2,2)
    g  = zeros(2,2)
    A = spzeros(10,10)
    r = pa.ρ/3
    w = 1.0
    Y = pa.L
    B = 1.0
    C = 1.0
    income = pa.zz
    V_adapt = Vector{SparseMatrixCSC}(undef, pa.max_adapt_iter)
    return Household(V, μ, μ_dense, σ, σ_dense, cpol, spol, income, u, g, B, C, r, w, Y, A, V_adapt)
end


### ITERATION FUNCTIONS

function VFI!(hh::Household, G::Grid, pa::Params)

    Az = [-sparse(I, G.J, G.J)*pa.λ1 sparse(I, G.J, G.J)*pa.λ1
          sparse(I, G.J, G.J)*pa.λ2 -sparse(I, G.J, G.J)*pa.λ2]

    for iter = 1:pa.maxit
        HJB!(hh, G, pa)
        A11,_ = FD_operator(G, μ = hh.spol[:, 1], σ = zeros(G.J), dims = 1, BC_name = :y1)
        A22,_ = FD_operator(G, μ = hh.spol[:, 2], σ = zeros(G.J), dims = 1, BC_name = :y2)
        A1 = hcat(A11, spzeros(size(A11, 1), size(A22, 2)))
	A2 = hcat(spzeros(size(A22, 1), size(A11, 2)), A22)
        hh.A = vcat(A1, A2) .+ Az

        B = (1/pa.Δ + pa.ρ) .* sparse(I, 2*G.J, 2*G.J) .- hh.A
        b = hh.u[:] .+ hh.V[:] ./ pa.Δ
        V_new = B\b
        V_change = V_new .- hh.V[:]
        hh.V .= reshape(V_new, G.J, length(pa.discrete_types))

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

    VaF = zeros(G.J, length(pa.discrete_types))
    VaB = zeros(G.J, length(pa.discrete_types))
    for j = 1:length(pa.discrete_types)
        VaF[:, j] .= deriv_sparse(G, hh.V[:, j], operator = :D1F, dims = 1, name = pa.discrete_types[j])
        VaB[:, j] .= deriv_sparse(G, hh.V[:, j], operator = :D1B, dims = 1, name = pa.discrete_types[j])
    end
    cF = pa.u1inv.(VaF)
    cB = pa.u1inv.(VaB)
    c0 = copy(hh.income)

    sF = hh.income .- cF
    sB = hh.income .- cB
    IF = sF .> 1e-6 # do not set to 0, floating point error
    IB = (sB .< -1e-6) #.* (IF .== 0)
    I0 = 1 .- IF .- IB

    hh.spol = sF .* IF .+ sB .* IB
    hh.cpol = cF .* IF .+ cB .* IB .+ c0 .* I0
    hh.u = pa.u.(hh.cpol)

    for j = 1:length(pa.discrete_types)
        hh.μ[j] = hh.spol[:, j]
        hh.σ[j] = zeros(G.J)
    end
end

function KF!(hh::Household, G_dense::Grid, pa::Params) # use G_dense, c.f. HJB!

    Az = [-sparse(I, G_dense.J, G_dense.J)*pa.λ1 sparse(I, G_dense.J, G_dense.J)*pa.λ1
          sparse(I, G_dense.J, G_dense.J)*pa.λ2 -sparse(I, G_dense.J, G_dense.J)*pa.λ2]
    A11,_ = FD_operator(G_dense, μ = hh.μ_dense[1], σ = hh.σ_dense[1], dims = 1, BC_name = pa.discrete_types[1])
    A22,_ = FD_operator(G_dense, μ = hh.μ_dense[2], σ = hh.σ_dense[2], dims = 1, BC_name = pa.discrete_types[2])
    A1 = hcat(A11, spzeros(size(A11,1), size(A22, 2)))
    A2 = hcat(spzeros(size(A22,1), size(A11, 2)), A22)
    AT = (vcat(A1, A2) .+ Az)'

    # KF 1
    AT1 = copy(AT)
    b = zeros(length(pa.discrete_types)*G_dense.J)
    i_fix = 1
    b[i_fix] = 0.1
    row = hcat(zeros(1, i_fix-1), 1.0, zeros(1, length(pa.discrete_types)*G_dense.J - i_fix))
    AT1[i_fix, :] = row

    gg = AT1 \ b # !! Singular
    a = G_dense.value[:, G_dense.names_dict[:a]]
    da = G_dense.range[G_dense.names_dict[:a]] * minimum(G_dense.h[:, G_dense.names_dict[:a]])
    g_sum = sum(gg) * da
    gg ./= g_sum
    g1 = reshape(gg, G_dense.J, 2)

    # KF 2
    g = zeros(G_dense.J, length(pa.discrete_types))
    g[a .== pa.amin, :] .= 1/length(pa.zz) / da
    for n = 1:pa.maxit_KF
        B = 1/pa.Δ_KF .* sparse(I, length(pa.discrete_types)*G_dense.J, length(pa.discrete_types)*G_dense.J) .- AT
        b = g[:] ./ pa.Δ_KF
        g_new = B \ b
        dif = maximum(abs.(g[:] .- g_new))
        if dif < pa.crit_KF
            break
        end
        g = reshape(g_new, G_dense.J, 2)
        if n == pa.maxit_KF
            println("KF did not converge. Remaining Gap: ", dif)
        end
    end

    # Some tests
    mass = sum(g .* da)
    if abs(mass - 1) > 1e-5
        println("Distribution not normalized!")
    end
    if maximum(abs.(g1 .- g)) > 1e-5
        println("Distributions g1 and g2 do not align!")
    else
        hh.g = g;
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
    G = setup_grid(pa, surplus = pa.surplus, dense = false);
    # Dense grid
    G_dense = setup_grid(pa, surplus = pa.surplus, dense = true);
    # Projection matrix from sparse to dense: this is for KF! and consistent aggregation
    G.BH_dense = get_projection_matrix(G, G_dense.grid, G_dense.lvl);
    hh.income = hh.r .* G.value[:, G.names_dict[:a]] .+ hh.w .* pa.zz
    hh.V = pa.u.(hh.income) ./ pa.ρ # V0
    return Problem(pa, hh, G, G_dense)
end

function stationary!(r, p::Problem) # p as parameter, has to be the second position

    # @assert hh.r < pa.ρ || hh.r > -0.1 "init $(hh.r) is too large or small"

    # for iter = 1:pa.maxit
        p.hh.income = r .* p.G.value[:, p.G.names_dict[:a]] .+ p.hh.w .* p.pa.zz
        # State-constrained boundary conditions
        left_bound = p.pa.u1.(p.hh.income[p.G.grid[:, 1] .== 0, :])
        right_bound = p.pa.u1.(p.hh.income[p.G.grid[:, 1] .== 1, :])
        BC = Vector{Dict}(undef, p.G.d)
        for t = 1:length(p.pa.discrete_types) # [:y1, :y2]
            BC[1] = Dict(
                :lefttype => :VNB, :righttype => :VNF,
                :leftfn => (x -> left_bound[t] * ones(size(x, 1))),
                :rightfn => (x -> right_bound[t] * ones(size(x, 1)))
            )
            gen_FD!(p.G, BC, name = p.pa.discrete_types[t])
            gen_FD!(p.G_dense, BC, name = p.pa.discrete_types[t])
        end

        # VALUE FUNCTION ITERATION
        VFI!(p.hh, p.G, p.pa)
        # KOLMOGOROV FORWARD
        p.hh.μ_dense = [p.G.BH_dense * p.hh.μ[j] for j  = 1:length(p.pa.discrete_types)]
        p.hh.σ_dense = [p.G.BH_dense * p.hh.σ[j] for j  = 1:length(p.pa.discrete_types)]
        KF!(p.hh, p.G_dense, p.pa)
        # MARKET CLEARING
        a = p.G_dense.value[:, p.G.names_dict[:a]]
        B = sum(a .* p.hh.g .* p.G_dense.dx)

    #     # UPDATE INTEREST RATE
    #     if hh.B > pa.crit
    #         # println("Excess Supply")
    #         rmax = copy(hh.r)
    #         hh.r = 0.5*(hh.r + rmin)
    #     elseif hh.B < -pa.crit
    #         # println("Excess Demand")
    #         rmin = copy(hh.r)
    #         hh.r = 0.5*(hh.r + rmax)
    #     elseif abs(hh.B) < pa.crit
    #         println("Equilibrium Found, Interest rate =", hh.r)
    #         break
    #     end
    # end
    return B
end

function main!(p::Problem, u0)

    probN = IntervalNonlinearProblem(stationary!, u0, p)

    for iter = 1:p.pa.max_adapt_iter
        println(" MainIteration = ", iter)
        # stationary!(hh, G, G_dense, pa, rmin, rmax)
        r = solve(probN, Bisection())
        p.hh.V_adapt[iter] = p.hh.V
        p.G.G_adapt[iter] = p.G.grid
        adapt_grid!( # generate BH_adapt projection and update grid
            p.G, p.hh.V,
            AddRule = :tol, # Expand nodes with hierarchical coefficient greater than 'AddTol'
            AddTol = 1e-5,
            KeepTol = 1e-6
        )
        if p.G.stats_dict[:n_change] == 0
            break
        end
        # update value function crt. the new grid
        p.hh.V = p.G.BH_adapt * p.hh.V
        # update the matrix projection to the dense grid
        p.G.BH_dense = get_projection_matrix(p.G, p.G_dense.grid, p.G_dense.lvl)
    end
end

u0 = (0.001, 0.018)
p = setup_p();
@time main!(p, u0) # 6s
