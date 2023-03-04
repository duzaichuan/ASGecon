using LinearAlgebra, SparseArrays
using Combinatorics, NLsolve, LinearSolve

include("../../lib/grid_setup.jl")
include("../../lib/grid_hierarchical.jl")
include("../../lib/grid_projection.jl")
include("../../lib/grid_FD.jl")
include("../../lib/grid_adaption.jl")

#!! when itr is empty in reduce(vcat,[]), errors appear
@kwdef struct Params # Huggett_discrete
    # Grid construction
    l::Int64             = 5
    surplus::Vector{Int64} = [0]
    l_dense::Int64       = 8
    d::Int64             = 1 # total dimension
    d_idio::Int64        = 1
    d_agg::Int64         = 0
    amin::Float64        = -1.0
    amax::Float64        = 15.0
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
    η::Float64 = 2.0
    u     = x -> x ^ (1 - γ) / (1 - γ)
    u1    = x -> x ^ (-γ)
    u1inv = x -> x ^ (-1/γ)
    v     = x -> x ^ (1 + η) / (1 + η)
    v1    = x -> x ^ (η)
    v1inv = x -> x ^ (1/η)

    # Earning parameters
    zz::Matrix{Float64}            = [0.3 1.0]
    λ1::Float64                    = 0.97
    λ2::Float64                    = 0.09
    L::Float64                     = λ1/(λ1 + λ2) * zz[1] + λ2/(λ1 + λ2) * zz[2]
    discrete_types::Vector{Symbol} = [:y1, :y2]

    # Firms
    ϵ::Float64 = 10.0
    χ::Float64 = 100.0
    τ_empl::Float64 = 0.0
    # Government
    λ_π::Float64 = 1.2
    λ_Y::Float64 = 0.02
    τ_lab::Float64 = 0.2
    UI::Float64 = 0.2
    G::Float64 = 0.0
    gov_bond_supply::Float64 = 1.0

    range::Matrix{Float64}  = max .- min
    dxx_dims::Vector{Int64} = Int[]
    dxy_dims::Vector{Int64} = Int[]
    names::Vector{Symbol} = [:a]
    named_dims::Vector{Int64} = [1]
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
    lpol::Matrix{Float64}
    s_dense::Matrix{Float64}
    c0::Matrix{Float64}
    c_left::Matrix{Float64}
    c_right::Matrix{Float64}
    u::Matrix{Float64} # utilities
    g::Matrix{Float64} # distribution
    B::Float64
    C::Float64
    N::Float64
    excess_bonds::Float64
    excess_saving::Float64
    excess_supply::Float64
    excess_capital::Float64
    r::Float64
    τ::Float64
    w::Float64
    Y::Float64
    A::SparseMatrixCSC
    V_adapt::Vector{Matrix{Float64}}
end

function Household(pa::Params)
    V = zeros(2,2)
    μ = Vector{Array}(undef, length(pa.discrete_types))
    μ_dense = Vector{Array}(undef, length(pa.discrete_types))
    σ = Vector{Array}(undef, length(pa.discrete_types))
    σ_dense = Vector{Array}(undef, length(pa.discrete_types))
    cpol = zeros(2,2)
    spol = zeros(2,2)
    lpol = zeros(2,2)
    s_dense = zeros(2,2)
    c0 = zeros(2,2)
    c_left = zeros(2,2)
    c_right = zeros(2,2)
    u  = zeros(2,2)
    g  = zeros(2,2)
    A = spzeros(10,10)
    r = 0.01
    w = 0.75
    Y = 1.0
    N = 1.0
    mc = (pa.ϵ - 1) / pa.ϵ
    Π = (1 - mc) * Y
    τ = Π + pa.τ_lab * w * N - pa.G - r * pa.gov_bond_supply
    B = 1.0
    C = 1.0
    excess_bonds = 0.0
    excess_saving = 0.0
    excess_supply = 0.0
    excess_labor = 0.0
    V_adapt = Vector{Matrix{Float64}}(undef, pa.max_adapt_iter)
    return Household(V, μ, μ_dense, σ, σ_dense, cpol, spol, lpol, s_dense, c0, c_left, c_right, u, g, B, C, N, excess_bonds, excess_saving, excess_supply, excess_labor, r, τ, w, Y, A, V_adapt)
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
        probB = LinearProblem(B, b)
        V_new = solve(probB).u # V_new = B\b
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

    lF = pa.v1inv.((1 - pa.τ_lab) * hh.w .* pa.zz .* pa.u1.(cF))
    lB = pa.v1inv.((1 - pa.τ_lab) * hh.w .* pa.zz .* pa.u1.(cB))
    l0 = pa.v1inv.((1 - pa.τ_lab) * hh.w .* pa.zz .* pa.u1.(hh.c0))

    sF = @. hh.r * G.value[:, G.names_dict[:a]] + (1 - pa.τ_lab) * hh.w * pa.zz * lF + hh.τ - cF
    sB = @. hh.r * G.value[:, G.names_dict[:a]] + (1 - pa.τ_lab) * hh.w * pa.zz * lB + hh.τ - cB
    s0 = @. hh.r * G.value[:, G.names_dict[:a]] + (1 - pa.τ_lab) * hh.w * pa.zz * l0 + hh.τ - hh.c0

    IF = sF .> 1e-6 # do not set to 0, floating point error
    IB = (sB .< -1e-6) .* (IF .== 0)
    I0 = 1 .- IF .- IB

    hh.spol = @. sF * IF + sB * IB
    hh.cpol = @. cF * IF + cB * IB + hh.c0 * I0
    hh.lpol = @. lF * IF + lB * IB + l0 * I0
    hh.u = @. pa.u(hh.cpol) - pa.v(hh.lpol)

    for j = 1:length(pa.discrete_types)
        hh.μ[j] = hh.spol[:, j]
        hh.σ[j] = zeros(G.J)
    end

    # Tests
    @assert all(abs.(s0) .< 1e-8)
    @assert all(hh.spol[G.grid[:, 1] .== 1, :] .<= 0)
    @assert all(hh.spol[G.grid[:, 1] .== 0, :] .>= 0)
    @assert all(abs.(hh.lpol .- ((1 - pa.τ_lab) * hh.w * pa.zz .* hh.cpol .^(-pa.γ)).^(1/pa.η)) .< 1e-8)
    @assert all(abs.(cF[G.grid[:, 1] .== 1, :] .- hh.c_right) .< 1e-8)
    @assert all(abs.(cF[G.grid[:, 1] .== 0, :] .- hh.c_left) .< 1e-8)

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

    gg = AT1 \ b
    g_sum = sum(gg) * G_dense.dx[1] * G_dense.dx[2]
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
    G = setup_grid(pa, level = pa.l, surplus = pa.surplus);
    # Dense grid
    G_dense = setup_grid(pa, level = pa.l_dense, surplus = pa.surplus);
    # Projection matrix from sparse to dense: this is for KF! and consistent aggregation
    G.BH_dense = get_projection_matrix(G, G_dense.grid, G_dense.lvl);
    c_guess = hh.r .* G.value[:, G.names_dict[:a]] .+ (1 - pa.τ_lab) * hh.w .* pa.zz .+ hh.τ
    hh.V = pa.u.(c_guess) ./ pa.ρ # V0
    return Problem(pa, hh, G, G_dense)
end

function stationary!(x::Vector{Float64}, p::Problem) # p as parameter, has to be the second position

    (; pa, hh, G, G_dense) = p;
    hh.r = x[1]
    hh.Y = x[2]
    mc = (pa.ϵ - 1.0) / pa.ϵ
    hh.w = copy(mc)
    Π = (1 - mc) * hh.Y
    hh.N = copy(hh.Y)
    hh.τ = Π + pa.τ_lab * hh.w * hh.N - pa.G - hh.r * pa.gov_bond_supply

    # Boundary condition
    c_guess = hh.r .* G.value[:, G.names_dict[:a]] .+ (1 - pa.τ_lab) * hh.w .* pa.zz .+ hh.τ
    f(x,a) = @. x - hh.τ - hh.r*a - ((1-pa.τ_lab) * hh.w * pa.zz)^(1/pa.η + 1) * x^(-pa.γ/pa.η)
    f_prime(x) = @. 1 + pa.γ/pa.η * ((1 - pa.τ_lab) * hh.w * pa.zz)^(1/pa.η + 1) * x^(-pa.γ/pa.η-1)

    hh.c_right = newton_nonlin(f, f_prime, c_guess, pa.amax, pa.crit)
    hh.c_left = newton_nonlin(f, f_prime, c_guess, pa.amin, pa.crit)
    hh.c0 = newton_nonlin(f, f_prime, c_guess, G.value[:, G.names_dict[:a]], pa.crit)

    # State-constrained boundary conditions
    left_bound = pa.u1.(c_left)
    right_bound = pa.u1.(c_right)
    BC = Vector{Dict}(undef, G.d)
    for j = 1:length(pa.discrete_types) # [:y1, :y2]
        BC[1] = Dict(
            :lefttype => :VNB, :righttype => :VNF,
            # Since boundary points to add in gen_FD are not located in the grid
            # points of left (right) bounds, we need interpolation
            :leftfn => (points -> sparse_project(G, points, left_bound[:, j])),
            :rightfn => (points -> sparse_project(G, points, right_bound[:, j]))
        )
        gen_FD!(G, BC, name = pa.discrete_types[j])
        gen_FD!(G_dense, BC, name = pa.discrete_types[j])
    end

    # VALUE FUNCTION ITERATION
    VFI!(hh, G, pa)
    # KOLMOGOROV FORWARD
    hh.μ_dense = [G.BH_dense * hh.μ[j] for j  = 1:length(pa.discrete_types)]
    hh.σ_dense = [G.BH_dense * hh.σ[j] for j  = 1:length(pa.discrete_types)]
    KF!(hh, G_dense, pa)
    # MARKET CLEARING
    a = G_dense.value[:, G.names_dict[:a]]
    hh.B = sum(a .* hh.g .* prod(G_dense.dx))
    hh.N = sum((G.BH_dense * (pa.zz .* hh.lpol)) .* hh.g .* prod(G_dense.dx))
    hh.C = sum(G.BH_dense * hh.cpol .* hh.g .* prod(G_dense.dx))
    hh.excess_saving = sum(G.BH_dense * hh.spol .* hh.g .* prod(G_dense.dx))
    hh.excess_bonds = hh.B - pa.gov_bond_supply
    hh.excess_supply = Y - C
    hh.excess_labor = N - hh.N

    return [hh.excess_bonds, hh.excess_supply]
end

function main!(p::Problem)
    (; pa, hh, G, G_dense) = p;

    for iter = 1:pa.max_adapt_iter
        println(" MainIteration = ", iter)
        x = nlsolve(f!, [hh.r, hh.Y])
        hh.r = x.zero[1]
        hh.Y = x.zero[2]
        hh.B = stationary!([hh.r, hh.Y], p)
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
    F[1] = stationary!([x[1], x[2]], p)
end

main!(p) # 22s, still 2.5x slower than original matlab code
# show(stdout, "text/plain", VaF)
