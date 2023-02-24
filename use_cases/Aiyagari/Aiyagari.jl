using LinearAlgebra, SparseArrays
using Combinatorics, NonlinearSolve, LinearSolve

include("params_diffusion.jl")
include("../../lib/grid_setup.jl")
include("../../lib/grid_hierarchical.jl")
include("../../lib/grid_projection.jl")
include("../../lib/grid_FD.jl")
include("../../lib/grid_adaption.jl")

#!! when itr is empty in reduce(vcat,[]), errors appear

### HOUSEHOLD VARIABLES
mutable struct Household
    V::Vector{Float64}
    cpol::Vector{Float64}
    spol::Vector{Float64}
    s_dense::Vector{Float64}
    income::Vector{Float64}
    u::Vector{Float64} # utilities
    g::Vector{Float64} # distribution
    K::Float64
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
    K = 1.0
    C = 1.0
    income = [pa.zmean]
    V_adapt = Vector{Vector{Float64}}(undef, pa.max_adapt_iter)
    return Household(V, cpol, spol, s_dense, income, u, g, K, C, r, w, Y, A, V_adapt)
end

### ITERATION FUNCTIONS
function VFI!(hh::Household, G::Grid, pa::Params)

    Az, const_z = FD_operator(G, μ = pa.θz .* (pa.zmean .- G.value[:, G.names_dict[:z]]), σ = pa.σz * ones(G.J), dims = 2)

    for iter = 1:pa.maxit
        HJB!(hh, G, pa)
        Ak, const_k = FD_operator(G, μ = hh.spol, σ = zeros(G.J), dims = 1)
        hh.A = Ak .+ Az

        B = (1/pa.Δ + pa.ρ) .* sparse(I, G.J, G.J) .- hh.A
        b = hh.u .+ hh.V ./ pa.Δ .+ const_k .+ const_z
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
    IF = (sF .> 1e-8) # BC takes care of this: (G.grid[:,1]<1)
    IB = (sB .< -1e-8) .* (1 .- IF) # BC takes care of this: (G.grid[:,1]>0)
    I0 = 1 .- IF .- IB

    hh.spol = sF .* IF .+ sB .* IB
    hh.cpol = cF .* IF .+ cB .* IB .+ c0 .* I0
    hh.u = pa.u.(hh.cpol)
end

function KF!(hh::Household, G_dense::Grid, pa::Params) # use G_dense, c.f. HJB!

    Az,_ = FD_operator(G_dense, μ = pa.θz .* (pa.zmean .- G_dense.value[:, G_dense.names_dict[:z]]), σ = pa.σz .* ones(G_dense.J), dims = 2)
    Ak,_ = FD_operator(G_dense, μ = hh.s_dense, σ = zeros(G_dense.J), dims = 1)
    AT = (Ak .+ Az)'

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
    hh.income = hh.r .* G.value[:, G.names_dict[:k]] .+ hh.w .* G.value[:, G.names_dict[:z]]
    hh.V = pa.u.(hh.income) ./ pa.ρ # V0
    return Problem(pa, hh, G, G_dense)
end

function stationary!(K, p::Problem) # p as parameter, has to be the second position

    (; pa, hh, G, G_dense) = p
    hh.Y = K^pa.α * pa.L^(1-pa.α)
    rk = pa.α * hh.Y / K
    hh.w = (1 - pa.α) * hh.Y / pa.L
    hh.r = rk - pa.δ
    hh.income = hh.r .* G.value[:, G.names_dict[:k]] .+ hh.w .* G.value[:, G.names_dict[:z]]

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
    gen_FD!(G_dense, BC) # Note this is not actually necessary for Huggett !! this step is time consuming

    # VALUE FUNCTION ITERATION
    VFI!(hh, G, pa)
    # KOLMOGOROV FORWARD
    hh.s_dense = G.BH_dense * hh.spol
    KF!(hh, G_dense, pa)
    # MARKET CLEARING
    k = G_dense.value[:, G.names_dict[:k]]
    hh.K = sum(k .* hh.g .* G_dense.dx[1] .* G_dense.dx[2])

    return K - hh.K
end

function main!(p::Problem, u0)

    probN = IntervalNonlinearProblem(stationary!, u0, p)
    (; pa, hh, G, G_dense) = p

    for iter = 1:pa.max_adapt_iter
        println(" MainIteration = ", iter)
        K = solve(probN, Bisection())
        stationary!(K.u, p)
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

u0 = (4.0, 6.0)
p = setup_p();
@time main!(p, u0) # 22s, still 2.5x slower than original matlab code
# show(stdout, "text/plain", setdiff(G.G_adapt[4], G.G_adapt[5]))
