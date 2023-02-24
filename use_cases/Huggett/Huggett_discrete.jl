using LinearAlgebra, SparseArrays
using Combinatorics, NonlinearSolve, LinearSolve

include("params_discrete.jl")
include("../../lib/grid_setup.jl")
include("../../lib/grid_hierarchical.jl")
include("../../lib/grid_projection.jl")
include("../../lib/grid_FD.jl")
include("../../lib/grid_adaption.jl")

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
    G = setup_grid(pa, level = pa.l, surplus = pa.surplus);
    # Dense grid
    G_dense = setup_grid(pa, level = pa.l_dense, surplus = pa.surplus);
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
