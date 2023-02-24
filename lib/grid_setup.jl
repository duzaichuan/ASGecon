using LinearAlgebra, SparseArrays, Combinatorics

mutable struct Grid
    const d::Int64 # total dimension
    const min::Matrix{Float64}
    const range::Matrix{Float64}
    const names_dict::Dict{Symbol, Int64}
    const dxx_dims::Vector{Int64}
    const dxy_dims::Vector{Int64}
    grid::Matrix{Float64}
    lvl::Matrix{Int64}
    h::Matrix{Float64} # grid distance in levels
    value::Matrix{Float64}
    dx::Union{Float64, Matrix{Float64}}
    J::Int64 # total numer of points
    sparse::Bool # if J is small, use full matrices
    H_comp::SparseMatrixCSC
    BH_dense::SparseMatrixCSC # project sparse adapt grid to the dense grid of KF! computation
    BH_adapt::SparseMatrixCSC # project old value function to new adapted grid
    bound_grid_dict::Dict{Symbol, Union{Array, SparseMatrixCSC}}
    BC_dict::Dict{Symbol, Vector{Dict}} # Boundary conditions might change for specific dimension
    DS_interior_dict::Dict{Symbol, Union{Array, SparseMatrixCSC}}
    DS_boundary_dict::Dict{Symbol, Union{Array, SparseMatrixCSC}} # Forward (Backward) Diff operator of left (right) bounds
    DS_const_dict::Dict{Symbol, Union{Array, SparseMatrixCSC}} # Backward (Forward) Diff of left (right) bounds derived from the boundary conditions BC_dict
    DSijs_dict::Dict{Symbol, Union{Array, SparseMatrixCSC}}
    DFull_dict::Dict{Symbol, Array} # non-sparse difference matrices
    stats_dict::Dict{Symbol, Union{Int64, Float64}}
    blacklist::Matrix{Float64}
    G_adapt::Vector{Matrix{Float64}}
end

function setup_grid(pa::Params; level::Int64, surplus::Vector{Int64})

    names_dict = Dict(pa.names[i] => pa.named_dims[i] for i = 1:pa.d)
    grid, lvl_grid = gen_sparse_grid(pa.d, level, surplus)
    J = size(grid, 1)
    h = 2.0 .^ (-lvl_grid)
    value = grid .* pa.range .+ pa.min
    dx = zeros(1, pa.d)
    for i = 1:pa.d
        dx[i] = pa.range[pa.named_dims[i]] * minimum(h[:, pa.named_dims[i]])
    end
    _, H_comp = gen_H_mat(grid, lvl_grid)

    if isdefined(pa, :discrete_types)
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
    else
        DS_boundary_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}(
            :MainD1F => Vector{SparseMatrixCSC}(undef, pa.d),
            :MainD1B => Vector{SparseMatrixCSC}(undef, pa.d),
            :MainD1C => Vector{SparseMatrixCSC}(undef, pa.d),
            :MainD2  => Vector{SparseMatrixCSC}(undef, pa.d)
        )
        DSijs_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}(
            :MainD1F  => Vector{Matrix{Float64}}(undef, pa.d),
            :MainD1B  => Vector{Matrix{Float64}}(undef, pa.d),
            :MainD1C  => Vector{Matrix{Float64}}(undef, pa.d),
            :MainD2   => Vector{Matrix{Float64}}(undef, pa.d)
        )
        DS_const_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}(
            :DCH_Main  => Vector{SparseMatrixCSC}(undef, pa.d),
            :MainD1F  => Vector{Vector{Float64}}(undef, pa.d),
            :MainD1B  => Vector{Vector{Float64}}(undef, pa.d),
            :MainD1C  => Vector{Vector{Float64}}(undef, pa.d),
            :MainD2   => Vector{Vector{Float64}}(undef, pa.d)
        )
        DFull_dict = Dict{Symbol, Union{Array, SparseMatrixCSC}}(
            :MainD1F  => Vector{Matrix{Float64}}(undef, pa.d),
            :MainD1B  => Vector{Matrix{Float64}}(undef, pa.d),
            :MainD1C  => Vector{Matrix{Float64}}(undef, pa.d),
            :MainD2   => Vector{Matrix{Float64}}(undef, pa.d)
        )
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

#=
GRID-SETUP AND PROJECTION FUNCTIONS
=#
function gen_sparse_grid(d, n, surplus=zeros(d))

    l = n .* ones(Int, d) .+ surplus
    # NOTE: grid_1D{i} stores the points for level l = i-1
    grid_1D = Vector{Vector{Float64}}(undef, maximum(l)+1)
    grid_1D[1] = [0.5]
    grid_1D[2] = [0.0, 1.0]
    for i = maximum(l)+1:-1:3
        grid_1D[i] = setdiff(0:2.0^(-i+1):1, 0:2.0^(-i+2):1)
    end
    grid_1D_size = map(x->length(x), grid_1D)

    n_surplus = count(!=(0), surplus)
    normal_dims = findall(iszero, surplus)
    surplus_dims = findall(!iszero, surplus)
    d_pos_level = min(n, d - n_surplus)
    # ?? in most cases, normal_dim_combs = normal_dims
    normal_dim_combs = collect(combinations(normal_dims, d_pos_level)) # vector of vector

    surplus_levels = map(x->0:(n+x), surplus[surplus_dims]) # vector of range
    normal_levels = fill(0:n, d_pos_level) # vector of range
    level_combs = ndgrid2([surplus_levels; normal_levels]) # put surplus dimenstions to first columns
    surplus_level_combs = level_combs[:, 1:n_surplus]
    normal_level_combs = level_combs[:, n_surplus+1:end]

    # criterion for sparse grids: |l| <= n (|l| <= n+d -1), start with a regular
    # (symmetric) sparse. Surplus levels account for the interaction effect
    # between certain dimensions. see SpareGridTutorial.pdf Wouter Edeling
    if n_surplus == 0
        keep = sum(normal_level_combs, dims=2)[:] .<= n
        # the sum function returns a one column BitMatrix, convert to Vector
    else # !! if normal levels are empty, no zero is in keep, i.e. no symmetric sparse operation
        keep = (sum(max.(surplus_level_combs .- surplus[surplus_dims]', 0), dims=2) .+ sum(normal_level_combs, dims=2))[:] .<= n
        surplus_level_combs = surplus_level_combs[keep, :]
    end
    normal_level_combs = normal_level_combs[keep, :]
    levels_cell = Vector{Matrix{Int64}}(undef, length(normal_dim_combs))
    for i = 1:length(normal_dim_combs)
        temp = zeros(Int, count(!=(0), keep), d)
        if n_surplus == 0
            temp .= normal_level_combs
        else
            temp[:, surplus_dims] .= surplus_level_combs
            temp[:, normal_dim_combs[i]] .= normal_level_combs
        end
        levels_cell[i] = temp
    end
    # lvls = vcat(levels_cell...) # slower than reduce(vcat, levels_cell)
    lvls = reduce(vcat, levels_cell)
    lvls = unique(lvls, dims=1) # cf. matlab's unique, unique in julia does not sort
    lvl_size = grid_1D_size[lvls .+ 1]
    # lvl_num_el = prod(lvl_size, dims=2)

    # For each combination of levels across dimensions, generate the grid
    # according to those levels; combine all such grids
    grid_cell = Vector{Matrix{Float64}}(undef,size(lvls,1))
    lvl_cell = Vector{Matrix{Int64}}(undef,size(lvls,1))
    for i = 1:size(lvls,1)
        siz = lvl_size[i,:]
        grid_out = zeros(prod(siz), d)
        for j = 1:d
            x = grid_1D[lvls[i,j]+1]
            s = ones(Int, d)
            s[j] = length(x)
            x = reshape(x, s...)
            s .= siz
            s[j] = 1
            x = repeat(x, s...)
            grid_out[:, j] .= x[:]
        end
        grid_cell[i] = grid_out
        lvl_cell[i] = repeat(lvls[i,:], 1, size(grid_cell[i],1))'
    end
    grid = reduce(vcat, grid_cell)
    lvl = reduce(vcat, lvl_cell)
    crosswalk = sortperm(eachrow(grid))
    grid .= grid[crosswalk,:]
    lvl .= lvl[crosswalk,:]
    return grid, lvl
end

"Matlab ndgrid-like function for grid levels."
function ndgrid2(input)

    d = length(input)
    siz = map(x -> length(x), input)
    output = zeros(Int, prod(siz), d)
    for k = 1:d
        x = input[k]
        s = ones(Int, d)
        s[k] = length(x)
        x = reshape(x, s...)
        s .= siz
        s[k] = 1
        x = repeat(x, s...)
        output[:, k] .= x[:]
    end
    return output
end


"""
gen_bound_grid!(G)

Generate the bound_grid_dict field of Grid.

Bounds are extra 0&1 points in each k dimension. Filling these 0&1s into
the grid helps the construction of interior difference operators along d.

grid_to_bound is the inverse index mapping, recovering grid with extra 0&1s
(bound_grid) to the original grid.
"""
function gen_bound_grid!(G::Grid)

    bound_ids = Int[]

    for k = 1:G.d
        new_nodes = [G.grid; G.grid]
        new_nodes[1:G.J, k] .= 0.0
        new_nodes[G.J+1:end, k] .= 1.0
        new_lvls = [G.lvl; G.lvl]
        new_lvls[:, k] .= 1.0
        bound_grid = [G.grid; new_nodes]
        bound_lvl = [G.lvl; new_lvls]

        bound_grid_rows = eachrow(bound_grid)
        IA = unique(i -> bound_grid_rows[i], eachindex(bound_grid_rows))
        bound_grid = bound_grid[IA, :]
        bound_lvl = bound_lvl[IA, :]
        _, bound_Hk = gen_H_mat(bound_grid, bound_lvl, komit = k)
        bound_Ek = sparse(bound_Hk \ Matrix{Float64}(I, size(bound_Hk))) # inv(collect(bound_Hk)) !! singular k=2
        IC = indexin(bound_grid_rows, eachrow(bound_grid))
        grid_to_bound = IC[1:G.J]

        G.bound_grid_dict[:grid_to_bound][k] = grid_to_bound
        G.bound_grid_dict[:grid][k] = bound_grid
        G.bound_grid_dict[:lvl][k] = bound_lvl
        G.bound_grid_dict[:bound_Hk][k] = bound_Hk
        G.bound_grid_dict[:bound_Ek][k] = bound_Ek
        G.bound_grid_dict[:left_neighbor_bound][k], G.bound_grid_dict[:right_neighbor_bound][k] = find_neighbors(bound_grid, 1:size(bound_grid, 1), k)
        bound_ids = [bound_ids; k .* ones(size(bound_grid, 1))]
    end

    bound_grid_all = reduce(vcat, G.bound_grid_dict[:grid])
    bound_lvl_all = reduce(vcat, G.bound_grid_dict[:lvl])
    bound_grid_all_rows = eachrow(bound_grid_all)
    IA = unique(i -> bound_grid_all_rows[i], eachindex(bound_grid_all_rows))
    # !! julia unique does not sort, so the indices (IC) are different. Might cause problems.
    bound_grid_all = bound_grid_all[IA, :]
    bound_lvl_all = bound_lvl_all[IA, :]
    IC = indexin(bound_grid_all_rows, eachrow(bound_grid_all))
    BH_comp = get_projection_matrix(G, bound_grid_all, bound_lvl_all)
    for k = 1:G.d
        ids = IC[bound_ids .== k]
        G.bound_grid_dict[:BH_grid_to_bound_comp][k] = BH_comp[ids, :]
    end
    # G.bound_grid_dict[:bound_all_H], G.bound_grid_dict[:bound_all_H_comp] = gen_H_mat(bound_grid_all, bound_lvl_all)
end
