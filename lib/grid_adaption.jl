#=
GRID ADAPTION FUNCTIONS
=#
function adapt_grid!(G::Grid, f; # f is an array of nodal values for function, could be a vector or matrix with columns of discrete types' size
                     AddRule = :tol, # Expand nodes with hierarchical coefficient greater than 'AddTol'
                     AddTol = 1e-4,
                     KeepTol = 8e-5,
                     BlackListOpt = :children, # Do not add children who are on the blacklist
                     AddDims = 1:size(G.grid, 2))

    @assert KeepTol < AddTol "keep_tol should be smaller than add_tol"
    aH = G.H_comp * f

    if AddRule == :tol
        add_idx = findall(maximum(abs.(aH), dims=2)[:] .> AddTol * (maximum(f[:]) - minimum(f[:])))
    end
    keep = maximum(abs.(aH), dims=2)[:] .> KeepTol * (maximum(f[:]) - minimum(f[:]))
    if isempty(G.blacklist)
        G.blacklist = unique(G.grid[.!keep, :], dims=1)
    else
        G.blacklist = unique([G.blacklist; G.grid[.!keep, :]], dims=1)
    end
    G.stats_dict[:coarsened] = length(keep) - count(!=(0), keep)
    G.stats_dict[:keep_tol] = KeepTol

    println("Evaluating $(length(add_idx)) points to refine on ...")

    # Find children of points we want to add
    G.stats_dict[:n_old] = size(G.grid, 1)
    new_points, new_levels = add_points(G.grid, G.lvl, add_idx, dims = AddDims)
    # Delete points we want to coarsen
    grid_new = G.grid[keep, :]
    lvl_new = G.lvl[keep, :]

    if !isempty(new_points)
        # Gather children points we want to add that are not already on the coarsened grid
        IA = [findfirst(isequal(x), eachrow(new_points)) for x in setdiff(eachrow(new_points),eachrow(grid_new))]
        new_points = new_points[IA, :]
        new_levels = new_levels[IA, :]
        # Also remove points on the blacklist
        IA = [findfirst(isequal(x), eachrow(new_points)) for x in setdiff(eachrow(new_points),eachrow(G.blacklist))]
        new_points = new_points[IA, :]
        new_levels = new_levels[IA, :]
    end
    G.stats_dict[:n_new_points] = size(new_points, 1)
    grid_new = [grid_new; new_points]
    lvl_new = [lvl_new; new_levels]

    # Add parents of all points on new grid
    parents, parent_lvls = add_parents(grid_new, lvl_new)
    if !isempty(parents)
        IA = [findfirst(isequal(x), eachrow(parents)) for x in setdiff(eachrow(parents), eachrow(grid_new))]
        parents = parents[IA, :]
        parent_lvls = parent_lvls[IA, :]
        grid_new = [grid_new; parents]
        lvl_new = [lvl_new; parent_lvls]
    end
    G.stats_dict[:n_new_parents] = size(parents, 1)

    G.stats_dict[:n_new] = size(grid_new, 1)
    G.stats_dict[:n_change] = G.stats_dict[:n_new] - G.stats_dict[:n_old]
    IA = sortperm(eachrow(grid_new))
    grid_new = grid_new[IA, :]
    lvl_new = lvl_new[IA, :]
    update_grid!(G, grid_new, lvl_new)

    # Report summary of results
    println("Mean level of abs(aH): $(mean(abs.(aH[:]))). Maximum level of grid: $(maximum(lvl_new))")
    println("Nodes coarsened: $(G.stats_dict[:coarsened]), children added: $(G.stats_dict[:n_new_points]), parents added: $(G.stats_dict[:n_new_parents]). Net change: $(G.stats_dict[:n_old]) -> $(G.stats_dict[:n_new]) => $(G.stats_dict[:n_change]), blacklisted: $(size(G.blacklist, 1))")
end

function add_points(grid::Matrix{Float64}, levels::Matrix{Int64}, add_idx::Vector{Int64}; dims = nothing)

    d = size(grid, 2)
    # Add children of points where adaptive expansion is needed
    new_points_cell = Vector{Matrix{Float64}}(undef, size(add_idx, 1))
    new_levels_cell = Vector{Matrix{Int64}}(undef, size(add_idx, 1))

    for i = 1:size(add_idx, 1)
        point = grid[add_idx[i], :]
        l = levels[add_idx[i], :]
        level_new = l .+ 1
        h_new = 2.0 .^(-level_new)
        new_points = [point' .+ diagm(h_new)
                      point' .- diagm(h_new)]
        new_levels = repeat(l' .+ diagm(ones(Int, d)), 2)
        if !isnothing(dims)
            # Optional: Add only points in certain dimensions
            new_points = new_points[[dims; d .+ dims], :]
            new_levels = new_levels[[dims; d .+ dims], :]
        end
        outside = (minimum(new_points, dims=2) .< 0 .|| maximum(new_points, dims=2) .> 1)[:]
        new_points_cell[i] = new_points[.!outside, :]
        new_levels_cell[i] = new_levels[.!outside, :]
    end

    new_points = reduce(vcat, new_points_cell)
    new_levels = reduce(vcat, new_levels_cell)
    new_points_rows = eachrow(new_points)
    IA = unique(i -> new_points_rows[i], eachindex(new_points_rows))
    new_points = new_points[IA, :] # unsorted
    new_levels = new_levels[IA, :]

    return new_points, new_levels
end

function add_parents(grid::Matrix{Float64}, levels::Matrix{Int64})

    d = size(grid, 2)
    maxl = maximum(levels) + 1
    parent_mat, full_grid_vec, l_vec = find_parents(maxl)

    parents_cell = Vector{Matrix{Float64}}(undef, maxl * d)
    parent_levels_cell = Vector{Matrix{Int64}}(undef, maxl * d)
    parents_old = copy(grid)
    parent_levels_old = copy(levels)

    for i = 1:(maxl * d)
        parents, parent_levels = add_same_dim_parents(parents_old, parent_levels_old, parent_mat, full_grid_vec, l_vec, maxl)
        # If new parent was already in parents_old, drop it
        # This command also removes duplicates in parents
        # At this point, parents_old has:
        # (1) been used in add_same_dim_parents, so don't need to again
        # (2) been added to parents_cell, so don't need to add
        IA = [findfirst(isequal(x), eachrow(parents)) for x in setdiff(eachrow(parents), eachrow(parents_old))]
        parents = parents[IA, :]
        parent_levels = parent_levels[IA, :]
        
        if isempty(parents)
            break
        end

        parents_cell[i] = parents
        parent_levels_cell[i] = parent_levels
        
        parents_old = copy(parents)
        parent_levels_old = copy(parent_levels)
    end
    idx_def = filter(i -> isassigned(parents_cell,i), 1:length(parents_cell))
    if !isempty(parents_cell[idx_def])
        parents = reduce(vcat, parents_cell[idx_def])
        parent_levels = reduce(vcat, parent_levels_cell[idx_def])
    else
        parents = Float64[]
        parent_levels = Int[]
    end
    return parents, parent_levels
end

function find_parents(maxl::Int64)
    nmax = 2^maxl
    full_grid_cell = Vector{Vector{Float64}}(undef, maxl+1)
    full_grid_cell[1] = [0.5]
    full_grid_cell[2] = [0.0, 1.0]
    l_cell = Vector{Vector{Int64}}(undef, maxl+1)
    l_cell[1] = [0]
    l_cell[2] = ones(Int, 2)
    for i = maxl+1:-1:3
        full_grid_cell[i] = setdiff(0:2.0^(-i+1):1, 0:2.0^(-i+2):1)
        l_cell[i] = fill(i-1, length(full_grid_cell[i]))
    end
    full_grid_vec = reduce(vcat, full_grid_cell)
    l_vec = reduce(vcat, l_cell)
    IA = sortperm(full_grid_vec)
    full_grid_vec = full_grid_vec[IA]
    l_vec = l_vec[IA]

    # Find all ancestors (parents) for each node
    # (i, j) in parent_mat denotes that node j is parent of node i
    icell = Vector{Vector{Float64}}(undef, maxl)
    jcell = Vector{Vector{Float64}}(undef, maxl)
    icell[1] = [0.0, 1.0]
    jcell[1] = [0.5, 0.5]
    for l = 2:maxl
        h = 2.0^(-l)
        temp = full_grid_cell[l+1]
        icell[l] = repeat(temp, 2)
        jcell[l] = [temp .+ h; temp .- h]
    end
    indices_i = Int.(reduce(vcat, icell) .* nmax) .+ 1
    indices_j = Int.(reduce(vcat, jcell) .* nmax) .+ 1
    parent_mat = sparse(indices_i, indices_j, 1.0, nmax+1, nmax+1)
    for l = 1:maxl
        parent_mat = parent_mat * (parent_mat .+ sparse(1.0I, nmax+1, nmax+1))
    end
    parent_mat = min.(parent_mat, 1.0)

    return parent_mat, full_grid_vec, l_vec
end

function add_same_dim_parents(grid, levels, parent_mat, full_grid_vec, l_vec, maxl)
    d = size(grid, 2)
    nmax = 2^maxl

    parents_cell = Vector{Matrix{Float64}}(undef, d)
    parents_levels_cell = Vector{Matrix{Int64}}(undef, d)
    for k = 1:d
        idx = Int.(grid[:, k] .* nmax) .+ 1
        grid_idx = findnz(parent_mat[idx, :])[1]
        parent_idx = findnz(parent_mat[idx, :])[2]
        temp = grid[grid_idx, :]
        temp[:, k] .= full_grid_vec[parent_idx]
        parents_cell[k] = temp

        temp = levels[grid_idx, :]
        temp[:, k] .= l_vec[parent_idx]
        parents_levels_cell[k] = temp
    end
    parents = reduce(vcat, parents_cell)
    parent_levels = reduce(vcat, parents_levels_cell)

    return parents, parent_levels
end


"""
Update grid after adaptation

INPUTS:
- G: Grid struct
- grid: New grid points
- lvl: Levels associated with new grid points

"""
function update_grid!(G::Grid, grid_new, lvl_new)

    @assert size(grid_new, 2) == G.d "Updated grid must have same dimensionality as previous grid."
    G.BH_adapt = H_basis(grid_new, lvl_new, G.grid, G.lvl) * G.H_comp # project value function on the old grid to the new adapted grid
    G.h = 2.0 .^(-lvl_new)
    G.J = size(grid_new, 1)
    _, G.H_comp = gen_H_mat(grid_new, lvl_new)
    G.grid = grid_new
    G.lvl = lvl_new

    # Economic units, defined by min and max inputs
    G.value = grid_to_value(G)
    # For sufficiently small grids, use full matrices instead
    G.sparse = G.J > 100
    gen_bound_grid!(G)
    gen_FD_interior!(G)
end

grid_to_value(G::Grid) = G.grid .* G.range .+ G.min
