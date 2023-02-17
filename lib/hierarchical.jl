"""
FUNCTION: gen_H_mat # hierarchical surplus operator
Inputs
  - grid:  (n X d) matrix of the grid, where each row is coordinates
  - level: (n X d) matrix of level of each node in grid

Outputs
  - H: H basis hierarchization surplus transformation matrix for each dimension
  - H_comp: d-dimensional hierarchization transformation matrix
  - komit: omit certain dimension, used for computing the FD operator

"""
function gen_H_mat(grid, lvl; komit = 0)

    n = size(grid, 1)
    d = size(grid, 2)
    H_comp = sparse(1.0I, n, n) # Level 0 hierarchization coefficients are 1, as init
    H = fill(sparse(1.0I, n, n), d) # Level 0 is already done by speye command
    if komit == 0
        compute_dims = 1:d
    else
        compute_dims = setdiff(1:d, komit)
    end

    for k in compute_dims

        dim_min = minimum(grid[:, k])
        dim_max = maximum(grid[:, k])
        left = grid[:, k] .== dim_min
        right = grid[:, k] .== dim_max

        for l = 1:maximum(lvl[:, k])
            subgrid_idx = findall(<=(l), lvl[:, k])
            left_neighbor, right_neighbor = find_neighbors(grid, subgrid_idx, k)
            # Note: Neighbors will be wrong for boundaries without following, but
            # they are never used for H and D matrix generation, so can be ignored
            left_neighbor[left] .= 0 # NaN
            right_neighbor[right] .= 0 # NaN

            if l == 1
                idx_left = findall(==(0), grid[:, k])
                idx_right = findall(==(1), grid[:, k])
                H[k] = H[k] .+ sparse(
                    [idx_left; idx_right],
                    [right_neighbor[idx_left]; left_neighbor[idx_right]],
                    -1.0,
                    n, n
                )
            else
                indices = findall(==(l), lvl[:, k])
                H[k] = H[k] .+ sparse(
                    [indices; indices],
                    [right_neighbor[indices]; left_neighbor[indices]],
                    -0.5,
                    n, n
                )
            end
        end
        H_comp = H_comp * H[k]
    end

    return H, H_comp
end

"""
Function: find_neighbors(grid, lvl)

Return the indices of neighbors.
The reference is the original grid matrix.

Notice that the naming of indices variables:
_idx contains the original indices, whereas _sdx means that indices are indexing the
sorted subgrid, and _ssdx indexing the subgrid of some sorted subgrids.
"""
function find_neighbors(grid, subgrid_idx, k)

    d = size(grid, 2)
    n = size(grid, 1)
    left_neighbors = zeros(Int, n)
    right_neighbors = zeros(Int, n)
    subgrid = grid[subgrid_idx, :]
    n_subgrid = size(subgrid, 1)

    subgrid_sorted_sdx = sortperm(eachrow(subgrid), by=x->(x[setdiff(1:d, k)], x[k])) #!! time consuming
    subgrid_sorted_idx = subgrid_idx[subgrid_sorted_sdx]

    left_sdx = [1; 1:n_subgrid-1]
    right_sdx = [2:n_subgrid; 1]
    left_idx = subgrid_sorted_idx[left_sdx]
    right_idx = subgrid_sorted_idx[right_sdx]
    left_neighbors[subgrid_sorted_idx] .= left_idx
    right_neighbors[subgrid_sorted_idx] .= right_idx

    return left_neighbors, right_neighbors
end


"""
FUNCTION: H_basis(points, lvl_points, grid, lvl_grid) # hierarchical basis functions

Project (interpolate) the (new) points to the mother hat functions phi of
each level combinations' corresponding hierachical basis.

INPUTS:
- points: (m x d) matrix of points for projection
- lvl_points: (m x d) matrix of levels associated with points
              (optional - enter empty [] if points are not associated with grid)
- grid: (n x d) matrix of grid nodes
- lvl_grid: (n x d) matrix of levels associated with grid

The benefit of representing the d-dimensional grid points in a n-by-d matrix, where
n, number of rows, is the total size of points:

- This form of storing is consistent with the value function representation, one column concatenate in order.
- In particular, this n-by-d stores the indices of the value function.
- Simplify the multi-dimensional points to an 'one' dimensional column representation,
- so tensor products can be easily handled and understood.
"""
function H_basis(points, lvl_points, grid, lvl_grid)

    n_points = size(points,1)
    n_grid = size(grid, 1)
    grid_groups, glvl_groups, ggroup_sdx, glvl_sorted_idx = split_by_level(grid, lvl_grid)
    points_groups, plvl_groups, pgroup_sdx, plvl_sorted_idx = split_by_level(points, lvl_points)
    output = Vector{Matrix{Float64}}(undef, size(glvl_groups, 1))

    # each loop computes each level's (low2high) hierarchical hats
    Threads.@threads for i = 1:size(glvl_groups, 1)
        lvl_temp = glvl_groups[i, :]
        d_temp = lvl_temp .> 0 # Level 0's hat function is a constant? !! may lead to empty [] following
        grid_sdx = ggroup_sdx[i]
        grid_temp = grid_groups[i][:, d_temp]
        h_temp = 2.0 .^ (- lvl_temp[d_temp]) # identifies the hat phi function for a specific level

        # The hierachical structure requires points being evaluated in hat
        # function in all lower levels (adding up later)
        plvl_groups_ssdx = all(plvl_groups .>= repeat(lvl_temp, 1, size(plvl_groups, 1))', dims=2)[:]
        # the size of point indices is updated crpt the grid level group
        points_idx = plvl_sorted_idx[reduce(vcat, pgroup_sdx[plvl_groups_ssdx], init = Int[])]
        points_temp = points[points_idx, d_temp]

        phi = ones(length(points_idx), length(grid_sdx), count(!=(0), d_temp))
        # construction of d-piecewise linear hat functions centered on grid
        # points in each grid level combination
        for k = 1:count(!=(0), d_temp)
            phi[:,:,k] .= max.(1 .- abs.(points_temp[:, k] .- grid_temp[:, k]') ./ h_temp[k], 0)
        end
        # tensor product
        phi = prod(phi, dims=3)
        phi_sdx = findall(!iszero, phi)
        x_sdx = getindex.(phi_sdx, 1) # convert the Cartisian indices to vector of integers
        y_sdx = getindex.(phi_sdx, 2)
        output[i] = [points_idx[x_sdx] grid_sdx[y_sdx] phi[phi_sdx]] # phi of Float64 makes the indices not Int64
    end

    output = reduce(vcat, output)
    output[:, 2] .= glvl_sorted_idx[Int.(output[:, 2])] # translate the grid sub-indices to original ones
    BH = sparse(Int.(output[:, 1]), Int.(output[:, 2]), output[:, 3], n_points, n_grid) # transform the content of index matrix to indices of BH
    # Use index data (output[:,1:2]) to construct the projection matrix. Each
    # row of BH contains the total φs of every hierachical hat functions where a
    # projected point locates. To see this clearer, check out:
    # output_korted = sortslices(output, dims=1)
    return BH
end


"""
Function: split_by_level(grid, lvl)

- Split grids into groups of possible level combinations
- Each group of levels
"""
function split_by_level(grid, lvl)

    lvl_sorted_idx = sortperm(eachrow(lvl)) # crucial for mapping sorted-grid indices to original indices
    lvl_lorted = lvl[lvl_sorted_idx,:]
    grid_sorted = grid[lvl_sorted_idx,:] # grid sorted by level groups
    lvl_groups = unique(lvl_lorted,dims=1) # return
    n_groups = size(lvl_groups,1)
    group_sdx = Vector{Vector{Int64}}(undef, n_groups) # return
    grid_groups = Vector{Matrix{Float64}}(undef, n_groups) # return
    lvl_lorted_rows = eachrow(lvl_lorted) # in Julia<1.9 the collect is needed

    for i = 1:n_groups
        # count observations in each group
        obs_per_group = count(==(lvl_groups[i,:]), lvl_lorted_rows)
        if i == 1
            group_sdx[i] = 1:obs_per_group
        else
            group_sdx[i] = range(last(group_sdx[i-1])+1, length=obs_per_group)
        end
        # collect grid points according to level sorted grid points
        grid_groups[i] = grid_sorted[group_sdx[i], :]
    end

    return grid_groups, lvl_groups, group_sdx, lvl_sorted_idx
end

function H_basis_small(point, grid, lvl_grid, h)
    phi = max.(1 .- abs.(point' .- grid) ./ h, 0)
    phi[lvl_grid .== 0] .= 1.0
    phi = prod(phi, dims = 2)'
end
