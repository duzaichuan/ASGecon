function get_projection_matrix(G::Grid, points, lvl_points)

    overlap_grid_idx = indexin(eachrow(points), eachrow(G.grid))
    overlap_bool = map(!isnothing, overlap_grid_idx)
    nonoverlap_bool = .!overlap_bool
    overlap_bound_idx = findall(!isnothing, overlap_grid_idx)
    BH_comp = sparse( # the overlaps are identities
        overlap_bound_idx, overlap_grid_idx[overlap_bool],
        ones(count(overlap_bool)),
        size(points, 1), G.J
    )
    if count(nonoverlap_bool) == 0
        return BH_comp
    else
        # use H_basis to compute hierachical hats of nonoverlap points
        # the overlap points share with the known grid
        BH_comp_nonoverlap = H_basis(points[nonoverlap_bool, :],
                                     lvl_points[nonoverlap_bool, :],
                                     G.grid, G.lvl) * G.H_comp
        BH_comp[nonoverlap_bool, :] = BH_comp_nonoverlap # if use .= will be dense matrix

        return BH_comp
    end
end

"Constructs H basis function FOR ONE POINT"
function sparse_project(G::Grid, points, x)
    n = size(points, 1)
    x_project = zeros(n, size(x, 2))

    if n < 5000
        xH = G.H_comp * x
        for i = 1:n
            BH = H_basis_small(points[i, :], G.grid, G.lvl, G.h)
            x_project[i, :] = BH * xH
        end
    else
        BH = get_projection_matrix(G, points, points .+ 100)
        x_project = BH * x
    end
    return x_project
end
