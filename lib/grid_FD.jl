"""
Constructs interior values of FD operators
"""
function gen_FD_interior!(G::Grid)
    grid_to_bound = G.bound_grid_dict[:grid_to_bound]

    for k = 1:G.d
        bound_grid = G.bound_grid_dict[:grid][k]
        bound_lvl = G.bound_grid_dict[:lvl][k]
        n_bound = size(bound_grid, 1)
        BH_grid_to_bound_comp = G.bound_grid_dict[:BH_grid_to_bound_comp][k]
        bound_Hk = G.bound_grid_dict[:bound_Hk][k]
        bound_Ek = G.bound_grid_dict[:bound_Ek][k]
        
        interior_idx = findall(0.0 .< bound_grid[:, k] .< 1.0)
        left_neighbor_bound, right_neighbor_bound = G.bound_grid_dict[:left_neighbor_bound][k], G.bound_grid_dict[:right_neighbor_bound][k]
        left_interior = left_neighbor_bound[interior_idx]
        right_interior = right_neighbor_bound[interior_idx]
        left_dist = bound_grid[interior_idx, k] .- bound_grid[left_interior, k]
        right_dist = bound_grid[right_interior, k] .- bound_grid[interior_idx, k]

        DF = sparse([interior_idx; interior_idx],
            [interior_idx; right_interior],
            [-1 ./ right_dist; 1 ./ right_dist],
            n_bound, n_bound)
        D1F_interior = bound_Ek * DF * bound_Hk * BH_grid_to_bound_comp ./ G.range[k]
        # Divided by G.range[k], transform the economic values to 0-1 grid values
        G.DS_interior_dict[:D1F][k] = D1F_interior[grid_to_bound[k], :]

        DB = sparse([interior_idx; interior_idx],
            [interior_idx; left_interior],
            [1 ./ left_dist; -1 ./ left_dist],
            n_bound, n_bound)
        D1B_interior = bound_Ek * DB * bound_Hk * BH_grid_to_bound_comp ./ G.range[k]
        G.DS_interior_dict[:D1B][k] = D1B_interior[grid_to_bound[k], :]

        a, b, c = stencil_central1(left_dist, right_dist)
        DC = sparse([interior_idx; interior_idx; interior_idx],
                    [left_interior; interior_idx; right_interior],
                    [a; b; c],
                    n_bound, n_bound)
        D1C_interior = bound_Ek * DC * bound_Hk * BH_grid_to_bound_comp ./ G.range[k]
        G.DS_interior_dict[:D1C][k] = D1C_interior[grid_to_bound[k], :]

        if issubset(k, G.dxx_dims)
            a, b, c =  stencil_central2(left_dist, right_dist) # !!
            D2 = sparse([interior_idx; interior_idx; interior_idx],
                        [left_interior; interior_idx; right_interior],
                        [a; b; c],
                        n_bound, n_bound)
            D2_interior = bound_Ek * D2 * bound_Hk * BH_grid_to_bound_comp ./ (G.range[k] ^ 2)
            G.DS_interior_dict[:D2][k] = D2_interior[grid_to_bound[k], :]
        end
    end
end

function stencil_central1(dx_left, dx_right)
    # From https://drive.google.com/file/d/0B81VL20ggLWyVEdLalF6R0NFdWM/view
    a = @. - dx_right / (dx_left * (dx_left + dx_right))
    b = @. (dx_right - dx_left) / (dx_left * dx_right)
    c = @. dx_left / (dx_right * (dx_left + dx_right))
    return a, b, c
end

function stencil_central2(dx_left, dx_right)

    # From Sundqvist and Veronis (1970)
    # and https://drive.google.com/file/d/0B81VL20ggLWyVEdLalF6R0NFdWM/view
    # Second derivative operator
    a = @.  2.0 / (dx_left * (dx_left + dx_right))
    b = @. -2.0 / (dx_left * dx_right)
    c = @.  2.0 / (dx_right * (dx_left + dx_right))
    return a, b, c
end

"""
Constructs FD operators for given boundary conditions:
- Most of the work focuses on computing boundary values of the FD Matrix
- Finally, combine those boundary values with the interior FD Matrix (create_aux_fields!)

INPUTS:
- G: Grid struct
- BC: Boundary condition inputs. This is a (d x 1) cell, where each
element is a struct with the following properties
- left.type: type of left boundary condition
- right.type: type of right boundary condition
- left.f: values associated with left boundary condition
- right.f: values associated with right boundary condition
- name: (Optional) name associated with boundary condition, useful when
using multiple boundary conditions for the same grid
"""
function gen_FD!(G::Grid, BC::Vector{Dict}; name = :Main)
    # If BCs have not changed for a dimension and G.DS appears correctly
    # populated, skip computation of that dimension
    compute_dims = findall(k -> !(haskey(G.BC_dict, name) && G.BC_dict[name][k] == BC[k] && G.J == size(G.DS_boundary_dict[Symbol(name, :D1F)], k)), 1:G.d)
    n_dims = length(compute_dims)
    G.BC_dict[name] = BC # vector of length G.d

    grid_to_bound = G.bound_grid_dict[:grid_to_bound]
    const_value = Vector{Vector{Float64}}(undef, G.d)

    for k in compute_dims

        BC_left = BC[k][:lefttype]
        BC_right = BC[k][:righttype]
        if BC_left != :zero && BC_left != :one && haskey(BC[k], :leftfn)
            f_bound_left = BC[k][:leftfn]
        else
            f_bound_left = []
        end
        if BC_right != :zero && BC_right != :one && haskey(BC[k], :rightfn)
            f_bound_right = BC[k][:rightfn]
        else
            f_bound_right = []
        end

        bound_grid = G.bound_grid_dict[:grid][k]
        bound_lvl = G.bound_grid_dict[:lvl][k]
        n_bound = size(bound_grid, 1)
        BH_grid_to_bound_comp = G.bound_grid_dict[:BH_grid_to_bound_comp][k]
        bound_Hk = G.bound_grid_dict[:bound_Hk][k]
        bound_Ek = G.bound_grid_dict[:bound_Ek][k]

        left_neighbor_bound, right_neighbor_bound = G.bound_grid_dict[:left_neighbor_bound][k], G.bound_grid_dict[:right_neighbor_bound][k]
        left_idx_bound = findall(bound_grid[:, k] .== 0)
        n_left_bound = length(left_idx_bound)
        right_idx_bound = findall(bound_grid[:, k] .== 1)
        n_right_bound = length(right_idx_bound)
        boundary_idx = [left_idx_bound; right_idx_bound]
        n_boundary = length(boundary_idx)
        interior_idx = findall(0.0 .< bound_grid[:, k] .< 1.0)

        left_dist = bound_grid[:, k] .- bound_grid[left_neighbor_bound, k]
        right_dist = bound_grid[right_neighbor_bound, k] .- bound_grid[:, k]

        # Distance for exterior ghost nodes (not actually created in memory)
        left_offset = minimum(right_dist[left_idx_bound])
        right_offset = minimum(left_dist[right_idx_bound])
        left_dist[left_idx_bound] .= left_offset
        right_dist[right_idx_bound] .= right_offset
        a1, b1, c1 = stencil_central1(left_dist, right_dist)
        a2, b2, c2 = stencil_central2(left_dist, right_dist)

        # Boundary Difference operator construction
        DF_left = sparse(
            [left_idx_bound; left_idx_bound],
            [left_idx_bound; right_neighbor_bound[left_idx_bound]],
            [-1 ./ right_dist[left_idx_bound]; 1 ./ right_dist[left_idx_bound]],
            n_bound, n_bound
        )
        if in(BC_right, [:zero, :VNF])
            DF_right = spzeros(n_bound, n_bound)
        end
        # Forward difference
        DFH = bound_Ek * (DF_left .+ DF_right) * bound_Hk * BH_grid_to_bound_comp ./ G.range[k]
        G.DS_boundary_dict[Symbol(name, :D1F)][k] = DFH[grid_to_bound[k], :]

        # Backward difference
        DB_right = sparse(
            [right_idx_bound; right_idx_bound],
            [right_idx_bound; left_neighbor_bound[right_idx_bound]],
            [1 ./ left_dist[right_idx_bound]; -1 ./ left_dist[right_idx_bound]],
            n_bound, n_bound
        )
        if in(BC_left, [:zero, :VNB])
            DB_left = spzeros(n_bound, n_bound)
        end
        DBH = bound_Ek * (DB_left .+ DB_right) * bound_Hk * BH_grid_to_bound_comp ./ G.range[k]
        G.DS_boundary_dict[Symbol(name, :D1B)][k] = DBH[grid_to_bound[k], :]

        # Central difference
        if in(BC_left, [:zero, :VNB])
            DC_left = sparse(
                [left_idx_bound; left_idx_bound; left_idx_bound],
                [left_idx_bound; left_idx_bound; right_neighbor_bound[left_idx_bound]],
                [a1[left_idx_bound]; b1[left_idx_bound]; c1[left_idx_bound]],
                n_bound, n_bound
            )
        end
        if in(BC_right, [:zero, :VNF])
            DC_right = sparse(
                [right_idx_bound; right_idx_bound; right_idx_bound],
                [left_neighbor_bound[right_idx_bound]; right_idx_bound; right_idx_bound],
                [a1[right_idx_bound]; b1[right_idx_bound]; c1[right_idx_bound]],
                n_bound, n_bound
            )
        end
        DCH = bound_Ek * (DC_left .+ DC_right) * bound_Hk * BH_grid_to_bound_comp ./ G.range[k]
        G.DS_boundary_dict[Symbol(name, :D1C)][k] = DCH[grid_to_bound[k], :]

        if issubset(k, G.dxx_dims)
            if in(BC_left, [:zero, :VNB])
                D2_left = sparse(
                    [left_idx_bound; left_idx_bound; left_idx_bound],
                    [left_idx_bound; left_idx_bound; right_neighbor_bound[left_idx_bound]],
                    [a2[left_idx_bound]; b2[left_idx_bound]; c2[left_idx_bound]],
                    n_bound, n_bound
                )
            end
            if in(BC_right, [:zero, :VNF])
                D2_right = sparse(
                    [right_idx_bound; right_idx_bound; right_idx_bound],
                    [left_neighbor_bound[right_idx_bound]; right_idx_bound; right_idx_bound],
                    [a2[right_idx_bound]; b2[right_idx_bound]; c2[right_idx_bound]],
                    n_bound, n_bound
                )
            end
            D2H = bound_Ek * (D2_left .+ D2_right) * bound_Hk * BH_grid_to_bound_comp ./ G.range[k]^2
            G.DS_boundary_dict[Symbol(name, :D2)][k] = D2H[grid_to_bound[k], :]
        end

        # Const term construction
        G.DS_const_dict[Symbol(name, :D1F)][k] = zeros(n_bound)
        G.DS_const_dict[Symbol(name, :D1B)][k] = zeros(n_bound)
        G.DS_const_dict[Symbol(name, :D1C)][k] = zeros(n_bound)
        G.DS_const_dict[Symbol(name, :D2)][k] = zeros(n_bound)
        G.DS_const_dict[Symbol(:DCH_, name)][k] = spzeros(n_bound, n_boundary)

        const_c_left = zeros(length(left_idx_bound))
        const_c_right = zeros(length(left_idx_bound))
        
        # Note: No const terms for reflecting boundaries
        if BC_left == :VNB
            const_c_left .= f_bound_left(bound_grid[left_idx_bound, :])[:] .* (-left_offset)
            
            # VNB: fx(0) = (f(0) - f(-1)) / h => f(-1) = f(0) - h*fx(0)
            G.DS_const_dict[Symbol(name, :D1B)][k][left_idx_bound] .= const_c_left .* (-1/left_offset)
            G.DS_const_dict[Symbol(name, :D1C)][k][left_idx_bound] .= const_c_left .* a1[left_idx_bound]
            if n_left_bound == 1
                G.DS_const_dict[Symbol(:DCH_, name)][k][left_idx_bound, 1:n_left_bound] .= a1[n_left_bound]
            else
                G.DS_const_dict[Symbol(:DCH_, name)][k][left_idx_bound, 1:n_left_bound] = spdiagm(a1[1:n_left_bound])
            end
            if issubset(k, G.dxx_dims)
                G.DS_const_dict[Symbol(name, :D2)][k][left_idx_bound] .= const_c_left .* a2[left_idx_bound]
            end
        end
        if BC_right == :VNF
            const_c_right .= f_bound_right(bound_grid[right_idx_bound, :])[:] .* right_offset
            
            # VNF: fx(J) = (f(J+1) - f(J)) / h => f(J+1) = f(J) + h*fx(J)
            G.DS_const_dict[Symbol(name, :D1F)][k][right_idx_bound] .= const_c_right .* (1/right_offset) # !!
            G.DS_const_dict[Symbol(name, :D1C)][k][right_idx_bound] .= const_c_right .* c1[right_idx_bound]
            if n_right_bound == 1
                G.DS_const_dict[Symbol(:DCH_, name)][k][right_idx_bound, end-n_right_bound+1:end] .= c1[n_right_bound]
            else
                G.DS_const_dict[Symbol(:DCH_, name)][k][right_idx_bound, end-n_right_bound+1:end] = spdiagm(c1[1:n_right_bound])
            end
            if issubset(k, G.dxx_dims)
                G.DS_const_dict[Symbol(name, :D2)][k][right_idx_bound] .= const_c_right .* c2[right_idx_bound]
            end
        end
        const_value[k] = [const_c_left; const_c_right]

        G.DS_const_dict[Symbol(name, :D1B)][k] = G.DS_const_dict[Symbol(name, :D1B)][k][grid_to_bound[k]]
        G.DS_const_dict[Symbol(name, :D1F)][k] = G.DS_const_dict[Symbol(name, :D1F)][k][grid_to_bound[k]]
        G.DS_const_dict[Symbol(name, :D1C)][k] = G.DS_const_dict[Symbol(name, :D1C)][k][grid_to_bound[k]]
        if issubset(k, G.dxx_dims)
            G.DS_const_dict[Symbol(name, :D2)][k] = G.DS_const_dict[Symbol(name, :D2)][k][grid_to_bound[k]]
        end

        create_aux_fields!(G; dims = k, name = name)
    end
end

function create_aux_fields!(G::Grid; dims, name)
    # Dense matrix operations for sparse matrix is slow, especially indexing
    # assignment. So convert the sparse matrix into (I,J,V) format using findnz,
    # manipulate the values or the structure in the dense vectors (I,J,V), and
    # then reconstruct the sparse matrix.
    if issubset(dims, G.dxx_dims)
        fields = [:D1F, :D1B, :D1C, :D2]
    else
        fields = [:D1F, :D1B, :D1C]
    end

    for field in fields
        i_interior, j_interior, s_interior = findnz(G.DS_interior_dict[field][dims])
        i_boundary, j_boundary, s_boundary = findnz(G.DS_boundary_dict[Symbol(name, field)][dims])
        G.DSijs_dict[Symbol(name, field)][dims] = [i_interior j_interior s_interior
                                                   i_boundary j_boundary s_boundary]
        # For sufficiently small grids, use full matrices instead
        if !G.sparse
            G.DFull_dict[Symbol(name, field)][dims] = collect(G.DS_interior_dict[field][dims] .+ G.DS_boundary_dict[Symbol(name, field)][dims])
        end
    end
end


"""
Constructs finite difference operators given grid, drift, and diffusion

INPUTS:
- G: Grid struct
- mu: (J x d') matrix of drift terms
- sigma: (J x d') matrix of diffusion terms
- dims: (d' x 1) vector, subset of dimensions referenced by mu and sigma
- BC_name: (Optional) name of boundary conditions to use

OUTPUTS:
- mat: FD operator matrix
- bound_const: Residual constant terms

"""
function FD_operator(G::Grid; μ, σ, dims, BC_name = :Main) # build matrix A

    if G.d == 1 && !G.sparse && any(iszero, σ)
        mat = sparse(
            μ .* (μ .>= 0) .* G.DFull_dict[Symbol(BC_name, :D1F)][1] .+
                μ .* (μ .< 0) .* G.DFull_dict[Symbol(BC_name, :D1B)][1]
        )
        return mat, 0
    end

    if G.sparse
        mat = spzeros(G.J, G.J)
    else
        mat = zeros(G.J, G.J)
    end

    bound_const = zeros(G.J)

    #=
        DRIFT
    =#
    for i = 1:length(dims)
        k = dims[i]
        if any(!iszero, μ[:, i])
            if G.sparse
                ijs = [vec_x_spijs(μ[:, i] .* (μ[:, i] .>= 0), G.DSijs_dict[Symbol(BC_name, :D1F)][k])
                       vec_x_spijs(μ[:, i] .* (μ[:, i] .< 0), G.DSijs_dict[Symbol(BC_name, :D1B)][k])]
                mat_drift = sparse(ijs[:, 1], ijs[:, 2], ijs[:, 3], G.J, G.J)
            else
                mat_drift = μ[:, i] .* (μ[:, i] .>= 0) .* G.DFull_dict[Symbol(BC_name, :D1F)][k] .+ μ[:, i] .* (μ[:, i] .< 0) .* G.DFull_dict[Symbol(BC_name, :D1B)][k]
            end
            mat = mat .+ mat_drift

            const_drift = μ[:, i] .* (μ[:, i] .>= 0) .* G.DS_const_dict[Symbol(BC_name, :D1F)][k] .+ μ[:, i] .* (μ[:, i] .< 0) .* G.DS_const_dict[Symbol(BC_name, :D1B)][k]
            bound_const = bound_const .+ const_drift
        end
    end

    #=
        DIFFUSION
    =#
    for i = 1:length(dims)
        k = dims[i]
        if any(!iszero, σ[:, i])
            if G.sparse
                ijs = vec_x_spijs(1/2 .* σ[:, i] .^ 2, G.DSijs_dict[Symbol(BC_name, :D2)][k])
                mat_diffusion = sparse(ijs[:, 1], ijs[:, 2], ijs[:, 3], G.J, G.J)
            else
                mat_diffusion = 1/2 .* σ[:, i] .^ 2 .* G.DFull_dict[Symbol(BC_name, :D2)][k]
            end
            mat = mat .+ mat_diffusion

            const_diffusion = 1/2 .* σ[:, i].^2 .* G.DS_const_dict[Symbol(BC_name, :D2)][k]
            bound_const = bound_const .+ const_diffusion
        end
    end

    return mat, bound_const
end

"""
Performs elementwise multiplication between a n-COLUMN vector and a
(n x n) sparse matrix (stored using the [i,j,s] format), and returns
a (n x 3) matrix of [i,j,s] which can be then fed into
sparse(i,j,s,n,n) to create the product matrix
"""
function vec_x_spijs(v, ijs)

    v_expand = v[Int.(ijs[:, 1])]
    keep = v_expand .!= 0
    ijs_new = ijs[keep, :]
    ijs_new[:, 3] .= ijs_new[:, 3] .* v_expand[keep]

    return ijs_new # ijs_new[:,3] is s/Δa
end

function deriv_sparse(G::Grid, f; operator::Symbol, dims::Int64, name=:Main)
    deriv = (G.DS_interior_dict[operator][dims] .+ G.DS_boundary_dict[Symbol(name, operator)][dims]) * f .+ G.DS_const_dict[Symbol(name, operator)][dims]
end
