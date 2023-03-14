module MacroASG

using LinearAlgebra
using Statistics
using SparseArrays
import Combinatorics: combinations

export Grid, setup_grid
export get_projection_matrix
export FD_operator, deriv_sparse, gen_FD!
export adapt_grid!
export newton_nonlin

include("grid_setup.jl")
include("grid_hierarchical.jl")
include("grid_projection.jl")
include("grid_FD.jl")
include("grid_adaption.jl")
include("utils/newton_nonlin.jl")

end # module
