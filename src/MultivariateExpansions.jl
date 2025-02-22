module MultivariateExpansions

using MultiIndexing, LinearAlgebra, UnivariateApprox, ArgCheck
import AcceleratedKernels as AK

# Multivariate basis assembly and evaluation
include("multivariate_basis.jl")

# Score matching capabilities
include("score.jl")

end
