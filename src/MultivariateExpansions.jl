module MultivariateExpansions

using MultiIndexing, LinearAlgebra, UnivariateApprox, ArgCheck
import AcceleratedKernels as AK
using KernelAbstractions: allocate, zeros, GPU, get_backend

# Multivariate basis assembly and evaluation
include("multivariate_basis.jl")

# Score matching capabilities
include("score.jl")

end
