module MultivariateExpansions

using MultiIndexing, LinearAlgebra, MuladdMacro, UnivariateApprox

# Multivariate basis assembly and evaluation
include("multivariate_basis.jl")

# Score matching capabilities
include("score.jl")

end
