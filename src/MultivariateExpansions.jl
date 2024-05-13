module MultivariateExpansions

using MultiIndexing, LinearAlgebra

# Univariate polynomial families
include("univariate_poly.jl")

# Multivariate polynomial assembly and evaluation
include("multivariate_poly.jl")

# Score matching capabilities
include("score.jl")

end
