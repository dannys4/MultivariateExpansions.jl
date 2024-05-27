module MultivariateExpansions

using MultiIndexing, LinearAlgebra, MuladdMacro

# Univariate basis functions
include("univariate_basis.jl")

# Univariate polynomial families
include("univariate_poly.jl")

# Univariate mollified basis
include("mollified_basis.jl")

# Multivariate polynomial assembly and evaluation
include("multivariate_poly.jl")

# Score matching capabilities
include("score.jl")

end
