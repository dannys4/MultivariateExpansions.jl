using MultivariateExpansions
using Test, Random, MultiIndexing, Statistics, UnivariateApprox

@testset "MultivariateExpansions.jl" begin

    @testset "Generic multivariate basis evaluation" begin
        include("multivariate_basis.jl")
    end
    @testset "Score matching" begin
        include("score.jl")
    end
end
