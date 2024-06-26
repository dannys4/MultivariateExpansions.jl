using MultivariateExpansions
using Test, Random, MultiIndexing, Statistics

Monomials() = MultivariateExpansions.MonicOrthogonalPolynomial(Returns(0.),Returns(0.))

@testset "MultivariateExpansions.jl" begin
    @testset "Univariate polynomials" begin
        include("univariate_poly.jl")
    end

    @testset "Univariate bases" begin
        include("univariate_basis.jl")
    end

    @testset "Mollified basis" begin
        include("mollified_basis.jl")
    end

    @testset "Generic multivariate basis evaluation" begin
        include("multivariate_basis.jl")
    end
    @testset "Score matching" begin
        include("score.jl")
    end
end
