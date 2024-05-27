using MultivariateExpansions
using Test, Random, MultiIndexing, Statistics

Monomials() = MultivariateExpansions.MonicOrthogonalPolynomial(Returns(0.),Returns(0.))

@testset "MultivariateExpansions.jl" begin
    @testset "Generic multivariate polynomial evaluation" begin
        include("polynomial_assembly.jl")
    end

    @testset "Univariate polynomials" begin
        include("univariate_poly.jl")
    end

    @testset "Univariate bases" begin
        include("univariate_basis.jl")
    end

    @testset "Mollified basis" begin
        include("mollified_basis.jl")
    end

    @testset "Score matching" begin
        include("score.jl")
    end
end
