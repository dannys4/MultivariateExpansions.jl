using MultivariateExpansions
using Test, Random, MultiIndexing, Statistics

@testset "MultivariateExpansions.jl" begin
    @testset "Generic polynomial evaluation" begin
        include("polynomial_assembly.jl")
    end

    @testset "Score matching" begin
        include("score.jl")
    end
end
