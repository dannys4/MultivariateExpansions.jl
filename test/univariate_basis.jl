p, N_pts = 6, 100
rng = Xoshiro(284028)
pts = 2rand(rng, N_pts) .- 1
basis = JacobiPolynomial(0.5, 0.75)
import MultivariateExpansions: Evaluate, EvalDiff, EvalDiff2
# Mollification m(x) = x
struct NoMollification <: MultivariateExpansions.Mollifier end
MultivariateExpansions.Evaluate(::NoMollification, x) = one(x)
MultivariateExpansions.EvalDiff(::NoMollification, x) = (one(x), zero(x))
MultivariateExpansions.EvalDiff2(::NoMollification, x) = (one(x), zero(x), zero(x))

basis = JacobiPolynomial(0.75, 0.52)

@testset "Basis evaluation" begin
    @testset "No mollification" begin
        # Evaluation
        moll_basis = MollifiedBasis(0, basis, NoMollification())

        eval_space_ref = zeros(p+1, N_pts)
        Evaluate!(eval_space_ref, basis, pts)

        eval_space = zeros(p+1, N_pts)
        Evaluate!(eval_space, moll_basis, pts)

        @test eval_space ≈ eval_space_ref atol=1e-14

        out_of_place_eval = Evaluate(p, basis, pts)
        @test out_of_place_eval ≈ eval_space atol=1e-14

        # Derivatives
        diff_space = zeros(p+1, N_pts)
        EvalDiff!(eval_space, diff_space, basis, pts)

        diff_space_ref = zeros(p+1, N_pts)
        EvalDiff!(eval_space_ref, diff_space_ref, basis, pts)

        @test eval_space ≈ eval_space_ref atol=1e-14
        @test diff_space ≈ diff_space_ref atol=1e-14

        out_of_place_eval, out_of_place_diff = EvalDiff(p, basis, pts)
        @test out_of_place_eval ≈ eval_space atol=1e-14
        @test out_of_place_diff ≈ diff_space atol=1e-14

        # Second derivatives
        diff2_space = zeros(p+1, N_pts)
        # EvalDiff2!(eval_space, diff_space, diff2_space, basis, pts)

        diff2_space_ref = zeros(p+1, N_pts)
        # EvalDiff2!(eval_space_ref, diff_space_ref, diff2_space_ref, basis, pts)

        @test eval_space ≈ eval_space_ref atol=1e-14
        @test diff_space ≈ diff_space_ref atol=1e-14
        @test diff2_space ≈ diff2_space_ref atol=1e-14

        # out_of_place_eval, out_of_place_diff, out_of_place_diff2 = EvalDiff2(p, basis, pts)
        # @test out_of_place_eval ≈ eval_space atol=1e-14
        # @test out_of_place_diff ≈ diff_space atol=1e-14
        # @test out_of_place_diff2 ≈ diff2_space atol=1e-14
    end

    @testset "SquaredExponential" begin
        moll_basis = MollifiedBasis(0, basis, SquaredExponential())
    end
end