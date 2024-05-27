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

@testset "Mollification evaluation" begin
    @testset "SquaredExponential" begin
        moll = SquaredExponential()
        moll_ref = x-> exp.(-x^2/4)
        moll_eval_ref = moll_ref.(pts)
        moll_diff_ref = (moll_ref.(pts .+ 1e-5) - moll_ref.(pts .- 1e-5))/(2e-5)
        moll_diff2_ref = (moll_ref.(pts .+ 1e-5) + moll_ref.(pts .- 1e-5) - 2moll_ref.(pts))/(1e-10)

        moll_eval = Evaluate.((moll,), pts)

        moll_eval_ratio = moll_eval ./ moll_eval_ref

        # Assert ratio is constant
        @test all(isapprox.(moll_eval_ratio, moll_eval_ratio[1], atol=1e-14))

        moll_eval_diffs = EvalDiff.((moll,), pts)
        moll_eval, moll_diff = first.(moll_eval_diffs), last.(moll_eval_diffs)
        moll_eval_ratio = moll_eval ./ moll_eval_ref
        moll_diff_ratio = moll_diff ./ moll_diff_ref

        # Assert ratio is constant
        @test all(isapprox.(moll_eval_ratio, moll_eval_ratio[1], atol=1e-14))
        @test all(isapprox.(moll_diff_ratio, moll_diff_ratio[1], rtol=1e-9))

        moll_evals = EvalDiff2.((moll,), pts)
        moll_eval_ratio = [m_ev / r_ev for ((m_ev,_,_), r_ev) in zip(moll_evals, moll_eval_ref)]
        moll_diff_ratio = [m_diff / r_diff for ((_,m_diff,_), r_diff) in zip(moll_evals, moll_diff_ref)]
        moll_diff2_ratio = [m_diff2 / r_diff2 for ((_,_,m_diff2), r_diff2) in zip(moll_evals, moll_diff2_ref)]

        # Assert ratio is constant
        @test all(isapprox.(moll_eval_ratio, moll_eval_ratio[1], atol=1e-14))
        @test all(isapprox.(moll_diff_ratio, moll_diff_ratio[1], rtol=1e-9))
        @test all(isapprox.(moll_diff2_ratio, moll_diff2_ratio[1], rtol=1e-5))
    end

    @testset "GaspariCohn" begin
        pts = -5:0.1:5
        bound = 3.
        fd_delta = 1e-5
        gc = GaspariCohn(bound)
        eval_gc_ref = Evaluate.((gc,), pts)
        @test all(eval_gc_ref[abs.(pts) .> bound] .== 0.)

        eval_gc_ref_plus_fd = Evaluate.((gc,), pts .+ fd_delta)
        eval_gc_ref_minus_fd = Evaluate.((gc,), pts .- fd_delta)
        diff_gc_ref = (eval_gc_ref_plus_fd - eval_gc_ref_minus_fd) / (2fd_delta)
        diff2_gc_ref = (eval_gc_ref_plus_fd - 2eval_gc_ref .+ eval_gc_ref_minus_fd) / (fd_delta^2)

        eval_gc_diffs = EvalDiff.((gc,), pts)
        eval_gc, diff_gc = first.(eval_gc_diffs), last.(eval_gc_diffs)
        @test eval_gc ≈ eval_gc_ref atol=1e-14
        @test diff_gc[abs.(pts) .< bound] ≈ diff_gc_ref[abs.(pts) .< bound] rtol=fd_delta
        @test all(diff_gc[abs.(pts) .> bound] .== 0.)

        eval_gc_diffs = EvalDiff2.((gc,), pts)
        eval_gc, diff2_gc = first.(eval_gc_diffs), last.(eval_gc_diffs)
        diff_gc = map(x->x[2], eval_gc_diffs)
        @test eval_gc ≈ eval_gc_ref atol=1e-14
        @test diff_gc[abs.(pts) .< bound] ≈ diff_gc_ref[abs.(pts) .< bound] rtol=fd_delta
        @test all(diff_gc[abs.(pts) .> bound] .== 0.)
        @test diff2_gc[abs.(pts) .< bound] ≈ diff2_gc_ref[abs.(pts) .< bound] rtol=fd_delta
        @test all(diff2_gc[abs.(pts) .> bound] .== 0.)
    end
end

@testset "Mollified basis evaluation" begin
    eval_space_ref = Matrix{Float64}(undef, p+1, N_pts)
    diff_space_ref = similar(eval_space_ref)
    diff2_space_ref = similar(eval_space_ref)
    EvalDiff2!(eval_space_ref, diff_space_ref, diff2_space_ref, basis, pts)

    @testset "No mollification" begin
        # Evaluation
        moll_basis = MollifiedBasis(0, basis, NoMollification())

        eval_space = zeros(p+1, N_pts)
        Evaluate!(eval_space, moll_basis, pts)

        @test eval_space ≈ eval_space_ref atol=1e-14

        out_of_place_eval = Evaluate(p, basis, pts)
        @test out_of_place_eval ≈ eval_space atol=1e-14

        # Derivatives
        diff_space = zeros(p+1, N_pts)
        EvalDiff!(eval_space, diff_space, basis, pts)

        @test eval_space ≈ eval_space_ref atol=1e-14
        @test diff_space ≈ diff_space_ref atol=1e-14

        out_of_place_eval, out_of_place_diff = EvalDiff(p, basis, pts)
        @test out_of_place_eval ≈ eval_space atol=1e-14
        @test out_of_place_diff ≈ diff_space atol=1e-14

        # Second derivatives
        diff2_space = zeros(p+1, N_pts)
        EvalDiff2!(eval_space, diff_space, diff2_space, basis, pts)

        diff2_space_ref = zeros(p+1, N_pts)
        EvalDiff2!(eval_space_ref, diff_space_ref, diff2_space_ref, basis, pts)

        @test eval_space ≈ eval_space_ref atol=1e-14
        @test diff_space ≈ diff_space_ref atol=1e-14
        @test diff2_space ≈ diff2_space_ref atol=1e-14

        out_of_place_eval, out_of_place_diff, out_of_place_diff2 = EvalDiff2(p, basis, pts)
        @test out_of_place_eval ≈ eval_space atol=1e-14
        @test out_of_place_diff ≈ diff_space atol=1e-14
        @test out_of_place_diff2 ≈ diff2_space atol=1e-14
    end

    @testset "SquaredExponential" begin
        # Start mollifying at cubics
        start_degree = 3
        moll = SquaredExponential()
        moll_eval = Evaluate.((moll,), pts)
        moll_basis = MollifiedBasis(start_degree, basis, moll)

        eval_space = zeros(p+1, N_pts)
        Evaluate!(eval_space, moll_basis, pts)

        @test eval_space[1:start_degree,:] ≈ eval_space_ref[1:start_degree,:] atol=1e-14
        ratio = eval_space[start_degree+1:end,:] ./ eval_space_ref[start_degree+1:end,:]

        @test ratio ≈ repeat(moll_eval',p+1-start_degree,1) atol=1e-14

        out_of_place_eval = Evaluate(p, moll_basis, pts)
        @test out_of_place_eval ≈ eval_space atol=1e-14

        # Derivatives
        diff_space = zeros(p+1, N_pts)
        EvalDiff!(eval_space, diff_space, moll_basis, pts)

        @test eval_space[1:start_degree,:] ≈ eval_space_ref[1:start_degree,:] atol=1e-14
        @test diff_space[1:start_degree,:] ≈ diff_space_ref[1:start_degree,:] atol=1e-14

        fd_delta = 1e-5
        eval_moll_plus_fd = Evaluate(p, moll_basis, pts .+ fd_delta)
        eval_moll_minus_fd = Evaluate(p, moll_basis, pts .- fd_delta)
        diff_space_fd = (eval_moll_plus_fd - eval_moll_minus_fd) / (2fd_delta)

        @test diff_space ≈ diff_space_fd rtol=10fd_delta
    end
end