d, p, N_pts = 5, 5, 1000
rng = Xoshiro(284028)
mset = CreateTotalOrder(d, p)
fmset = FixedMultiIndexSet(mset)
pts = randn(rng, N_pts, d)
univariateEvals = ntuple(j->reduce(hcat, pts[:,j] .^ k for k in 0:p)', d)
@testset "Polynomial Assembly" begin
    coeffs = randn(rng, length(fmset))
    out = zeros(N_pts)
    basisAssembly!(out, fmset, coeffs, univariateEvals)
    all_out_correct = true
    for pt_idx in 1:N_pts
        expected_pt_eval = 0.
        @inbounds for (coeff_idx,midx) in enumerate(mset)
            idx_eval = prod(pts[pt_idx,:] .^ midx)
            expected_pt_eval += coeffs[coeff_idx] * idx_eval
        end
        all_out_correct_pt = (isapprox(out[pt_idx], expected_pt_eval, atol=1e-12))
        all_out_correct &= all_out_correct_pt
        if !all_out_correct_pt
            @test isapprox(out[pt_idx], expected_pt_eval, atol=1e-12)
        end
    end
    @test all_out_correct
    out2 = basisAssembly(fmset, coeffs, univariateEvals)
    @test isapprox(out, out2, atol=1e-12)
end

@testset "Polynomial Evaluation" begin
    out = zeros(length(fmset), N_pts)
    Evaluate!(out, fmset, univariateEvals)
    all_out_correct = true
    for pt_idx in 1:N_pts
        expected_pt_eval = 0.
        @inbounds for (poly_idx,midx) in enumerate(mset)
            idx_eval = prod(pts[pt_idx,:] .^ midx)
            all_out_correct &= isapprox(out[poly_idx, pt_idx],  idx_eval, atol=1e-12)
        end
        @test all_out_correct
    end
    out2 = Evaluate(fmset, univariateEvals)
    @test isapprox(out, out2, atol=1e-12)
end

@testset "Polynomial average" begin
    out = zeros(length(fmset))
    basesAverage!(out, fmset, univariateEvals)
    all_out_correct = true
    @inbounds for (poly_idx,midx) in enumerate(mset)
        expected_poly_avg = 0.
        for pt_idx in 1:N_pts
            idx_eval = prod(pts[pt_idx,:] .^ midx)
            expected_poly_avg += idx_eval
        end
        expected_poly_avg /= N_pts
        all_out_correct &= isapprox(out[poly_idx], expected_poly_avg, atol=1e-12)
    end
    @test all_out_correct
    out2 = basesAverage(fmset, univariateEvals)
    @test isapprox(out, out2, atol=1e-12)
end

@testset "Multivariate basis evaluations" begin
    bases = [MonicLegendrePolynomial(), PhysicistHermitePolynomial(), LaguerrePolynomial(),
            JacobiPolynomial(0.25, 0.45), MollifiedBasis(3, ProbabilistHermitePolynomial(), GaspariCohn(3.))]
    mv_basis = MultivariateBasis(bases...)
    univariate_eval_diffs = map(j->EvalDiff2(p, bases[j], pts[:,j]), 1:d)
    univariate_eval = first.(univariate_eval_diffs)
    univariate_diff = map(x->x[2], univariate_eval_diffs)
    univariate_diff2 = map(x->x[3], univariate_eval_diffs)

    out_eval = ntuple(_->zeros(p+1,N_pts), d)
    Evaluate!(out_eval, mv_basis, pts)
    @test all(out_eval[j] ≈ univariate_eval[j] for j in 1:d)
    out_eval_oop = Evaluate(ntuple(_->p,d), mv_basis, pts)
    @test all(out_eval[j] ≈ out_eval_oop[j] for j in 1:d)

    out_diff = ntuple(_->zeros(p+1,N_pts), d)
    EvalDiff!(out_eval, out_diff, mv_basis, pts)
    @test all(out_eval[j] ≈ univariate_eval[j] for j in 1:d)
    @test all(out_diff[j] ≈ univariate_diff[j] for j in 1:d)
    out_eval_oop, out_diff_oop = EvalDiff(ntuple(_->p,d), mv_basis, pts)
    @test all(out_eval[j] ≈ out_eval_oop[j] for j in 1:d)
    @test all(out_diff[j] ≈ out_diff_oop[j] for j in 1:d)

    out_diff2 = ntuple(_->zeros(p+1,N_pts), d)
    EvalDiff2!(out_eval, out_diff, out_diff2, mv_basis, pts)
    @test all(out_eval[j] ≈ univariate_eval[j] for j in 1:d)
    @test all(out_diff[j] ≈ univariate_diff[j] for j in 1:d)
    @test all(out_diff2[j] ≈ univariate_diff2[j] for j in 1:d)
    out_eval_oop, out_diff_oop, out_diff2_oop = EvalDiff2(ntuple(_->p,d), mv_basis, pts)
    @test all(out_eval[j] ≈ out_eval_oop[j] for j in 1:d)
    @test all(out_diff[j] ≈ out_diff_oop[j] for j in 1:d)
    @test all(out_diff2[j] ≈ out_diff2_oop[j] for j in 1:d)
end