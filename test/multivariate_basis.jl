d, p, N_pts = 5, 5, 1000
rng = Xoshiro(284028)
mset = CreateTotalOrder(d, p)
fmset = FixedMultiIndexSet(mset)
pts = randn(rng, d, N_pts)
univariateEvals = ntuple(j->reduce(hcat, pts[j,:] .^ k for k in 0:p)', d)
@testset "Polynomial Assembly" begin
    coeffs = randn(rng, length(fmset))
    out = zeros(N_pts)
    basisAssembly!(out, fmset, coeffs, univariateEvals)
    all_out_correct = true
    for pt_idx in 1:N_pts
        expected_pt_eval = 0.
        @inbounds for (coeff_idx,midx) in enumerate(mset)
            idx_eval = prod(pts[:,pt_idx] .^ midx)
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
            idx_eval = prod(pts[:,pt_idx] .^ midx)
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
        for pt_idx in axes(pts,2)
            idx_eval = prod(pts[:,pt_idx] .^ midx)
            expected_poly_avg += idx_eval
        end
        expected_poly_avg /= N_pts
        all_out_correct &= isapprox(out[poly_idx], expected_poly_avg, atol=1e-12)
    end
    @test all_out_correct
    out2 = basesAverage(fmset, univariateEvals)
    @test isapprox(out, out2, atol=1e-12)
end