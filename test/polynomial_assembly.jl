
@testset "FixedMultiIndexSet Polynomial Evaluation" begin
    d, p, N_pts = 5, 5, 1
    rng = Xoshiro(284028)
    mset = CreateTotalOrder(d, p)
    fmset = FixedMultiIndexSet(mset)
    coeffs = randn(rng, length(fmset))
    pts = randn(rng, d, N_pts)
    univariateEvals = ntuple(j->reduce(hcat, pts[j,:] .^ k for k in 0:p)', d)
    out = zeros(N_pts)
    polynomialAssembly!(out, fmset, coeffs, univariateEvals)
    for pt_idx in 1:N_pts
        expected_pt_eval = 0.
        @inbounds for (coeff_idx,midx) in enumerate(mset)
            idx_eval = prod(pts[:,pt_idx] .^ midx)
            expected_pt_eval += coeffs[coeff_idx] * idx_eval
        end
        @test out[pt_idx] ≈ expected_pt_eval atol=1e-12
    end
end