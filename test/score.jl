
@testset "Score loss assembly" begin
    d, p, N_pts = 5, 5, 1000
    rng = Xoshiro(284028)
    mset = CreateTotalOrder(d, p)
    fmset = FixedMultiIndexSet(mset)
    coeffs = randn(rng, length(fmset))
    pts = randn(rng, d, N_pts)
    fd_delta = 1e-6
    grad_fd = zeros(d, N_pts)
    laplacian = zeros(N_pts)
    pts_fd = copy(pts)
    univariateEvals = ntuple(j -> reduce(hcat, pts[j, :] .^ k for k in 0:p)', d)
    out = zeros(N_pts)
    basisAssembly!(out, fmset, coeffs, univariateEvals)

    for i in 1:d
        out_plus_fd = zeros(N_pts)
        out_minus_fd = zeros(N_pts)
        pts_fd[i, :] .+= fd_delta
        univariateEvals_fd = ntuple(j -> reduce(hcat, pts_fd[j, :] .^ k for k in 0:p)', d)
        basisAssembly!(out_plus_fd, fmset, coeffs, univariateEvals_fd)
        pts_fd[i, :] .-= 2fd_delta
        univariateEvals_fd = ntuple(j -> reduce(hcat, pts_fd[j, :] .^ k for k in 0:p)', d)
        basisAssembly!(out_minus_fd, fmset, coeffs, univariateEvals_fd)
        pts_fd[i, :] .+= fd_delta
        grad_term = (out_plus_fd - out_minus_fd) / (2fd_delta)
        grad_fd[i, :] = grad_term
        laplacian_term = (out_plus_fd - 2out + out_minus_fd) / (fd_delta^2)
        laplacian += laplacian_term
    end
    grad_fd_sq_mean = mean(sum(grad_fd .^ 2, dims = 1)) / 2
    laplacian_mean = mean(laplacian)
    fd_score = grad_fd_sq_mean + laplacian_mean
    univariateDiff = ntuple(j -> reduce(hcat, k * pts[j, :] .^ (k - 1) for k in 1:p)', d)
    univariateDiff2 = ntuple(
        j -> reduce(hcat, k * (k - 1) * pts[j, :] .^ (k - 2) for k in 2:p)', d)
    score_est = score_loss(fmset, coeffs, univariateEvals, univariateDiff, univariateDiff2)
    @test isapprox(score_est, fd_score, rtol = 20fd_delta)
end

@testset "Check optimal coeffs" begin
    d, p, N_pts = 3, 3, 10000
    rng = Xoshiro(284028)
    mset = CreateTotalOrder(d, p)
    quad_idxs = [j for (j, m) in enumerate(mset) if sum(m .> 0) == 1 && sum(m) == 2]
    fmset = FixedMultiIndexSet(mset)
    pts = randn(rng, d, N_pts)
    univariateEvals = ntuple(j -> reduce(hcat, pts[j, :] .^ k for k in 0:p)', d)
    univariateDiff = ntuple(j -> reduce(hcat, k * pts[j, :] .^ (k - 1) for k in 1:p)', d)
    univariateDiff2 = ntuple(
        j -> reduce(hcat, k * (k - 1) * pts[j, :] .^ (k - 2) for k in 2:p)', d)
    G, v = MultivariateExpansions.score_optimal_coeff(
        fmset, univariateEvals, univariateDiff, univariateDiff2)
    coeffs = G \ v
    for (j, m) in enumerate(mset)
        is_quad_idx = sum(m .> 0) == 1 && sum(m) == 2
        if is_quad_idx
            @test isapprox(coeffs[j], -0.5, rtol = 10sqrt(N_pts))
        else
            @test isapprox(coeffs[j], 0.0, rtol = 10sqrt(N_pts))
        end
    end
end