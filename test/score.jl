
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

@testset "Check time-based score matching" begin
    dim = 3
    mset_mat = [zeros(Int, dim) I(dim) 2I(dim);zeros(Int, 1, 2dim+1)]
    mset = MultiIndexSet(mset_mat)
    fmset = FixedMultiIndexSet(mset)
    M_pts = 2
    rng = Xoshiro(284028)
    data = 0.25ones(M_pts, dim) #randn(rng, M_pts, dim)
    time_pts = [0.]
    time_wts = [1.]
    basis = MultivariateBasis((ProbabilistHermitePolynomial() for _ in 1:dim)..., LaguerrePolynomial())
    integrator = MultivariateExpansions.EulerMaruyamaIntegrator(M_pts, dim)
    A, b, constant_term, linear_space_index, quad_space_index = MultivariateExpansions.score_optimal_coeff_time(fmset, basis, data, integrator, time_pts, time_wts)
    sort!(linear_space_index)
    sort!(quad_space_index)
    @test constant_term == 1
    @test linear_space_index == collect(2:dim+1)
    @test quad_space_index == collect(dim+2:2dim+1)
    # coeffs = A \ b
    # @info "" coeffs
end