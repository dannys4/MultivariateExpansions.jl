export score_loss, score_optimal_coeff

"""
    score_loss(fmset, coeffs, univariateEval, univariateDiff, univariateDiff2)
"""
function score_loss(fmset::FixedMultiIndexSet{d}, coeffs::AbstractVector,
        univariateEval::NT, univariateDiff::NT,
        univariateDiff2::NT) where {d, NT <: NTuple{d, AbstractMatrix}}
    N_midx = length(fmset.starts) - 1
    M_pts = size(univariateEval[1], 2)
    @assert length(coeffs)==N_midx "Length of coeffs must match number of multi-indices"
    @inbounds for i in 1:d
        @assert size(univariateEval[i], 1)>=fmset.max_orders[i] + 1 "Degree must match"
        @assert size(univariateEval[i], 2)==M_pts "Number of points must match"
        @assert size(univariateDiff[i], 1)==fmset.max_orders[i] "Degree must match"
        @assert size(univariateDiff[i], 2)==M_pts "Number of points must match"
        @assert size(univariateDiff2[i], 1)==fmset.max_orders[i] - 1 "Degree must match"
        @assert size(univariateDiff2[i], 2)==M_pts "Number of points must match"
    end

    # Allocate workspaces
    grad_square = 0.0
    laplacian = 0.0
    @inbounds for pt_idx in 1:M_pts
        # Take partial derivative in each dim
        for grad_dim in 1:d
            grad = 0.0

            # For each multi-index
            for midx in 1:N_midx
                # Get bounds on nonzero values of midx
                start_midx = fmset.starts[midx]
                end_midx = fmset.starts[midx + 1] - 1
                start_midx > end_midx && continue
                contains_dim = false
                first_diff = 1.0
                second_diff = 1.0
                # For each nonconstant dimension in this midx's term
                for j in start_midx:end_midx
                    idx_dim = fmset.nz_indices[j]
                    idx_power = fmset.nz_values[j]
                    contains_dim |= idx_dim == grad_dim
                    if idx_dim == grad_dim
                        first_diff *= univariateDiff[idx_dim][idx_power, pt_idx]
                        second_diff *= idx_power < 2 ? 0.0 :
                                       univariateDiff2[idx_dim][idx_power - 1, pt_idx]
                    else
                        eval_term = univariateEval[idx_dim][idx_power + 1, pt_idx]
                        first_diff *= eval_term
                        second_diff *= eval_term
                    end
                end
                contains_dim || continue
                grad = muladd(coeffs[midx], first_diff, grad)
                laplacian = muladd(coeffs[midx], second_diff, laplacian)
            end

            grad_square += grad^2
        end
    end
    score_est = (grad_square / 2 + laplacian) / M_pts
    score_est
end

function find_constant_term(fmset::FixedMultiIndexSet)::Int
    for midx in 1:length(fmset)
        start_midx = fmset.starts[midx]
        end_midx = fmset.starts[midx + 1] - 1
        if start_midx > end_midx
            return midx
        end
    end
    0
end

function calculate_upper_tri_gradient_point!(
    gradient_coeff_grad::AbstractMatrix,
    laplace_coeff_grad::AbstractVector,
    fmset::FixedMultiIndexSet{d},
    univariateEvals::NT,
    univariateDiff::NT,
    univariateDiff2::NT,
    pt_idx::Integer) where {d, NT <: NTuple{d, <:AbstractMatrix}}

    for grad_dim in axes(gradient_coeff_grad, 1)
        for midx in 1:length(fmset)
            firstDiff = 1.0
            secondDiff = 1.0
            start_midx = fmset.starts[midx]
            end_midx = fmset.starts[midx + 1] - 1
            contains_grad_dim = false
            for j in start_midx:end_midx
                idx_dim = fmset.nz_indices[j]
                idx_power = fmset.nz_values[j]
                contains_grad_dim |= idx_dim == grad_dim
                if idx_dim == grad_dim
                    firstDiff *= univariateDiff[idx_dim][idx_power + 1, pt_idx]
                    secondDiff *= idx_power < 2 ? 0.0 :
                                  univariateDiff2[idx_dim][idx_power + 1, pt_idx]
                else
                    eval_term = univariateEvals[idx_dim][idx_power + 1, pt_idx]
                    @info "" eval_term
                    firstDiff *= eval_term
                    secondDiff *= eval_term
                end
            end
            !contains_grad_dim && continue
            laplace_coeff_grad[midx] += secondDiff
            gradient_coeff_grad[grad_dim, midx] = firstDiff
        end
    end
end

"""
    score_optimal_coeff(fmset, univariateEvals, univariateDiff, univariateDiff2)
"""
function score_optimal_coeff(
        fmset::FixedMultiIndexSet{d}, univariateEvals::NT, univariateDiff::NT,
        univariateDiff2::NT) where {d, NT <: NTuple{d, <:AbstractMatrix}}
    M_pts = size(univariateEvals[1], 2)
    N_midx = length(fmset)
    gradient_coeff_grad = zeros(N_midx, N_midx)
    laplacian_coeff_grad = zeros(N_midx)
    gradient_coeff_grad_pt = zeros(d, N_midx)
    constant_term = find_constant_term(fmset)
    for pt_idx in 1:M_pts
        calculate_upper_tri_gradient_point!(gradient_coeff_grad_pt, laplacian_coeff_grad, fmset, univariateEvals, univariateDiff, univariateDiff2, pt_idx)
        # G += Gj' * Gj, but only the upper triangular part
        LinearAlgebra.BLAS.syrk!('U', 'T', true, gradient_coeff_grad_pt, true, gradient_coeff_grad)
    end
    gradient_coeff_grad_herm = Hermitian(gradient_coeff_grad)
    gradient_coeff_grad_herm /= M_pts
    laplacian_coeff_grad /= -M_pts
    gradient_coeff_grad_herm[constant_term, constant_term] = 1.0
    gradient_coeff_grad_herm, laplacian_coeff_grad
end

function score_optimal_coeff(fmset::FixedMultiIndexSet{d}, basis::MultivariateBasis{d}, pts::AbstractMatrix) where {d}
    evals, diffs, diff2s = EvalDiff2(tuple(fmset.max_orders), basis, pts)
    score_optimal_coeff(fmset, evals, diffs, diff2s)
end

function time_dim_constant_indices(fmset::FixedMultiIndexSet{d}) where {d}
    constant_index = 0
    linear_space_index = zeros(Int, d-1)
    quad_space_index = zeros(Int, d-1)
    indices = (linear_space_index, quad_space_index)
    for midx in 1:length(fmset)
        start_midx = fmset.starts[midx]
        end_midx = fmset.starts[midx + 1] - 1
        if start_midx > end_midx
            constant_index = midx
            continue
        end
        # If this basis function is constant in time
        if fmset.nz_indices[end_midx] != d
            nz_idx = fmset.nz_indices[end_midx]
            nz_val = fmset.nz_values[end_midx]
            # Check that it is below degree 3 in space without mixed terms
            if end_midx != start_midx || nz_val > 2
                @info "" start_midx end_midx nz_val
                throw(ArgumentError("Invalid fixed multi-index for time dimension constant term"))
            end
            # Enter the coeff index into the linear or quad space list depending on nz_val
            indices[nz_val][nz_idx] = midx
        end
    end
    constant_index, linear_space_index, quad_space_index
end

struct EulerMaruyamaIntegrator
    rng::AbstractRNG
    workspace::Matrix{Float64}
    scale::Float64
    function EulerMaruyamaIntegrator(M_pts::Int, dim::Int, scale::Float64=1., rng::AbstractRNG = Random.GLOBAL_RNG)
        new(rng, Matrix{Float64}(undef, M_pts, dim), scale)
    end
end

function (em::EulerMaruyamaIntegrator)(data::AbstractMatrix{U}, dt::Float64) where {U}
    randn!(em.rng, em.workspace)
    sqrt_dt = sqrt(dt)
    @inbounds @simd for j in eachindex(data)
        data[j] = muladd(sqrt_dt, em.workspace[j], em.scale*data[j])
    end
end

function score_optimal_coeff_time(
    fmset::FixedMultiIndexSet{d},
    basis::MultivariateBasis{d},
    data::AbstractMatrix{U},
    stochastic_integrator!,
    time_pts::AbstractVector,
    time_wts::AbstractVector) where {d,U}

    dim = d-1
    M_pts = size(data, 1)
    size(data, 2) == dim || throw(ArgumentError("Expected data to have $dim cols, but got $(size(data, 2))"))
    new_data = Matrix{U}(undef, M_pts, d)
    new_data[:,1:end-1] .= data
    data = new_data # Alias

    N_midx = length(fmset)
    gradient_coeff_grad_t = zeros(N_midx, N_midx)
    laplacian_coeff_grad_t = zeros(N_midx)

    gradient_coeff_grad = zero(gradient_coeff_grad_t)
    laplacian_coeff_grad = zero(laplacian_coeff_grad_t)

    # Only want spatial gradient, not time
    gradient_coeff_grad_pt_t = zeros(dim, N_midx)

    constant_term, linear_space_index, quad_space_index = time_dim_constant_indices(fmset)

    prev_time = 0.
    eval_space = ntuple(j -> Matrix{U}(undef, fmset.max_orders[j] + 1, M_pts), d)
    diff_space = ntuple(j -> Matrix{U}(undef, fmset.max_orders[j] + 1, M_pts), d)
    diff2_space = ntuple(j -> Matrix{U}(undef, fmset.max_orders[j] + 1, M_pts), d)

    for t_idx = eachindex(time_pts)
        t_wt = time_wts[t_idx]
        data[:,end] .= time_pts[t_idx]
        stochastic_integrator!(@view(data[:,1:end-1]), time_pts[t_idx] - prev_time)
        EvalDiff2!(eval_space, diff_space, diff2_space, basis, data)
        for pt_idx in 1:M_pts
            calculate_upper_tri_gradient_point!(gradient_coeff_grad_pt_t, laplacian_coeff_grad_t, fmset, eval_space, diff_space, diff2_space, pt_idx)
            # G += Gj' * Gj, but only the upper triangular part
            LinearAlgebra.BLAS.syrk!('U', 'T', true, gradient_coeff_grad_pt_t, true, gradient_coeff_grad_t)
            @info "" gradient_coeff_grad_pt_t'gradient_coeff_grad_pt_t
        end
        @. gradient_coeff_grad  += (t_wt/M_pts)*gradient_coeff_grad_t
        @. laplacian_coeff_grad -= (t_wt/M_pts)*laplacian_coeff_grad_t
        gradient_coeff_grad_t .= 0.0
        laplacian_coeff_grad_t .= 0.0
        prev_time = time_pts[t_idx]
    end

    gradient_coeff_grad_herm = Hermitian(gradient_coeff_grad)
    gradient_coeff_grad_herm[constant_term, constant_term] = 1.0
    gradient_coeff_grad_herm, laplacian_coeff_grad, constant_term, linear_space_index, quad_space_index
end