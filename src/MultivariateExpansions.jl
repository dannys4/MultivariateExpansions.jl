module MultivariateExpansions

using MultiIndexing, LinearAlgebra
export polynomialAssembly!, polynomialAssembly
export polynomialsEval!, polynomialsEval
export polynomialsAverage!, polynomialsAverage
export score_loss, score_optimal_coeff

"""
    polynomialAssembly!(out, fmset, coeffs, univariateEvals)
Evaluate a polynomial expansion at a set of points given the coefficients and univariate evaluations at each marginal point.

# Arguments
- `out::AbstractVector`: The output vector to store the polynomial evaluations, (M,)
- `fmset::FixedMultiIndexSet{d}`: The fixed multi-index set defining the polynomial space (N,d)
- `coeffs::AbstractVector`: The coefficients of the polynomial expansion (N,)
- `univariateEvals::NTuple{d, AbstractMatrix}`: The univariate evaluations at each marginal point (d,(p_j, M)), where p_j is the maximum degree of the j-th univariate polynomial
"""
function polynomialAssembly!(out::AbstractVector, fmset::FixedMultiIndexSet{d},
        coeffs::AbstractVector, univariateEvals::NTuple{d, T}) where {
        d, T <: AbstractMatrix}
    # M = num points, N = num multi-indices, d = input dimension
    # out = (M,)
    # coeffs = (N,)
    # univariateEvals = (d,(p_j, M))
    N_midx = length(fmset.starts) - 1
    M_pts = length(out)
    @assert length(coeffs)==N_midx "Length of coeffs must match number of multi-indices"

    @inbounds for i in 1:d
        @assert size(univariateEvals[i], 1)>=fmset.max_orders[i] + 1 "Degree must match"
        @assert size(univariateEvals[i], 2)==length(out) "Number of points must match"
    end

    for pt_idx in 1:M_pts
        @inbounds for midx in 1:N_midx
            start_midx = fmset.starts[midx]
            end_midx = fmset.starts[midx + 1] - 1

            termVal = 1.0
            for j in start_midx:end_midx
                dim = fmset.nz_indices[j]
                power = fmset.nz_values[j]
                termVal *= univariateEvals[dim][power + 1, pt_idx]
            end
            out[pt_idx] = muladd(coeffs[midx], termVal, out[pt_idx])
        end
    end
    nothing
end

"""
    polynomialsEval!(out, fmset, univariateEvals)
Evaluate a set of multivariate polynomials given the evaluations of the univariate polynomials

# Arguments
- `out::AbstractMatrix`: The output matrix to store the polynomial evaluations, (N,M)
- `fmset::FixedMultiIndexSet{d}`: The fixed multi-index set defining the polynomial space (N,d)
- `univariateEvals::NTuple{d, AbstractMatrix}`: The univariate evaluations at each marginal point (d,(p_j, M)), where p_j is the maximum degree of the j-th univariate polynomial

See also [`polynomialsEval`](@ref).
"""
function polynomialsEval!(out::AbstractMatrix, fmset::FixedMultiIndexSet{d},
        univariateEvals::NTuple{d, T}) where {d, T <: AbstractMatrix}
    # M = num points, N = num multi-indices, d = input dimension
    # out = (N,M)
    # univariateEvals = (d,(p_j, M))
    N_midx = length(fmset)
    M_pts = size(out, 2)
    @assert size(out, 1)==N_midx "Row count of out must match number of multi-indices"

    @inbounds for i in 1:d
        @assert size(univariateEvals[i], 1)>=fmset.max_orders[i] + 1 "Degree must match"
        @assert size(univariateEvals[i], 2)==M_pts "Number of points must match"
    end

    @inbounds Threads.@threads for pt_idx in 1:M_pts
        for midx in 1:N_midx
            start_midx = fmset.starts[midx]
            end_midx = fmset.starts[midx + 1] - 1

            termVal = 1.0
            for j in start_midx:end_midx
                dim = fmset.nz_indices[j]
                power = fmset.nz_values[j]
                termVal *= univariateEvals[dim][power + 1, pt_idx]
            end
            out[midx, pt_idx] = termVal
        end
    end
    nothing
end

"""
    polynomialsAverage!(out, fmset, univariateEvals)
Evaluate the average of a set of polynomials over a set of points given the univariate polynomial evaluations

# Arguments
- `out::AbstractVector`: The output vector to store the polynomial evaluations, (N,)
- `fmset::FixedMultiIndexSet{d}`: The fixed multi-index set defining the polynomial space (N,d)
- `univariateEvals::NTuple{d, AbstractMatrix}`: The univariate evaluations at each marginal point (d,(p_j, M)), where p_j is the maximum degree of the j-th univariate polynomial

See also [`polynomialsAverage`](@ref).
"""
function polynomialsAverage!(out::AbstractVector, fmset::FixedMultiIndexSet{d},
        univariateEvals::NTuple{d, T}) where {d, T <: AbstractMatrix}
    N_midx = length(fmset)
    M_pts = size(univariateEvals[1], 2)
    @assert length(out)==N_midx "Length of out must match number of multi-indices"

    @inbounds for i in 1:d
        @assert size(univariateEvals[i], 1)>=fmset.max_orders[i] + 1 "Degree must match"
        @assert size(univariateEvals[i], 2)==M_pts "Number of points must match"
    end
    tmp_all = zeros(eltype(univariateEvals[1]), M_pts, Threads.nthreads())
    @inbounds Threads.@threads for midx in 1:N_midx
        tmp = view(tmp_all, :, Threads.threadid())
        start_midx = fmset.starts[midx]
        end_midx = fmset.starts[midx + 1] - 1

        tmp .= 1.0
        for j in start_midx:end_midx
            dim = fmset.nz_indices[j]
            power = fmset.nz_values[j]
            tmp .*= univariateEvals[dim][power + 1, :]
        end
        out[midx] = sum(tmp)/M_pts
    end
    nothing
end

"""
    polynomialAssembly(fmset, coeffs, univariateEvals)
Out-of-place assembly of multivariate polynomial

See [`polynomialAssembly!`](@ref) for details.
"""
function polynomialAssembly(
        fmset::FixedMultiIndexSet{d}, coeffs::AbstractVector, univariateEvals) where {d}
    out = zeros(eltype(univariateEvals[1]), size(univariateEvals[1], 2))
    polynomialAssembly!(out, fmset, coeffs, univariateEvals)
    out
end

"""
    polynomialsEval(fmset, univariateEvals)
Out-of-place evaluation of multivariate polynomials

See [`polynomialsEval!`](@ref) for details.
"""
function polynomialsEval(fmset::FixedMultiIndexSet{d}, univariateEvals) where {d}
    out = zeros(eltype(univariateEvals[1]), length(fmset), size(univariateEvals[1], 2))
    polynomialsEval!(out, fmset, univariateEvals)
    out
end

"""
    polynomialsAverage(fmset, univariateEvals)
Out-of-place evaluation of the average of a set of polynomials

See [`polynomialsAverage!`](@ref) for details.
"""
function polynomialsAverage(fmset::FixedMultiIndexSet{d}, univariateEvals) where {d}
    out = zeros(length(fmset))
    polynomialsAverage!(out, fmset, univariateEvals)
    out
end

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

function score_optimal_coeff(
        fmset::FixedMultiIndexSet{d}, univariateEvals::NT, univariateDiff::NT,
        univariateDiff2::NT) where {d, NT <: NTuple{d, <:AbstractMatrix}}
    M_pts = size(univariateEvals[1], 2)
    N_midx = length(fmset)
    G = zeros(N_midx, N_midx)
    v = zeros(N_midx)
    Gj = zeros(d, N_midx)
    constant_term = 0
    for midx in 1:N_midx
        start_midx = fmset.starts[midx]
        end_midx = fmset.starts[midx + 1] - 1
        if start_midx > end_midx
            constant_term = midx
            break
        end
    end
    @inbounds for pt_idx in 1:M_pts
        Gj .= 0
        for grad_dim in 1:d
            for midx in 1:N_midx
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
                        firstDiff *= univariateDiff[idx_dim][idx_power, pt_idx]
                        secondDiff *= idx_power < 2 ? 0.0 :
                                      univariateDiff2[idx_dim][idx_power - 1, pt_idx]
                    else
                        eval_term = univariateEvals[idx_dim][idx_power + 1, pt_idx]
                        firstDiff *= eval_term
                        secondDiff *= eval_term
                    end
                end
                !contains_grad_dim && continue
                v[midx] += secondDiff
                Gj[grad_dim, midx] = firstDiff
            end
        end
        # G += Gj' * Gj, but only the upper triangular part
        LinearAlgebra.BLAS.syrk!('U', 'T', true, Gj, true, G)
    end
    H = Hermitian(G)
    H /= M_pts
    v /= -M_pts
    H[constant_term, constant_term] = 1.0
    H, v
end

end