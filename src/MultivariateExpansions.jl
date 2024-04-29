module MultivariateExpansions

using MultiIndexing
export polynomialAssembly!

# Assume all polynomials of degree zero are unity
# Assume we add to out
function polynomialAssembly!(out::AbstractVector, fmset::FixedMultiIndexSet{d}, coeffs::AbstractVector, univariateEvals::NTuple{d, AbstractMatrix}) where {d}
    # M = num points, N = num multi-indices, d = input dimension
    # out = (M,)
    # coeffs = (N,)
    # univariateEvals = (d,(p_j, M))
    N_midx = length(fmset.starts) - 1
    M_pts = length(out)
    @assert length(coeffs) == N_midx "Length of coeffs must match number of multi-indices"

    @inbounds for i in 1:d
        @assert size(univariateEvals[i], 1) >= fmset.max_orders[i] + 1 "Degree must match"
        @assert size(univariateEvals[i], 2) == length(out) "Number of points must match"
    end

    Threads.@threads for pt_idx in 1:M_pts
        @inbounds for midx in 1:N_midx
            start_midx = fmset.starts[midx]
            end_midx = fmset.starts[midx + 1] - 1

            termVal = 1.
            for j in start_midx:end_midx
                dim = fmset.nz_indices[j]
                power = fmset.nz_values[j]
                termVal *= univariateEvals[dim][power+1, pt_idx]
            end
            out[pt_idx] = muladd(coeffs[midx], termVal, out[pt_idx])
        end
    end
    nothing
end

abstract type AbstractDerivativeRecurrence end
struct ProbabilistHermiteDerivativeRecurrence <: AbstractDerivativeRecurrence end

"""
    score_loss(out, fmset, coeffs, univariateEvals, derivative_recurrence)

Estimates the loss when implicit score matching via

```math
\\mathbb{E}[\\|\\nabla f\\|^2 + \\Delta f]
```

where Δ is the Laplacian operator (i.e. sum of second derivatives). This assumes that
all univariate polynomials are part of an Appell sequence 
# Arguments
- `derivative_recurrence` is the derivative adjustment for the polynomial in dimension j at order k

"""
function score_loss(fmset::FixedMultiIndexSet{d},
    coeffs::AbstractVector, univariateEvals::NTuple{d, AbstractMatrix},
    ::ProbabilistHermiteDerivativeRecurrence) where {d}
    # M = num points, N = num multi-indices, d = input dimension
    # coeffs = (N,)
    # univariateEvals = (d,(p_j, M))

    N_midx = length(fmset.starts) - 1
    M_pts = length(out)

    @assert length(coeffs) == N_midx "Length of coeffs must match number of multi-indices"
    M = size(univariateEvals[1], 2)
    @inbounds for i in 1:d
        @assert size(univariateEvals[i], 1) >= fmset.max_orders[i] + 1 "Degree must match"
        @assert size(univariateEvals[i], 2) == M "Number of points must match"
    end

    # Allocate workspaces
    pointwise_loss = zeros(Threads.nthreads())
    pointwise_loss_counts = zeros(Int, Threads.nthreads())
    grad_workspace = zeros(d, Threads.nthreads())
    grad_term_workspace = zeros(d, Threads.nthreads())

    Threads.@threads for pt_idx in 1:M_pts
        # Retrieve workspaces
        grad = @view grad_workspace[:, Threads.threadid()]
        laplacian = 0.
        grad_term = @view grad_term_workspace[:, Threads.threadid()]
        laplace_term = @view grad_term_workspace[:, Threads.threadid()]

        # For each multi-index
        @inbounds for midx in 1:N_midx
            # Initialize workspace
            for dim in 1:d
                grad_term[dim] = 1.
                laplace_term[dim] = 1.
            end

            # Get bounds on nonzero values of midx
            start_midx = fmset.starts[midx]
            end_midx = fmset.starts[midx + 1] - 1

            # For each nonconstant dimension in this midx's term
            for j in start_midx:end_midx
                idx_dim = fmset.nz_indices[j]
                idx_power = fmset.nz_values[j]
                # Get the evaluation and first two derivatives
                midx_term = univariateEvals[idx_dim][idx_power+1, pt_idx]
                midx_diff = univariateEvals[idx_dim][idx_power, pt_idx]
                midx_diff2 = idx_power < 2 ? 0. : univariateEvals[idx_dim][idx_power-1, pt_idx]
                # Update the gradient and Laplacian for this term
                for dim in 1:d
                    if dim == idx_dim
                        grad_term[dim] *= midx_diff
                        laplace_term[dim] *= midx_diff2
                    else
                        grad_term[dim] *= midx_term
                        laplace_term[dim] *= midx_term
                    end
                end
            end
            # Update the gradient and Laplacian globally
            for dim in 1:d
                grad[dim] = muladd(coeffs[midx], grad_term[dim], grad[dim])
                laplacian = muladd(coeffs[midx], laplace_term[dim], laplacian)
            end
        end
        # Compute the pointwise loss and increment the count
        pointwise_loss_pt = 0.5(norm_grad' * norm_grad) + laplacian
        pointwise_loss[Threads.threadid()] += pointwise_loss_pt
        pointwise_loss_counts[Threads.threadid()] += 1
    end
    sum(pointwise_loss .* pointwise_loss_counts) / M
end

end
