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

function score_optimal_coeff(
        fmset::FixedMultiIndexSet{d}, univariateEvals::NT, univariateDiff::NT,
        univariateDiff2::NT) where {d, U, NT <: NTuple{d, <:AbstractMatrix{U}}}
    M_pts = size(univariateEvals[1], 2)
    N_midx = length(fmset)
    backend = get_backend(univariateEvals[1])
    G = zeros(backend, U, (N_midx, N_midx))
    v = zeros(backend, U, (N_midx,))
    Gj = zeros(backend, U, (d, N_midx))
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


function score_optimal_coeff_nohess(fmset::FixedMultiIndexSet{d}, univariateEvals::NT,
        univariateDiff::NT) where {d, U, NT <: NTuple{d, <:AbstractMatrix{U}}}
    M_pts = size(univariateEvals[1], 2)
    N_midx = length(fmset)
    backend = get_backend(univariateEvals[1])
    (;starts, nz_indices, nz_values) = fmset
    reduce_op = (x::Tuple{U, SVector{d, U}},y::Tuple{U, SVector{d, U}}) -> (x[1]+y[1], x[2]+y[2])
    init = (zero(U), zero(SVector{d, U}))
    G = zeros(backend, U, (N_midx, N_midx))
    h = zeros(backend, SVector{d, U}, (N_midx,))
    Threads.@threads for idx in CartesianIndices(G)
        i, j = Tuple(idx)
        i > j && continue
        G_ij, h_ij = AK.mapreduce(reduce_op, 1:M_pts, backend; init, neutral=init) do pt_idx
            start_midx_i, end_midx_i = starts[i], starts[i + 1] - 1
            start_midx_j, end_midx_j = starts[j], starts[j + 1] - 1
            G_ij = one(U)
            for k in start_midx_i:end_midx_i
                dim = nz_indices[k]
                power = nz_values[k]
                G_ij *= univariateEvals[dim][power + 1, pt_idx]
            end
            for k in start_midx_j:end_midx_j
                dim = nz_indices[k]
                power = nz_values[k]
                G_ij *= univariateEvals[dim][power + 1, pt_idx]
            end
            h_ij = zero(SVector{d, U})
            if i == j
                grad = zero(MVector{d, U})
                for k_prime in 1:d
                    ret = one(U)
                    has_grad = false
                    for k in start_midx_i:end_midx_i
                        dim = nz_indices[k]
                        power = nz_values[k]
                        if dim == k_prime
                            has_grad = true
                            ret *= univariateDiff[dim][power, pt_idx]
                        else
                            ret *= univariateEvals[dim][power + 1, pt_idx]
                        end
                    end
                    has_grad && (grad[k_prime] = ret)
                end
                h_ij = SVector(grad)
            end
            G_ij, h_ij
        end
        G[i, j] = G[j, i] = G_ij / M_pts
        if i == j
            h[i] = h_ij / M_pts
        end
    end
    h = reduce(hcat, h)'
    G, h
end
