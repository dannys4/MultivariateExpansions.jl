export MultivariateBasis
export basisAssembly!, basisAssembly
export basesAverage!, basesAverage
using UnivariateApprox: UnivariateBasis

import UnivariateApprox: Evaluate!, EvalDiff!, EvalDiff2!, Evaluate, EvalDiff, EvalDiff2

"""
    MultivariateBasis(univariateBases...)

A type to store a multivariate basis as a tuple of univariate bases.
"""
struct MultivariateBasis{N, T}
    univariateBases::T
    function MultivariateBasis(univariateBases::Vararg{UnivariateBasis, N}) where {N}
        basisTuple = tuple(univariateBases...)
        new{N, typeof(basisTuple)}(basisTuple)
    end
end

"""
    Evaluate!(eval_space, basis, pts)

Evaluate all univariate bases associated with a multivariate basis at a set of points.

# Arguments
- `eval_space::NTuple{N, AbstractMatrix}`: The output space to store the evaluations, (N,(p_j+1, M))
- `basis::MultivariateBasis{N}`: The multivariate basis to evaluate
- `pts::AbstractMatrix`: The points to evaluate the basis at, (M,N)
"""
function Evaluate!(eval_space::NTuple{N, AbstractMatrix{U}},
        basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    for j in 1:N
        Evaluate!(eval_space[j], basis.univariateBases[j], @view(pts[:, j]))
    end
    nothing
end

"""
    EvalDiff!(eval_space, diff_space, basis, pts)

Evaluate all univariate bases and their derivatives associated with a multivariate basis at a set of points.

# Arguments
- `eval_space::NTuple{N, AbstractMatrix}`: The output space to store the evaluations, (N,(p_j+1, M))
- `diff_space::NTuple{N, AbstractMatrix}`: The output space to store the derivatives, (N,(p_j+1, M))
- `basis::MultivariateBasis{N}`: The multivariate basis to evaluate
- `pts::AbstractMatrix`: The points to evaluate the basis at, (M,N)
"""
function EvalDiff!(
        eval_space::NTuple{N, AbstractMatrix{U}}, diff_space::NTuple{N, AbstractMatrix{U}},
        basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    for j in 1:N
        EvalDiff!(eval_space[j], diff_space[j], basis.univariateBases[j], @view(pts[:, j]))
    end
    nothing
end

"""
    EvalDiff2!(eval_space, diff_space, diff2_space, basis, pts)

Evaluate all univariate bases and their first two derivatives associated with a multivariate basis at a set of points.

# Arguments
- `eval_space::NTuple{N, AbstractMatrix}`: The output space to store the evaluations, (N,(p_j+1, M))
- `diff_space::NTuple{N, AbstractMatrix}`: The output space to store the first derivatives, (N,(p_j+1, M))
- `diff2_space::NTuple{N, AbstractMatrix}`: The output space to store the second derivatives, (N,(p_j+1, M))
- `basis::MultivariateBasis{N}`: The multivariate basis to evaluate
"""
function EvalDiff2!(
        eval_space::NTuple{N, AbstractMatrix{U}}, diff_space::NTuple{N, AbstractMatrix{U}},
        diff2_space::NTuple{N, AbstractMatrix{U}},
        basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    for j in 1:N
        EvalDiff2!(eval_space[j], diff_space[j], diff2_space[j],
            basis.univariateBases[j], @view(pts[:, j]))
    end
    nothing
end

"""
    Evaluate(p, basis, pts)

Evaluate all univariate bases associated with a multivariate basis at a set of points.

# Arguments
- `p::NTuple{N,Int}`: The maximum degree of each univariate basis (N)
- `basis::MultivariateBasis{N}`: The multivariate basis to evaluate
- `pts::AbstractMatrix`: The points to evaluate the basis at, (M,N)

See also [`Evaluate!`](@ref).
"""
function Evaluate(p::NTuple{N,Int}, basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    eval_space = ntuple(j -> Matrix{U}(undef, p[j] + 1, size(pts, 1)), N)
    Evaluate!(eval_space, basis, pts)
    eval_space
end

"""
    EvalDiff(p, basis, pts)

Evaluate all univariate bases and their derivatives associated with a multivariate basis at a set of points.

# Arguments
- `p::NTuple{N,Int}`: The maximum degree of each univariate basis (N)
- `basis::MultivariateBasis{N}`: The multivariate basis to evaluate
- `pts::AbstractMatrix`: The points to evaluate the basis at, (M,N)

See also [`EvalDiff!`](@ref).
"""
function EvalDiff(p::NTuple{N,Int}, basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    eval_space = ntuple(j -> Matrix{U}(undef, p[j] + 1, size(pts, 1)), N)
    diff_space = ntuple(j -> Matrix{U}(undef, p[j] + 1, size(pts, 1)), N)
    EvalDiff!(eval_space, diff_space, basis, pts)
    eval_space, diff_space
end

"""
    EvalDiff2(p, basis, pts)

Evaluate all univariate bases and their first two derivatives associated with a multivariate basis at a set of points.

# Arguments
- `p::NTuple{N,Int}`: The maximum degree of each univariate basis (N)
- `basis::MultivariateBasis{N}`: The multivariate basis to evaluate
- `pts::AbstractMatrix`: The points to evaluate the basis at, (M,N)

See also [`EvalDiff2!`](@ref).
"""
function EvalDiff2(p::NTuple{N,Int}, basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    eval_space = ntuple(j -> Matrix{U}(undef, p[j] + 1, size(pts, 1)), N)
    diff_space = ntuple(j -> Matrix{U}(undef, p[j] + 1, size(pts, 1)), N)
    diff2_space = ntuple(j -> Matrix{U}(undef, p[j] + 1, size(pts, 1)), N)
    EvalDiff2!(eval_space, diff_space, diff2_space, basis, pts)
    eval_space, diff_space, diff2_space
end

"""
    basisAssembly!(out, fmset, coeffs, univariateEvals)

Evaluate a multivariate expansion on a set of points given the coefficients and univariate bases evaluated at each point.

# Arguments
- `out::AbstractVector`: The output vector to store the expansion evaluations, (M,)
- `fmset::FixedMultiIndexSet{d}`: The fixed multi-index set defining the expansion space (N,d)
- `coeffs::AbstractVector`: The coefficients of the multivariate expansion (N,)
- `univariateEvals::NTuple{d, AbstractMatrix}`: The univariate evaluations at each marginal point (d,(p_j, M)), where p_j is the maximum degree of the j-th univariate basis
"""
function basisAssembly!(out::AbstractVector, fmset::FixedMultiIndexSet{d},
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
    Evaluate!(out, fmset, univariateEvals)

Evaluate the basis of a multivariate expansion given the evaluations of the univariate bases

# Arguments
- `out::AbstractMatrix`: The output matrix to store the expansion evaluations, (N,M)
- `fmset::FixedMultiIndexSet{d}`: The fixed multi-index set defining the expansion space (N,d)
- `univariateEvals::NTuple{d, AbstractMatrix}`: The univariate evaluations at each marginal point (d,(p_j, M)), where p_j is the maximum degree of the j-th univariate basis

See also [`Evaluate`](@ref).
"""
function Evaluate!(out::AbstractMatrix, fmset::FixedMultiIndexSet{d},
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
    basesAverage!(out, fmset, univariateEvals)

Evaluate the average of a set of basis functions over a set of points given the univariate basis evaluations

# Arguments
- `out::AbstractVector`: The output vector to store the basis averages, (N,)
- `fmset::FixedMultiIndexSet{d}`: The fixed multi-index set defining the function space (N,d)
- `univariateEvals::NTuple{d, AbstractMatrix}`: The univariate evaluations at each marginal point (d,(p_j, M)), where p_j is the maximum degree of the j-th univariate basis

See also [`basesAverage`](@ref).
"""
function basesAverage!(out::AbstractVector, fmset::FixedMultiIndexSet{d},
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
        out[midx] = sum(tmp) / M_pts
    end
    nothing
end

"""
    basisAssembly(fmset, coeffs, univariateEvals)

Out-of-place assembly of multivariate basis

See [`basisAssembly!`](@ref) for details.
"""
function basisAssembly(
        fmset::FixedMultiIndexSet{d}, coeffs::AbstractVector, univariateEvals) where {d}
    out = zeros(eltype(univariateEvals[1]), size(univariateEvals[1], 2))
    basisAssembly!(out, fmset, coeffs, univariateEvals)
    out
end

"""
    Evaluate(fmset, univariateEvals)

Out-of-place evaluation of multivariate expansion

See [`Evaluate!`](@ref) for details.
"""
function Evaluate(fmset::FixedMultiIndexSet{d}, univariateEvals) where {d}
    out = zeros(eltype(univariateEvals[1]), length(fmset), size(univariateEvals[1], 2))
    Evaluate!(out, fmset, univariateEvals)
    out
end

"""
    basesAverage(fmset, univariateEvals)

Out-of-place evaluation of the average of a set of basis functions

See [`basesAverage!`](@ref) for details.
"""
function basesAverage(fmset::FixedMultiIndexSet{d}, univariateEvals) where {d}
    out = zeros(length(fmset))
    basesAverage!(out, fmset, univariateEvals)
    out
end