export MultivariateBasis
export basisAssembly!, basisAssembly
export basesAverage!, basesAverage
using UnivariateApprox: UnivariateBasis

using Base: length

import UnivariateApprox: Evaluate!, EvalDiff!, EvalDiff2!, Evaluate, EvalDiff, EvalDiff2

abstract type AbstractMultivariateBasis{d} end

"""
    MultivariateBasis(univariateBases...)

A type to store a multivariate basis as a tuple of univariate bases.
"""
struct MultivariateBasis{d, T} <: AbstractMultivariateBasis{d}
    univariateBases::T
end

function MultivariateBasis(univariateBases::Vararg{UnivariateBasis, _d}) where {_d}
    basisTuple = tuple(univariateBases...)
    @argcheck _d isa Integer
    MultivariateBasis{_d, typeof(basisTuple)}(basisTuple)
end

"""
    MultivariateBasis(univariateBasis, N)

Create a multivariate basis by repeating a univariate basis N times.
"""
function MultivariateBasis(univariateBasis::UnivariateBasis, N::Int)
    basis = ntuple(_ -> univariateBasis, N)
    MultivariateBasis{N,typeof(basis)}(basis)
end

function Base.length(::MultivariateBasis{d}) where d
    d
end

"""
    Evaluate!(eval_space, basis, pts)

Evaluate all univariate bases associated with a multivariate basis at a set of points.

# Arguments
- `eval_space::NTuple{N, AbstractMatrix}`: The output space to store the evaluations, (N,(p_j+1, M))
- `basis::MultivariateBasis{N}`: The multivariate basis to evaluate
- `pts::AbstractMatrix`: The points to evaluate the basis at, (M,N)
"""
function UnivariateApprox.Evaluate!(eval_space::NTuple{N, AbstractMatrix{U}},
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
function UnivariateApprox.EvalDiff!(
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
function UnivariateApprox.EvalDiff2!(
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
function UnivariateApprox.Evaluate(p::NTuple{N,Int}, basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    eval_space = ntuple(j -> similar(pts, (p[j] + 1, size(pts, 1))), N)
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
function UnivariateApprox.EvalDiff(p::NTuple{N,Int}, basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    eval_space = ntuple(j -> similar(pts, (p[j] + 1, size(pts, 1))), N)
    diff_space = ntuple(j -> similar(pts, (p[j] + 1, size(pts, 1))), N)
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
function UnivariateApprox.EvalDiff2(p::NTuple{N,Int}, basis::MultivariateBasis{N}, pts::AbstractMatrix{U}) where {N, U}
    eval_space = ntuple(j -> similar(pts, (p[j] + 1, size(pts, 1))), N)
    diff_space = ntuple(j -> similar(pts, (p[j] + 1, size(pts, 1))), N)
    diff2_space = ntuple(j -> similar(pts, (p[j] + 1, size(pts, 1))), N)
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
function basisAssembly!(out::V1, fmset::FixedMultiIndexSet{d},
        coeffs::V2, univariateEvals::NTuple{d, M}; kwargs...) where {
        d, U, V1 <: AbstractVector{U}, V2 <: AbstractVector{U}, M <: AbstractMatrix{U}}
    # M = num points, N = num multi-indices, d = input dimension
    # out = (M,)
    # coeffs = (N,)
    # univariateEvals = (d,(p_j, M))
    N_midx = length(fmset.starts) - 1
    M_pts = length(out)
    @argcheck length(coeffs)==N_midx DimensionMismatch

    @inbounds for i in 1:d
        @argcheck size(univariateEvals[i], 1)>=fmset.max_orders[i] + 1
        @argcheck size(univariateEvals[i], 2)==M_pts DimensionMismatch
    end
    @argcheck AK.get_backend(out) == AK.get_backend(univariateEvals[1])
    @argcheck AK.get_backend(out) == AK.get_backend(coeffs)
    @argcheck AK.get_backend(out) == AK.get_backend(fmset.starts)

    (;starts, nz_indices, nz_values) = fmset

    AK.foreachindex(out; kwargs...) do i; @inbounds begin
        out[i] = zero(U)
        for midx in 1:N_midx
            start_midx = starts[midx]
            end_midx = starts[midx + 1] - 1

            termVal = one(U)
            for j in start_midx:end_midx
                dim = nz_indices[j]
                power = nz_values[j]
                termVal *= univariateEvals[dim][power + 1, i]
            end
            out[i] = muladd(coeffs[midx], termVal, out[i])
        end
    end; end
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
function Evaluate!(out::M, fmset::FixedMultiIndexSet{d},
        univariateEvals::NTuple{d, M}) where {d, U, M <: AbstractMatrix{U}}
    # M = num points, N = num multi-indices, d = input dimension
    # out = (N,M)
    # univariateEvals = (d,(p_j, M))
    N_midx = length(fmset)
    M_pts = size(out, 2)
    @argcheck size(out, 1)==N_midx DimensionMismatch

    @inbounds for i in 1:d
        @argcheck size(univariateEvals[i], 1)>=fmset.max_orders[i] + 1
        @argcheck size(univariateEvals[i], 2)==M_pts DimensionMismatch
    end
    (;starts, nz_indices, nz_values) = fmset
    AK.foreachindex(out) do idx; @inbounds begin
        pt_idx, midx = (idx - 1) ÷ N_midx + 1, (idx - 1) % N_midx + 1 # (col, row)
        start_midx = starts[midx]
        end_midx = starts[midx + 1] - 1

        termVal = one(U)
        for j in start_midx:end_midx
            dim = nz_indices[j]
            power = nz_values[j]
            termVal *= univariateEvals[dim][power + 1, pt_idx]
        end
        out[midx, pt_idx] = termVal
    end; end
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
function basesAverage!(out::AbstractVector{U}, fmset::FixedMultiIndexSet{d},
        univariateEvals::NTuple{d, M}) where {d, U, M <: AbstractMatrix{U}}
    N_midx = length(fmset)
    M_pts = size(univariateEvals[1], 2)
    @argcheck length(out)==N_midx

    @inbounds for i in 1:d
        @argcheck size(univariateEvals[i], 1)>=fmset.max_orders[i] + 1
        @argcheck size(univariateEvals[i], 2)==M_pts DimensionMismatch
    end
    (;starts, nz_indices, nz_values) = fmset
    backend = AK.get_backend(out)
    if backend isa GPU
        AK.mapreduce(+, CartesianIndices((M_pts, N_midx)), AK.get_backend(out); dims=1, init=zero(U), temp=out) do idx; @inbounds begin
            pt_idx, midx = Tuple(idx)
            start_midx = starts[midx]
            end_midx = starts[midx + 1] - 1

            tmp = one(U)
            for j in start_midx:end_midx
                dim = nz_indices[j]
                power = nz_values[j]
                tmp *= univariateEvals[dim][power + 1, pt_idx]
            end
            tmp
        end; end
        out ./= M_pts
    else
        # TODO: AK/OMT can't do reduction over one dimension. Replace once feature is added
        AK.foreachindex(out) do midx; @inbounds begin
            out[midx] = zero(U)
            for pt_idx in 1:M_pts
                termVal = one(U)
                for j in starts[midx]:starts[midx + 1] - 1
                    dim = nz_indices[j]
                    power = nz_values[j]
                    termVal *= univariateEvals[dim][power + 1, pt_idx]
                end
                out[midx] += termVal
            end
            out[midx] /= M_pts
        end; end
    end
    nothing
end

"""
    basisAssembly(fmset, coeffs, univariateEvals)

Out-of-place assembly of multivariate basis

See [`basisAssembly!`](@ref) for details.
"""
function basisAssembly(
        fmset::FixedMultiIndexSet{d}, coeffs::AbstractVector{U}, univariateEvals::NTuple{d,AbstractMatrix{U}}) where {d, U}
    M = size(univariateEvals[1], 2)
    out = zeros(get_backend(coeffs), U, (M,))
    basisAssembly!(out, fmset, coeffs, univariateEvals)
    out
end

"""
    Evaluate(fmset, univariateEvals)

Out-of-place evaluation of multivariate expansion

See [`Evaluate!`](@ref) for details.
"""
function Evaluate(fmset::FixedMultiIndexSet{d}, univariateEvals::NTuple{d,AbstractMatrix{U}}) where {d,U}
    N, M = length(fmset), size(univariateEvals[1], 2)
    out = zeros(get_backend(univariateEvals[1]), U, (N, M))
    Evaluate!(out, fmset, univariateEvals)
    out
end

"""
    basesAverage(fmset, univariateEvals)

Out-of-place evaluation of the average of a set of basis functions

See [`basesAverage!`](@ref) for details.
"""
function basesAverage(fmset::FixedMultiIndexSet{d}, univariateEvals::NTuple{d,AbstractMatrix{U}}) where {d,U}
    N = length(fmset)
    out = zeros(get_backend(univariateEvals[1]), U, (N,))
    basesAverage!(out, fmset, univariateEvals)
    out
end