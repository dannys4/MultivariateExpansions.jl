export MollifiedBasis
export SquaredExponential, GaspariCohn, ExponentialFilter

abstract type Mollifier end

"""
    MollifiedBasis(start, basis, moll)

Take a basis and "mollify" it by a mollifier `moll`. Only affects basis functions of degree `start` or higher.

One example is [Hermite Functions](https://en.wikipedia.org/wiki/Hermite_polynomials?oldformat=true#Hermite_functions)

# Example
```jldoctest
julia> basis = ProbabilistHermitePolynomial();

julia> moll = SquaredExponential();

julia> mollified_basis = MollifiedBasis(2, basis, moll); # starts mollifying at quadratic term
```
"""
struct MollifiedBasis{Start, B <: UnivariateBasis, M <: Mollifier} <: UnivariateBasis
    basis::B
    moll::M

    function MollifiedBasis(start::Int, basis::_B, moll::_M) where {_B, _M}
        new{start, _B, _M}(basis, moll)
    end
end

# Mollification m(x) = exp(-x^2/4)/sqrt(4pi)
struct SquaredExponential <: Mollifier end

Evaluate(::SquaredExponential, x) = exp(-x^2 / 4) / sqrt(4pi)
function EvalDiff(::SquaredExponential, x)
    (exp(-x^2 / 4) / sqrt(4pi), (-x / 2) * exp(-x^2 / 4) / sqrt(4pi))
end
function EvalDiff2(::SquaredExponential, x)
    (exp(-x^2 / 4) / sqrt(4pi), (-x / 2) * exp(-x^2 / 4) / sqrt(4pi),
        (x^2 / 2 - 1) * exp(-x^2 / 4) / sqrt(16pi))
end

# Mollification m(x) ≈ exp(-x^2/2) for some scaling with m(x) = 0 for |x| > B that's twice continuously differentiable
struct GaspariCohn{S <: Real} <: Mollifier
    input_scale::S
    function GaspariCohn(bound::T) where {T}
        new{T}(2 / bound)
    end
end

@inline function _gaspari_cohn_small(ra)
    @muladd (((-0.25 * ra + 0.5) * ra + 0.625) * ra + (-5.0 / 3.0)) * ra * ra + 1.0
    # - 0.25(ra^4) + 0.5(ra^3) + 0.625(ra^2) - (5/3)ra^2 + 1
end

@inline function _gaspari_cohn_diff_small(ra)
    @muladd (((-1.25 * ra + 2.0) * ra + 1.875) * ra + (-10.0 / 3.0)) * ra
    # - 1.25(ra^4) + 2(ra^3) + 1.875(ra^2) - (10/3)ra
end

@inline function _gaspari_cohn_diff2_small(ra)
    @muladd (((-5.0 * ra + 6.0) * ra + 3.75) * ra + (-10.0 / 3.0))
    # - 5(ra^3) + 6(ra^2) + 3.75ra -10/3
end

@inline function _gaspari_cohn_large(ra)
    @muladd (((((1.0 / 12.0) * ra + (-0.5)) * ra + 0.625) * ra + (5.0 / 3.0)) * ra + (-5.0)) * ra + 4 + (-2.0 / 3.0) / ra
    # (1/12)*(ra^5) - 0.5(ra^4) + 0.625(ra^3) + (5/3)(ra^2) - 5(ra) + 4 - (2/3) / ra
end

@inline function _gaspari_cohn_diff_large(ra)
    @muladd (((((5.0 / 12.0) * ra + (-2.0)) * ra + 1.875) * ra + (10.0 / 3.0)) * ra + (-5.0) + (2.0 / 3.0) / (ra * ra))
    # (5/12)*(ra^4) - 2(ra^3) + 1.875(ra^2) + (10/3)ra -5  + (2/3) / (ra^2)
end

@inline function _gaspari_cohn_diff2_large(ra)
    @muladd (((5.0 / 3.0) * ra + (-6.0)) * ra + 3.75) * ra + (10.0 / 3.0) + (-4.0 / 3.0) / (ra * ra * ra)
    # (5/3)*(ra^3) - 6(ra^2) + 3.75ra + 10/3 - (4/3) / (ra^3)
end

function Evaluate(m::GaspariCohn, x)
    x2 = x * m.input_scale
    ra = abs(x2)
    if ra < 1
        return _gaspari_cohn_small(ra)
    elseif ra < 2
        return _gaspari_cohn_large(ra)
    end
    0.0
end

function EvalDiff(m::GaspariCohn, x)
    x2 = x * m.input_scale
    ra = abs(x2)
    if ra < 1
        return _gaspari_cohn_small(ra), sign(x)*_gaspari_cohn_diff_small(ra)*m.input_scale
    elseif ra < 2
        return _gaspari_cohn_large(ra), sign(x)*_gaspari_cohn_diff_large(ra)*m.input_scale
    end
    0.0, 0.0
end

function EvalDiff2(m::GaspariCohn, x)
    x2 = x * m.input_scale
    ra = abs(x2)
    if ra < 1
        return _gaspari_cohn_small(ra), sign(x)*_gaspari_cohn_diff_small(ra)*m.input_scale, _gaspari_cohn_diff2_small(ra)*m.input_scale*m.input_scale
    elseif ra < 2
        return _gaspari_cohn_large(ra), sign(x)*_gaspari_cohn_diff_large(ra)*m.input_scale, _gaspari_cohn_diff2_large(ra)*m.input_scale*m.input_scale
    end
    0.0, 0.0, 0.0
end

struct ExponentialFilter{S<:Real}<: Mollifier
    gc::GaspariCohn{S}
    function ExponentialFilter(bound::S) where {S}
        new{S}(GaspariCohn(sqrt(bound)))
    end
end

Evaluate(m::ExponentialFilter, x) = Evaluate(m.gc, sqrt(x))
function EvalDiff(m::ExponentialFilter, x)
    eval, diff = EvalDiff(m.gc, sqrt(x))
    eval, diff / (2 * sqrt(x))
end

function EvalDiff2(m::ExponentialFilter, x)
    eval, diff, diff2 = EvalDiff2(m.gc, sqrt(x))
    eval, diff/(2 * sqrt(x)), diff2 / (4 * x) - diff / (4 * sqrt(x * x * x))
end

function Evaluate!(space::AbstractMatrix{U}, basis::MollifiedBasis{Start},
        x::AbstractVector{U}) where {Start, U}
    Evaluate!(space, basis.basis, x)
    @inbounds for i in axes(space, 2)
        moll_i = Evaluate(basis.moll, x[i])
        space[(Start + 1):end, i] .*= moll_i
    end
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U},
        basis::MollifiedBasis{Start}, x::AbstractVector{U}) where {Start, U}
    EvalDiff!(eval_space, diff_space, basis.basis, x)
    @inbounds for i in axes(eval_space, 2)
        eval_i, diff_i = EvalDiff(basis.moll, x[i])
        @simd for j in Start+1:size(eval_space, 1)
            diff_space[j, i] = diff_space[j, i] * eval_i + eval_space[j, i] * diff_i
            eval_space[j, i] *= eval_i
        end
    end
end

function EvalDiff2!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U},
        diff2_space::AbstractMatrix{U}, basis::MollifiedBasis{Start}, x::AbstractVector{U}) where {Start, U}
    EvalDiff2!(eval_space, diff_space, diff2_space, basis.basis, x)
    @inbounds for i in axes(eval_space, 2)
        eval_i, diff_i, diff2_i = EvalDiff2(basis.moll, x[i])
        @simd for j in Start+1:size(eval_space, 1)
            @muladd diff2_space[j, i] = diff2_space[j, i] * eval_i + 2 * diff_space[j, i] * diff_i + eval_space[j, i] * diff2_i
            @muladd diff_space[j, i] = diff_space[j, i] * eval_i + eval_space[j, i] * diff_i
            eval_space[j, i] *= eval_i
        end
    end
end