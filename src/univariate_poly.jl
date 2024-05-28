export LegendrePolynomial, MonicLegendrePolynomial,
       ProbabilistHermitePolynomial, PhysicistHermitePolynomial,
       JacobiPolynomial, MonicJacobiPolynomial,
       LaguerrePolynomial

# Abstract type for univariate polynomials
abstract type Polynomial <: UnivariateBasis end

"""
Struct representing a _monic_ orthogonal polynomial family via three-term recurrence relation:

```math
p_{k+1} = (x - a_k)p_k - b_kp_{k-1}
```
"""
struct MonicOrthogonalPolynomial{Ak,Bk} <: Polynomial
    ak::Ak
    bk::Bk
end


"""
Struct representing an orthogonal polynomial family via three-term recurrence relation:

```math
L_kp_{k+1} = (m_kx - a_k)p_k - b_kp_{k-1}
```
"""
struct OrthogonalPolynomial{Lk,Mk,Ak,Bk} <: Polynomial
    lk::Lk
    mk::Mk
    ak::Ak
    bk::Bk
end

_ProbHermiteAk = Returns(0)
_ProbHermiteBk = identity

# Probabilist Hermite polynomials `He_k`, orthogonal under `exp(-x^2/2)`
ProbabilistHermitePolynomial() = MonicOrthogonalPolynomial(_ProbHermiteAk,_ProbHermiteBk)

_PhysHermiteLk = Returns(1)
_PhysHermiteMk = Returns(2)
_PhysHermiteAk = Returns(0)
_PhysHermiteBk(k::Int) = 2k

# Physicist Hermite polynomials `H_k`, orthogonal under `exp(-x^2)`
PhysicistHermitePolynomial() = OrthogonalPolynomial(_PhysHermiteLk,_PhysHermiteMk,_PhysHermiteAk,_PhysHermiteBk)

_LegendreLk(k::Int) = k+1
_LegendreMk(k::Int) = 2k+1
_LegendreAk = Returns(0)
_LegendreBk(k::Int) = k
# Legendre polynomials `P_k`, orthogonal under `U[-1,1]`
LegendrePolynomial() = OrthogonalPolynomial(_LegendreLk,_LegendreMk,_LegendreAk,_LegendreBk)

_MonicLegendreAk = Returns(0.)
_MonicLegendreBk(k::Int) = (k*k)/muladd(4,k*k,-1.)
# Monic Legendre polynomials `P_k`, orthogonal under `U[-1,1]`
MonicLegendrePolynomial() = MonicOrthogonalPolynomial(_MonicLegendreAk,_MonicLegendreBk)

_LaguerreLk(k::Int) = k+1
_LaguerreMk(k::Int) = -1
_LaguerreAk(k::Int) = -muladd(2,k,1)
_LaguerreBk(k::Int) = k
LaguerrePolynomial() = OrthogonalPolynomial(_LaguerreLk,_LaguerreMk,_LaguerreAk,_LaguerreBk)

"""
    MonicJacobiPolynomial(α,β)

Monic Jacobi polynomials `P^(α,β)_k`, orthogonal on `[-1,1]` with weight `(1-x)^α(1+x)^β`
"""
function MonicJacobiPolynomial(alpha::T,beta::T) where {T}
    alpha == 0. && beta == 0. && return MonicLegendrePolynomial()

    Ak(k::Int) = (beta^2-alpha^2)/((alpha+beta+2k)*(alpha+beta+2k+2))
    Bk(k::Int) = 4k*(k+alpha)*(k+beta)*(k+alpha+beta)/(((2k+alpha+beta)^2)*(2k+alpha+beta+1)*(2k+alpha+beta-1))
    MonicOrthogonalPolynomial(Ak,Bk)
end

"""
    JacobiPolynomial(α,β)

Jacobi polynomials `P^(α,β)_k`, orthogonal on `[-1,1]` with weight `(1-x)^α(1+x)^β`
"""
function JacobiPolynomial(alpha::T,beta::T) where {T}
    alpha == 0. && beta == 0. && return LegendrePolynomial()

    Lk(k::Int) = 2(k+1)*(k+1+alpha+beta)*(2k+alpha+beta)
    Mk(k::Int) = (2k+alpha+beta+1)*(2k+2+alpha+beta)*(2k+alpha+beta)
    Ak(k::Int) = (beta^2-alpha^2)*(2k+alpha+beta+1)
    Bk(k::Int) = 2*(k+alpha)*(k+beta)*(2k+alpha+beta+2)
    OrthogonalPolynomial(Lk, Mk, Ak, Bk)
end

function Evaluate!(space::AbstractMatrix{U}, poly::OrthogonalPolynomial,x::AbstractVector{U}) where {U}
    N,deg = length(x),size(space,1)-1
    @assert size(space,2) == N
    @assert deg >= 0 "Degree must be nonnegative"
    if deg == 0
        space[0+1,:] .= one(U)
        return
    end

    lk, mk, ak, bk = poly.lk, poly.mk, poly.ak, poly.bk

    @inbounds for j in eachindex(x)
        space[0+1,j] = one(U)
        space[1+1,j] = muladd(mk(0), x[j], -ak(0))/lk(0)
        @simd for k in 1:deg-1
            idx = k+1
            space[idx+1,j] = muladd(muladd(mk(k), x[j], -ak(k)), space[idx,j], -bk(k)*space[idx-1,j])
            space[idx+1,j] /= lk(k)
        end
    end
    nothing
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, poly::OrthogonalPolynomial, x::AbstractVector{U}) where {U}
    N,deg = length(x),size(eval_space,1)-1
    @assert size(eval_space,2) == N "eval_space has $(size(eval_space,2)) columns, expected $N"
    @assert size(diff_space,1) == deg+1 && size(diff_space,2) == N "diff_space size $(size(diff_space)), expected ($(deg+1),$N)"
    @assert deg >= 0 "Degree must be nonnegative"
    if deg == 0
        eval_space[0+1,:] .= one(U)
        diff_space[0+1,:] .= zero(U)
        return
    end
    lk, mk, ak, bk = poly.lk, poly.mk, poly.ak, poly.bk

    @inbounds for j in eachindex(x)
        eval_space[0+1,j] = one(U)
        diff_space[0+1,j] = zero(U)
        eval_space[1+1,j] = muladd(mk(0), x[j], -ak(0))/lk(0)
        diff_space[1+1,j] = convert(U,mk(0)/lk(0))
        @simd for k in 1:deg-1
            idx = k+1
            pk_ = muladd(mk(k), x[j], -ak(k))
            eval_space[idx+1,j] = muladd(pk_,   eval_space[idx,j], -bk(k)*eval_space[idx-1,j])
            diff_space[idx+1,j] = muladd(mk(k), eval_space[idx,j], muladd(pk_, diff_space[idx,j], -bk(k)*diff_space[idx-1,j]))
            eval_space[idx+1,j] /= lk(k)
            diff_space[idx+1,j] /= lk(k)
        end
    end
    nothing
end

function EvalDiff2!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, diff2_space::AbstractMatrix{U}, poly::OrthogonalPolynomial, x::AbstractVector{U}) where {U}
    N,deg = length(x),size(eval_space,1)-1

    @assert size(eval_space,2) == N "eval_space has $(size(eval_space,2)) columns, expected $N"
    @assert size(diff_space,1) == deg+1 && size(diff_space,2) == N "diff_space size $(size(diff_space)), expected ($(deg+1),$N)"
    @assert size(diff2_space,1) == deg+1 && size(diff2_space,2) == N "diff2_space size $(size(diff2_space)), expected ($(deg+1),$N)"
    @assert deg >= 0 "Degree must be nonnegative"

    if deg == 0
        eval_space[0+1,:] .= one(U)
        diff_space[0+1,:] .= zero(U)
        diff2_space[0+1,:] .= zero(U)
        return
    end
    lk, mk, ak, bk = poly.lk, poly.mk, poly.ak, poly.bk

    @inbounds for j in eachindex(x)
        eval_space[0+1,j] = one(U)
        diff_space[0+1,j] = zero(U)
        diff2_space[0+1,j] = zero(U)
        eval_space[1+1,j] = muladd(mk(0), x[j], -ak(0))/lk(0)
        diff_space[1+1,j] = mk(0)/lk(0)
        diff2_space[1+1,j] = zero(U)

        @simd for k in 1:deg-1
            idx = k+1
            @muladd pk_ = mk(k)*x[j] - ak(k)
            @muladd eval_space[idx+1,j] = pk_*eval_space[idx,j] - bk(k)*eval_space[idx-1,j]
            diff_space[idx+1,j] = muladd(mk(k), eval_space[idx,j], muladd(pk_, diff_space[idx,j], -bk(k)*diff_space[idx-1,j]))
            diff2_space[idx+1,j] = muladd(2mk(k), diff_space[idx,j], muladd(pk_, diff2_space[idx,j], -bk(k)*diff2_space[idx-1,j]))

            eval_space[idx+1,j] /= lk(k)
            diff_space[idx+1,j] /= lk(k)
            diff2_space[idx+1,j] /= lk(k)
        end
    end
    nothing
end

function Evaluate!(space::AbstractMatrix{U},poly::MonicOrthogonalPolynomial,x::AbstractVector{U}) where {U}
    # Evaluate a monic orthogonal polynomial; may be faster than the general case
    N,deg = length(x),size(space,1)-1
    @assert size(space,2) == N
    @assert deg >= 0
    if deg == 0
        space[0+1,:] .= one(U)
        return
    end

    ak, bk = poly.ak, poly.bk

    @inbounds for j in eachindex(x)
        space[0+1,j] = one(U)
        space[1+1,j] = x[j] - ak(0)
        @simd for k in 1:deg-1
            idx = k+1
            space[idx+1,j] = muladd(x[j] - ak(k), space[idx,j], -bk(k)*space[idx-1,j])
        end
    end
    nothing
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, poly::MonicOrthogonalPolynomial, x::AbstractVector{U}) where {U}
    # Evaluate a monic orthogonal polynomial and its derivative; may be faster than the general case
    N,deg = length(x),size(eval_space,1)-1
    @assert size(eval_space,2) == N "eval_space has $(size(eval_space,2)) columns, expected $N"
    @assert size(diff_space,1) == deg+1 && size(diff_space,2) == N "diff_space size $(size(diff_space)), expected ($(deg+1),$N)"
    @assert deg >= 0 "Degree must be nonnegative"
    if deg == 0
        eval_space[0+1,:] .= one(U)
        diff_space[0+1,:] .= zero(U)
        return
    end

    ak, bk = poly.ak, poly.bk

    @inbounds for j in eachindex(x)
        eval_space[0+1,j] = one(U)
        diff_space[0+1,j] = zero(U)

        eval_space[1+1,j] = x[j] - ak(0)
        diff_space[1+1,j] = one(U)

        @simd for k in 1:deg-1
            idx = k+1
            x_sub_a = x[j] - ak(k)
            eval_space[idx+1,j] = muladd(x_sub_a, eval_space[idx,j], -bk(k)*eval_space[idx-1,j])
            diff_space[idx+1,j] = muladd(x_sub_a, diff_space[idx,j], muladd(-bk(k), diff_space[idx-1,j], eval_space[idx,j]))
        end
    end
    nothing
end

function EvalDiff2!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, diff2_space::AbstractMatrix{U}, poly::MonicOrthogonalPolynomial, x::AbstractVector{U}) where {U}
    N,deg = length(x),size(eval_space,1)-1
    @assert size(eval_space,2) == N "eval_space has $(size(eval_space,2)) columns, expected $N"
    @assert size(diff_space,1) == deg+1 && size(diff_space,2) == N "diff_space size $(size(diff_space)), expected ($(deg+1),$N)"
    @assert size(diff2_space,1) == deg+1 && size(diff2_space,2) == N "diff2_space size $(size(diff2_space)), expected ($(deg+1),$N)"
    @assert deg >= 0 "Degree must be nonnegative"

    if deg == 0
        eval_space[0+1,:] .= one(U)
        diff_space[0+1,:] .= zero(U)
        diff2_space[0+1,:] .= zero(U)
        return
    end

    ak, bk = poly.ak, poly.bk

    @inbounds for j in eachindex(x)
        eval_space[0+1,j] = one(U)
        diff_space[0+1,j] = zero(U)
        diff2_space[0+1,j] = zero(U)

        eval_space[1+1,j] = x[j] - ak(0)
        diff_space[1+1,j] = one(U)
        diff2_space[1+1,j] = zero(U)

        @simd for k in 1:deg-1
            idx = k+1
            x_sub_a = x[j] - ak(k)
            eval_space[idx+1,j] = muladd(x_sub_a, eval_space[idx,j], -bk(k)*eval_space[idx-1,j])
            diff_space[idx+1,j] = muladd(x_sub_a, diff_space[idx,j], muladd(-bk(k), diff_space[idx-1,j], eval_space[idx,j]))
            diff2_space[idx+1,j] = muladd(x_sub_a, diff2_space[idx,j], muladd(-bk(k), diff2_space[idx-1,j], 2*diff_space[idx,j]))
        end
    end
    nothing
end