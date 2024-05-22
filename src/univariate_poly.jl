export Evaluate, Evaluate!
export EvalDiff, EvalDiff!
export LegendrePolynomial, MonicLegendrePolynomial,
       ProbabilistHermite, PhysicistHermite, JacobiPolynomial, MonicJacobiPolynomial

abstract type Polynomial end

module Recurrence
    @enum _Recurrence Lk Mk Ak Bk
end

# p_{k+1} = (x - a_k)p_k - b_kp_{k-1}
struct MonicOrthogonalPolynomial{Ak,Bk} <: Polynomial
    ak::Ak
    bk::Bk
end

# L_kp_{k+1} = (m_kx - a_k)p_k - b_kp_{k-1}
struct OrthogonalPolynomial{Lk,Mk,Ak,Bk} <: Polynomial
    lk::Lk
    mk::Mk
    ak::Ak
    bk::Bk
end

_ProbHermiteAk = Returns(0)
_ProbHermiteBk = identity
ProbabilistHermite() = MonicOrthogonalPolynomial(_ProbHermiteAk,_ProbHermiteBk)

_PhysHermiteLk = Returns(1)
_PhysHermiteMk = Returns(2)
_PhysHermiteAk = Returns(0)
_PhysHermiteBk(k::Int) = 2k

PhysicistHermite() = OrthogonalPolynomial(_PhysHermiteLk,_PhysHermiteMk,_PhysHermiteAk,_PhysHermiteBk)

_LegendreLk(k::Int) = k+1
_LegendreMk(k::Int) = 2k+1
_LegendreAk(k::Int) = 0
_LegendreBk(k::Int) = k
LegendrePolynomial() = OrthogonalPolynomial(_LegendreLk,_LegendreMk,_LegendreAk,_LegendreBk)

_MonicLegendreAk = Returns(0.)
_MonicLegendreBk(k::Int) = (k*k)/(4k*k-1.)
MonicLegendrePolynomial() = MonicOrthogonalPolynomial(_MonicLegendreAk,_MonicLegendreBk)

function MonicJacobiPolynomial(alpha::T,beta::T) where {T}
    alpha == 0. && beta == 0. && return MonicLegendrePolynomial()

    Ak(k::Int) = (beta^2-alpha^2)/((alpha+beta+2k)*(alpha+beta+2k+2))
    Bk(k::Int) = 4k*(k+alpha)*(k+beta)*(k+alpha+beta)/(((2k+alpha+beta)^2)*(2k+alpha+beta+1)*(2k+alpha+beta-1))
    MonicOrthogonalPolynomial(Ak,Bk)
end

function JacobiPolynomial(alpha::T,beta::T) where {T}
    alpha == 0. && beta == 0. && return LegendrePolynomial()

    Lk(k::Int) = 2(k+1)*(k+1+alpha+beta)*(2k+alpha+beta)
    Mk(k::Int) = (2k+alpha+beta+1)*(2k+2+alpha+beta)*(2k+alpha+beta)
    Ak(k::Int) = (beta^2-alpha^2)*(2k+alpha+beta+1)
    Bk(k::Int) = 2*(k+alpha)*(k+beta)*(2k+alpha+beta+2)
    OrthogonalPolynomial(Lk, Mk, Ak, Bk)
end

# Evaluate the polynomial at x, space: matrix of size (d+1,N_pts) with max degree d
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
end

function Evaluate!(space::AbstractMatrix{U},poly::MonicOrthogonalPolynomial{Ak,Bk},x::AbstractVector{U}) where {Ak,Bk,U}
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
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, poly::MonicOrthogonalPolynomial{Ak,Bk}, x::AbstractVector{U}) where {Ak,Bk,U}
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
            eval_space[idx+1,j] = muladd(x[j] - ak(k), eval_space[idx,j], -bk(k)*eval_space[idx-1,j])
            diff_space[idx+1,j] = muladd(x[j] - ak(k), diff_space[idx,j], muladd(-bk(k), diff_space[idx-1,j], eval_space[idx,j]))
        end
    end
end

function Evaluate(max_degree::Int, poly::Polynomial, x::AbstractVector{U}) where {U}
    space = zeros(U,max_degree+1,length(x))
    Evaluate!(space,poly,x)
    space
end

function EvalDiff(max_degree::Int, poly::Polynomial, x::AbstractVector{U}) where {U}
    eval_space = zeros(U,max_degree+1,length(x))
    diff_space = zeros(U,max_degree+1,length(x))
    EvalDiff!(eval_space,diff_space,poly,x)
    eval_space,diff_space
end