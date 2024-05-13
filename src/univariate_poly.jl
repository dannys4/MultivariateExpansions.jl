export Evaluate, Evaluate!
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

_MonicLegendreAk = Returns(0)
_MonicLegendreBk(k::Int) = (k*k)/(4k*k-1)
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
function Evaluate!(space::AbstractMatrix{U}, poly::OrthogonalPolynomial{Lk,Mk,Ak,Bk},x::AbstractVector{U}) where {Lk,Mk,Ak,Bk,U}
    N,deg = length(x),size(space,1)-1
    @assert size(space,2) == N
    @assert deg >= 0
    space[0+1,:] .= one(U)
    deg == 0 && return
    lk, mk, ak, bk = poly.lk, poly.mk, poly.ak, poly.bk
    space[1+1,:] .= (mk(0)*x .- ak(0))/lk(0)
    for k in 1:deg-1
        idx = k+1
        for j in eachindex(x)
            space[idx+1,j] = (mk(k)*x[j] - ak(k))*space[idx,j] - bk(k)*space[idx-1,j]
            space[idx+1,j] /= lk(k)
        end
    end
end

function Evaluate!(space::AbstractMatrix{U},poly::MonicOrthogonalPolynomial{Ak,Bk},x::AbstractVector{U}) where {Ak,Bk,U}
    N,deg = length(x),size(space,1)-1
    @assert size(space,2) == N
    @assert deg >= 0
    space[0+1,:] .= one(U)
    deg == 0 && return
    ak, bk = poly.ak, poly.bk
    space[1+1,:] .= x .- ak(0)
    for k in 1:deg-1
        idx = k+1
        for j in eachindex(x)
            space[idx+1,j] = (x[j] - ak(k))*space[idx,j] - bk(k)*space[idx-1,j]
        end
    end
end

function Evaluate(max_degree::Int, poly::Polynomial, x::AbstractVector{U}) where {U}
    space = zeros(U,max_degree+1,length(x))
    Evaluate!(space,poly,x)
    space
end