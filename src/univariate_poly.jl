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

# _MonicJacobiAk(k::Int,p::Tuple) = (p[2]^2-p[1]^2)/((p[1]+p[2]+2k)*(p[1]+p[2]+2k+2))
# _MonicJacobiBk(k::Int,p::Tuple) = 4k*(k+p[1])*(k+p[2])*(k+p[1]+p[2])/(((2k+p[1]+p[2])^2)*(2k+p[1]+p[2]+1)*(2k+p[1]+p[2]-1))

struct _MonicJacobiCoeffs{r,AB} end

function (j::_MonicJacobiCoeffs{Val{Recurrence.Ak},Val{AB}})(k::Int) where {AB}
    alpha,beta = AB
    (beta^2-alpha^2)/((alpha+beta+2k)*(alpha+beta+2k+2))
end

function (j::_MonicJacobiCoeffs{Val{Recurrence.Bk},Val{AB}})(k::Int) where {AB}
    alpha, beta = AB
    4k*(k+alpha)*(k+beta)*(k+alpha+beta)/(((2k+alpha+beta)^2)*(2k+alpha+beta+1)*(2k+alpha+beta-1))
end

function MonicJacobiPolynomial(alpha::T,beta::T) where {T}
    Ak = _JacobiCoeffs{Val{Recurrence.Ak},Val{(alpha,beta)}}()
    Bk = _JacobiCoeffs{Val{Recurrence.Bk},Val{(alpha,beta)}}()
    MonicOrthogonalPolynomial{Ak,Bk}()
end


# function _JacobiLk(k::Int,::Val{AB}) where {AB}
#     alpha, beta = AB
#     2(k+1)*(k+1+alpha+beta)*(2k+alpha+beta)
# end

# function _JacobiMk(k::Int,::Val{AB}) where {AB}
#     alpha, beta = AB
#     (2k+alpha+beta+1)*(2k+2+alpha+beta)*(2k+alpha+beta)
# end

# function _JacobiAk(k::Int,::Val{AB}) where {AB}
#     alpha,beta = AB
#     (beta^2-alpha^2)*(2k+alpha+beta+1)
# end

# function _JacobiBk(k::Int,::Val{AB}) where {AB}
#     alpha, beta = AB
#     2*(k+alpha)*(k+beta)*(2k+alpha+beta+2)
# end

struct _JacobiCoeffs{r,AB} end

function (j::_JacobiCoeffs{Val{Recurrence.Lk},Val{AB}})(k::Int) where {AB}
    alpha, beta = AB
    2(k+1)*(k+1+alpha+beta)*(2k+alpha+beta)
end

function (j::_JacobiCoeffs{Val{Recurrence.Mk},Val{AB}})(k::Int) where {AB}
    alpha, beta = AB
    (2k+alpha+beta+1)*(2k+2+alpha+beta)*(2k+alpha+beta)
end

function (j::_JacobiCoeffs{Val{Recurrence.Ak},Val{AB}})(k::Int) where {AB}
    alpha,beta = AB
    (beta^2-alpha^2)*(2k+alpha+beta+1)
end

function (j::_JacobiCoeffs{Val{Recurrence.Bk},Val{AB}})(k::Int) where {AB}
    alpha, beta = AB
    2*(k+alpha)*(k+beta)*(2k+alpha+beta+2)
end

# function JacobiPolynomial(alpha::T,beta::T) where {T}
#     Lk = _JacobiCoeffs{Val{Recurrence.Lk},Val{(alpha,beta)}}()
#     Mk = _JacobiCoeffs{Val{Recurrence.Mk},Val{(alpha,beta)}}()
#     Ak = _JacobiCoeffs{Val{Recurrence.Ak},Val{(alpha,beta)}}()
#     Bk = _JacobiCoeffs{Val{Recurrence.Bk},Val{(alpha,beta)}}()
#     OrthogonalPolynomial(Lk,Mk,Ak,Bk)
# end

function JacobiPolynomial(alpha::T,beta::T) where {T}
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
    space[1,:] .= one(U)
    deg == 0 && return
    ak, bk = poly.ak, poly.bk
    space[2,:] .= x .- ak(0)
    @inbounds for k in 1:deg-1
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
"""
##
N_pts, deg = 10000, 10
alpha, beta = 2.78420, 8.5920
xgrid = range(-1,1,length=N_pts+2)[2:end-1]
space = zeros(deg+1,N_pts)
poly = MonicLegendrePolynomial() #JacobiPolynomial(alpha,beta)
Evaluate!(space,poly,xgrid)
deg1_pred = (alpha+1) .+ (alpha+beta+2) .* (xgrid .- 1)/2
deg2_pred = 0.5*(alpha+1)*(alpha+2) .+ (alpha+2)*(alpha+beta+3)*(xgrid .- 1)/2 .+ 0.5*(alpha+beta+3)*(alpha+beta+4)*((xgrid .- 1)/2).^2
##
fig = Figure()
ax = Axis(fig[1,1])
leg_poly_4 = x->(35x^4 - 30x^2 + 3)/35
leg_poly_5 = x->(63x^5 - 70x^3 + 15x)/63
lines!(ax, xgrid, space[6,:], label="Degree 4, computed")
lines!(ax, xgrid, leg_poly_5.(xgrid), label="Degree 4, Legendre")
# lines!(ax, xgrid, deg2_pred, label="Degree 2, predicted")
axislegend()
fig

##
@variables x α β
sym_space = zeros(typeof(x),deg+1,1)
sym_poly = JacobiPolynomial(α,β)
Evaluate!(sym_space, sym_poly, [x])
poly2 = (α + 1)*(α + 2)//2 + (α + 2)*(α + β + 3)*(x - 1)//2 + (α + β + 3)*(α + β + 4)*((x - 1)//2)^2//2
ff = simplify(substitute(sym_space[3], x=>1))
"""