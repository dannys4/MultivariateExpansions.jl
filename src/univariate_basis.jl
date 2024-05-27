export Evaluate, Evaluate!
export EvalDiff, EvalDiff!
export MollifiedBasis

abstract type UnivariateBasis end

abstract type Mollifier end

"""
    MollifiedBasis(start, basis, moll)

Take a basis and "mollify" it by a mollifier `moll`. Only affects basis functions of degree `start` or higher.

One example is [Hermite Functions](https://en.wikipedia.org/wiki/Hermite_polynomials?oldformat=true#Hermite_functions)

# Example
```jldoctest
julia> basis = ProbabilistHermite();

julia> moll = SquaredExponential();

julia> mollified_basis = MollifiedBasis(2, basis, moll); # starts mollifying at quadratic term
```
"""
struct MollifiedBasis{Start,B<:UnivariateBasis,M<:Mollifier} <: UnivariateBasis
    basis::B
    moll::M

    function MollifiedBasis(start::Int, basis::_B, moll::_M) where {_B,_M}
        new{start,_B,_M}(basis,moll)
    end
end

struct SquaredExponential <: Mollifier end

Evaluate(::SquaredExponential, x) = exp(-x^2/4)/sqrt(4pi)
EvalDiff(::SquaredExponential, x) = (exp(-x^2/4)/sqrt(4pi), (-x/2)*exp(-x^2/4)/sqrt(4pi))
EvalDiff2(::SquaredExponential, x) = (exp(-x^2/4)/sqrt(4pi), (-x/2)*exp(-x^2/4)/sqrt(4pi), (x^2/2-1)*exp(-x^2/4)/sqrt(16pi))

function Evaluate!(space::AbstractMatrix{U}, basis::MollifiedBasis{Start}, x::AbstractVector{U}) where {Start,U}
    Evaluate!(space, basis.basis, x)
    @inbounds for i in axes(space,2)
        moll_i = Evaluate(basis.moll, x[i])
        space[Start+1:end,i] .*= moll_i
    end
end

function EvalDiff!(eval_space::AbstractMatrix{U}, diff_space::AbstractMatrix{U}, basis::MollifiedBasis{Start}, x::AbstractVector{U}) where {Start,U}
    EvalDiff!(eval_space, diff_space, basis.basis, x)
    @inbounds for i in axes(eval_space,2)
        eval_i, diff_i = EvalDiff(basis.moll, x[i])
        @simd for j in (Start+1):size(eval_space,1)
            diff_space[j,i] = diff_space[j,i]*eval_i + eval_space[j,i]*diff_i
            eval_space[j,i] *= eval_i
        end
    end
end

"""
    Evaluate(max_degree, basis, x)

Evaluate the univariate basis `basis` at `x` and return the result.

# Example
```jldoctest
julia> Evaluate(2, LegendrePolynomial(), [0.5, 0.75])
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375
```

See also: [`Evaluate!`](@ref)
"""
function Evaluate(max_degree::Int, basis::UnivariateBasis, x::AbstractVector{U}) where {U}
    space = zeros(U,max_degree+1,length(x))
    Evaluate!(space,basis,x)
    space
end

"""
    EvalDiff(max_degree, basis, x)

Evaluate the univariate basis `basis` and its derivative at `x` and return the result.

# Example
```jldoctest
julia> eval_space, diff_space = EvalDiff(2, LegendrePolynomial(), [0.5, 0.75])
([1.0 1.0; 0.5 0.75; -0.125 0.34375], [0.0 0.0; 1.0 1.0; 1.5 2.25])

julia> eval_space
3×2 Matrix{Float64}:
  1.0    1.0
  0.5    0.75
 -0.125  0.34375

julia> diff_space
3×2 Matrix{Float64}:
 0.0  0.0
 1.0  1.0
 1.5  2.25
```

See also: [`EvalDiff!`](@ref)
"""
function EvalDiff(max_degree::Int, basis::UnivariateBasis, x::AbstractVector{U}) where {U}
    eval_space = zeros(U,max_degree+1,length(x))
    diff_space = zeros(U,max_degree+1,length(x))
    EvalDiff!(eval_space,diff_space,basis,x)
    eval_space,diff_space
end