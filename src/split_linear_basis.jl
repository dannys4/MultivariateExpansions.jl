# Create an expansion that splits the dimensions into linear and nonlinear inputs
struct SplitLinearBasis{B<:AbstractMultivariateBasis,V::AbstractVector{<:Integer}} <: AbstractMultivariateBasis
    basis::B
    basis_idxs::V
    dimension::Int
    function SplitLinearBasis(nonlinear_basis::_B, basis_idxs::_V, total_dimension::Int) where {_B,_V}
        @argcheck length(basis_idxs) == length(nonlinear_basis) DimensionMismatch
        new{_B,_V}(nonlinear_basis, basis_idxs, total_dimension)
    end
end
