var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MultivariateExpansions","category":"page"},{"location":"#MultivariateExpansions","page":"Home","title":"MultivariateExpansions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MultivariateExpansions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MultivariateExpansions]","category":"page"},{"location":"#MultivariateExpansions.MollifiedBasis","page":"Home","title":"MultivariateExpansions.MollifiedBasis","text":"MollifiedBasis(start, basis, moll)\n\nTake a basis and \"mollify\" it by a mollifier moll. Only affects basis functions of degree start or higher.\n\nOne example is Hermite Functions\n\nExample\n\njulia> basis = ProbabilistHermitePolynomial();\n\njulia> moll = SquaredExponential();\n\njulia> mollified_basis = MollifiedBasis(2, basis, moll); # starts mollifying at quadratic term\n\n\n\n\n\n","category":"type"},{"location":"#MultivariateExpansions.MonicOrthogonalPolynomial","page":"Home","title":"MultivariateExpansions.MonicOrthogonalPolynomial","text":"Struct representing a monic orthogonal polynomial family via three-term recurrence relation:\n\np_k+1 = (x - a_k)p_k - b_kp_k-1\n\n\n\n\n\n","category":"type"},{"location":"#MultivariateExpansions.OrthogonalPolynomial","page":"Home","title":"MultivariateExpansions.OrthogonalPolynomial","text":"Struct representing an orthogonal polynomial family via three-term recurrence relation:\n\nL_kp_k+1 = (m_kx - a_k)p_k - b_kp_k-1\n\n\n\n\n\n","category":"type"},{"location":"#MultivariateExpansions.EvalDiff!","page":"Home","title":"MultivariateExpansions.EvalDiff!","text":"EvalDiff!(eval_space::AbstractMatrix, diff_space::AbstractMatrix, basis::UnivariateBasis, x::AbstractVector)\n\nEvaluate the univariate basis basis and its derivative at x and store the results in eval_space and diff_space, respectively.\n\nExample\n\njulia> eval_space = zeros(3,2);\n\njulia> diff_space = zeros(3,2);\n\njulia> EvalDiff!(eval_space, diff_space, LegendrePolynomial(), [0.5, 0.75])\n\njulia> eval_space\n3×2 Matrix{Float64}:\n  1.0    1.0\n  0.5    0.75\n -0.125  0.34375\n\njulia> diff_space\n3×2 Matrix{Float64}:\n 0.0  0.0\n 1.0  1.0\n 1.5  2.25\n\nSee also: EvalDiff\n\n\n\n\n\n","category":"function"},{"location":"#MultivariateExpansions.EvalDiff-Union{Tuple{U}, Tuple{Int64, MultivariateExpansions.UnivariateBasis, AbstractVector{U}}} where U","page":"Home","title":"MultivariateExpansions.EvalDiff","text":"EvalDiff(max_degree::Int, basis::UnivariateBasis, x::AbstractVector)\n\nEvaluate the univariate basis basis and its derivative at x and return the result.\n\nExample\n\njulia> eval_space, diff_space = EvalDiff(2, LegendrePolynomial(), [0.5, 0.75]);\n\njulia> eval_space\n3×2 Matrix{Float64}:\n  1.0    1.0\n  0.5    0.75\n -0.125  0.34375\n\njulia> diff_space\n3×2 Matrix{Float64}:\n 0.0  0.0\n 1.0  1.0\n 1.5  2.25\n\nSee also: EvalDiff!\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.EvalDiff2!","page":"Home","title":"MultivariateExpansions.EvalDiff2!","text":"EvalDiff2!(eval_space::AbstractMatrix, diff_space::AbstractMatrix, diff2_space::AbstractMatrix, basis::UnivariateBasis, x::AbstractVector)\n\nEvaluate the univariate basis basis and its first two derivatives at x and store the results in eval_space, diff_space and diff2_space, respectively.\n\nExample\n\njulia> eval_space = zeros(3,2);\n\njulia> diff_space = zeros(3,2);\n\njulia> diff2_space = zeros(3,2);\n\njulia> EvalDiff2!(eval_space, diff_space, diff2_space, LegendrePolynomial(), [0.5, 0.75])\n\njulia> eval_space\n3×2 Matrix{Float64}:\n  1.0    1.0\n  0.5    0.75\n -0.125  0.34375\n\njulia> diff_space\n3×2 Matrix{Float64}:\n 0.0  0.0\n 1.0  1.0\n 1.5  2.25\n\njulia> diff2_space\n3×2 Matrix{Float64}:\n 0.0  0.0\n 0.0  0.0\n 3.0  3.0\n\nSee also: EvalDiff2\n\n\n\n\n\n","category":"function"},{"location":"#MultivariateExpansions.EvalDiff2-Union{Tuple{U}, Tuple{Int64, MultivariateExpansions.UnivariateBasis, AbstractVector{U}}} where U","page":"Home","title":"MultivariateExpansions.EvalDiff2","text":"EvalDiff2(max_degree::Int, basis::UnivariateBasis, x::AbstractVector)\n\nEvaluate the univariate basis basis and its first two derivatives at x and return the results.\n\nExample\n\njulia> eval_space, diff_space, diff2_space = EvalDiff2(2, LegendrePolynomial(), [0.5, 0.75]);\n\njulia> eval_space\n3×2 Matrix{Float64}:\n  1.0    1.0\n  0.5    0.75\n -0.125  0.34375\n\njulia> diff_space\n3×2 Matrix{Float64}:\n 0.0  0.0\n 1.0  1.0\n 1.5  2.25\n\njulia> diff2_space\n3×2 Matrix{Float64}:\n 0.0  0.0\n 0.0  0.0\n 3.0  3.0\n\nSee also: EvalDiff2!\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.Evaluate!","page":"Home","title":"MultivariateExpansions.Evaluate!","text":"Evaluate!(space::AbstractMatrix, basis::UnivariateBasis, x::AbstractVector)\n\nEvaluate the univariate basis basis at x and store the result in space.\n\nExample\n\njulia> space = zeros(3,2);\n\njulia> Evaluate!(space, LegendrePolynomial(), [0.5, 0.75])\n\njulia> space\n3×2 Matrix{Float64}:\n  1.0    1.0\n  0.5    0.75\n -0.125  0.34375\n\nSee also: Evaluate\n\n\n\n\n\n","category":"function"},{"location":"#MultivariateExpansions.Evaluate-Union{Tuple{U}, Tuple{Int64, MultivariateExpansions.UnivariateBasis, AbstractVector{U}}} where U","page":"Home","title":"MultivariateExpansions.Evaluate","text":"Evaluate(max_degree::Int, basis::UnivariateBasis, x::AbstractVector)\n\nEvaluate the univariate basis basis at x and return the result.\n\nExample\n\njulia> Evaluate(2, LegendrePolynomial(), [0.5, 0.75])\n3×2 Matrix{Float64}:\n  1.0    1.0\n  0.5    0.75\n -0.125  0.34375\n\nSee also: Evaluate!\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.JacobiPolynomial-Union{Tuple{T}, Tuple{T, T}} where T","page":"Home","title":"MultivariateExpansions.JacobiPolynomial","text":"JacobiPolynomial(α,β)\n\nJacobi polynomials P^(α,β)_k, orthogonal on [-1,1] with weight (1-x)^α(1+x)^β\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.MonicJacobiPolynomial-Union{Tuple{T}, Tuple{T, T}} where T","page":"Home","title":"MultivariateExpansions.MonicJacobiPolynomial","text":"MonicJacobiPolynomial(α,β)\n\nMonic Jacobi polynomials P^(α,β)_k, orthogonal on [-1,1] with weight (1-x)^α(1+x)^β\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.polynomialAssembly!-Union{Tuple{T}, Tuple{d}, Tuple{AbstractVector, MultiIndexing.FixedMultiIndexSet{d}, AbstractVector, Tuple{Vararg{T, d}}}} where {d, T<:(AbstractMatrix)}","page":"Home","title":"MultivariateExpansions.polynomialAssembly!","text":"polynomialAssembly!(out, fmset, coeffs, univariateEvals)\n\nEvaluate a polynomial expansion at a set of points given the coefficients and univariate evaluations at each marginal point.\n\nArguments\n\nout::AbstractVector: The output vector to store the polynomial evaluations, (M,)\nfmset::FixedMultiIndexSet{d}: The fixed multi-index set defining the polynomial space (N,d)\ncoeffs::AbstractVector: The coefficients of the polynomial expansion (N,)\nunivariateEvals::NTuple{d, AbstractMatrix}: The univariate evaluations at each marginal point (d,(pj, M)), where pj is the maximum degree of the j-th univariate polynomial\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.polynomialAssembly-Union{Tuple{d}, Tuple{MultiIndexing.FixedMultiIndexSet{d}, AbstractVector, Any}} where d","page":"Home","title":"MultivariateExpansions.polynomialAssembly","text":"polynomialAssembly(fmset, coeffs, univariateEvals)\n\nOut-of-place assembly of multivariate polynomial\n\nSee polynomialAssembly! for details.\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.polynomialsAverage!-Union{Tuple{T}, Tuple{d}, Tuple{AbstractVector, MultiIndexing.FixedMultiIndexSet{d}, Tuple{Vararg{T, d}}}} where {d, T<:(AbstractMatrix)}","page":"Home","title":"MultivariateExpansions.polynomialsAverage!","text":"polynomialsAverage!(out, fmset, univariateEvals)\n\nEvaluate the average of a set of polynomials over a set of points given the univariate polynomial evaluations\n\nArguments\n\nout::AbstractVector: The output vector to store the polynomial evaluations, (N,)\nfmset::FixedMultiIndexSet{d}: The fixed multi-index set defining the polynomial space (N,d)\nunivariateEvals::NTuple{d, AbstractMatrix}: The univariate evaluations at each marginal point (d,(pj, M)), where pj is the maximum degree of the j-th univariate polynomial\n\nSee also polynomialsAverage.\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.polynomialsAverage-Union{Tuple{d}, Tuple{MultiIndexing.FixedMultiIndexSet{d}, Any}} where d","page":"Home","title":"MultivariateExpansions.polynomialsAverage","text":"polynomialsAverage(fmset, univariateEvals)\n\nOut-of-place evaluation of the average of a set of polynomials\n\nSee polynomialsAverage! for details.\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.polynomialsEval!-Union{Tuple{T}, Tuple{d}, Tuple{AbstractMatrix, MultiIndexing.FixedMultiIndexSet{d}, Tuple{Vararg{T, d}}}} where {d, T<:(AbstractMatrix)}","page":"Home","title":"MultivariateExpansions.polynomialsEval!","text":"polynomialsEval!(out, fmset, univariateEvals)\n\nEvaluate a set of multivariate polynomials given the evaluations of the univariate polynomials\n\nArguments\n\nout::AbstractMatrix: The output matrix to store the polynomial evaluations, (N,M)\nfmset::FixedMultiIndexSet{d}: The fixed multi-index set defining the polynomial space (N,d)\nunivariateEvals::NTuple{d, AbstractMatrix}: The univariate evaluations at each marginal point (d,(pj, M)), where pj is the maximum degree of the j-th univariate polynomial\n\nSee also polynomialsEval.\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.polynomialsEval-Union{Tuple{d}, Tuple{MultiIndexing.FixedMultiIndexSet{d}, Any}} where d","page":"Home","title":"MultivariateExpansions.polynomialsEval","text":"polynomialsEval(fmset, univariateEvals)\n\nOut-of-place evaluation of multivariate polynomials\n\nSee polynomialsEval! for details.\n\n\n\n\n\n","category":"method"},{"location":"#MultivariateExpansions.score_loss-Union{Tuple{NT}, Tuple{d}, Tuple{MultiIndexing.FixedMultiIndexSet{d}, AbstractVector, NT, NT, NT}} where {d, NT<:Tuple{Vararg{AbstractMatrix, d}}}","page":"Home","title":"MultivariateExpansions.score_loss","text":"score_loss(fmset, coeffs, univariateEval, univariateDiff, univariateDiff2)\n\n\n\n\n\n","category":"method"}]
}
