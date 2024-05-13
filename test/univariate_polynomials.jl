p, N_pts = 6, 1000
rng = Xoshiro(284028)
pts = 2rand(rng, N_pts) .- 1

@testset "Monomial Eval" begin
    # Monomial is x^{k+1} = (x-0)x^k + 0x^{k-1}
    monomials = MultivariateExpansions.MonicOrthogonalPolynomial(Returns(0.),Returns(0.))
    space = zeros(p+1, N_pts)
    Evaluate!(space, monomials, pts)
    for k in 0:p
        @test isapprox(space[k+1,:], pts.^k, atol=1e-12)
    end
    out_of_place = Evaluate(p, monomials, pts)
    @test isapprox(out_of_place, space, atol=1e-12)
end

@testset "Legendre Polynomials" begin
    exact_legendre_polys = [
        Returns(1.),
        identity,
        x->(3x^2-1)/2,
        x->(5x^3-3x)/2,
        x->(35x^4-30x^2+3)/8,
        x->(63x^5-70x^3+15x)/8,
        x->(231x^6-315x^4+105x^2-5)/16
    ]
    legendre_lead_coeffs = [1, 1, 3/2, 5/2, 35/8, 63/8, 231/16]
    legendre_poly = MultivariateExpansions.LegendrePolynomial()
    monic_legendre_poly = MultivariateExpansions.MonicLegendrePolynomial()
    space = zeros(p+1, N_pts)
    monic_space = zeros(p+1, N_pts)
    Evaluate!(space, legendre_poly, pts)
    Evaluate!(monic_space, monic_legendre_poly, pts)
    for k in eachindex(exact_legendre_polys)
        @test isapprox(space[k,:], exact_legendre_polys[k].(pts), rtol=1e-12)
        @test isapprox(monic_space[k,:], exact_legendre_polys[k].(pts)/legendre_lead_coeffs[k], rtol=1e-12)
    end
    out_of_place = Evaluate(p, monic_legendre_poly, pts)
    @test isapprox(out_of_place, monic_space, rtol=1e-12)
end

@testset "Jacobi Polynomials" begin
    param_grid = 0:0.25:1
    N_param = length(param_grid)
    alphas = vec(repeat(param_grid,N_param,1))
    betas = vec(repeat(param_grid,1,N_param)')
    for (α,β) in zip(alphas, betas)
        exact_jacobi_polys = [
            Returns(1.),
            x->(α+1) + (α+β+2)*(x-1)/2,
            x->0.125*(4(α+1)*(α+2) + 4(α+2)*(α+β+3)*(x-1) + (α+β+3)*(α+β+4)*(x-1)^2)
        ]
        jacobi_lead_coeffs = [1, (α+β+2)/2, (α+β+3)*(α+β+4)/8]
        jacobi_poly = MultivariateExpansions.JacobiPolynomial(α, β)
        monic_jacobi_poly = MultivariateExpansions.MonicJacobiPolynomial(α, β)
        space = zeros(p+1, N_pts)
        monic_space = zeros(p+1, N_pts)
        Evaluate!(space, jacobi_poly, pts)
        Evaluate!(monic_space, monic_jacobi_poly, pts)
        for k in eachindex(exact_jacobi_polys)
            @test isapprox(space[k,:], exact_jacobi_polys[k].(pts), rtol=1e-12)
            @test isapprox(monic_space[k,:], exact_jacobi_polys[k].(pts)/jacobi_lead_coeffs[k], rtol=1e-6)
        end
        out_of_place = Evaluate(p, jacobi_poly, pts)
        @test isapprox(out_of_place, space, rtol=1e-12)
    end
end