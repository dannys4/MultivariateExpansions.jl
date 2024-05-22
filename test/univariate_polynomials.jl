p, N_pts = 6, 100
rng = Xoshiro(284028)
pts = 2rand(rng, N_pts) .- 1

@testset "Polynomial Evaluation" begin
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
        legendre_poly = LegendrePolynomial()
        monic_legendre_poly = MonicLegendrePolynomial()
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
            jacobi_poly = JacobiPolynomial(α, β)
            monic_jacobi_poly = MonicJacobiPolynomial(α, β)
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

    @testset "Probabilist Hermite Polynomials" begin
        exact_prob_hermite_polys = [
            Returns(1.),
            identity,
            x->x^2 - 1,
            x->x^3 - 3x,
            x->x^4 - 6x^2 + 3,
            x->x^5 - 10x^3 + 15x
        ]
        prob_hermite_poly = ProbabilistHermite()
        space = zeros(p+1, N_pts)
        Evaluate!(space, prob_hermite_poly, pts)
        for k in eachindex(exact_prob_hermite_polys)
            @test isapprox(space[k,:], exact_prob_hermite_polys[k].(pts), rtol=1e-12)
        end
        out_of_place = Evaluate(p, prob_hermite_poly, pts)
        @test isapprox(out_of_place, space, rtol=1e-12)
    end
end

@testset "Polynomial Derivatives" begin
    @testset "Monomials" begin
        monomials = MultivariateExpansions.MonicOrthogonalPolynomial(Returns(0.),Returns(0.))
        ref_eval_space = zeros(p+1, N_pts)
        Evaluate!(ref_eval_space, monomials, pts)
        eval_space = zeros(p+1, N_pts)
        diff_space = zeros(p+1, N_pts)
        EvalDiff!(eval_space, diff_space, monomials, pts)
        for k in 1:p+1
            @test isapprox(eval_space[k,:], ref_eval_space[k,:], atol=1e-12)
            # k - 1 because k corresponds to the power + 1
            @test isapprox(diff_space[k,:], (k == 1 ? zeros(N_pts) : (k-1)*ref_eval_space[k-1,:]), atol=1e-12)
        end
        out_of_place_eval, out_of_place_diff = EvalDiff(p, monomials, pts)
        @test isapprox(out_of_place_eval, eval_space, atol=1e-12)
        @test isapprox(out_of_place_diff, diff_space, atol=1e-12)
    end

    @testset "ProbabilistHermite" begin
        prob_hermite_poly = ProbabilistHermite()
        ref_eval_space = zeros(p+1, N_pts)
        Evaluate!(ref_eval_space, prob_hermite_poly, pts)
        ref_diff_space = zeros(p+1, N_pts)
        for k in 2:p+1
            ref_diff_space[k,:] = (k-1)*ref_eval_space[k-1,:]
        end
        eval_space = zeros(p+1, N_pts)
        diff_space = zeros(p+1, N_pts)
        EvalDiff!(eval_space, diff_space, prob_hermite_poly, pts)
        @test isapprox(eval_space, ref_eval_space, rtol=1e-12)
        @test isapprox(diff_space, ref_diff_space, rtol=1e-12)

        @test 0 == @allocated(EvalDiff!(eval_space, diff_space, prob_hermite_poly, pts))

        out_of_place_eval, out_of_place_diff = EvalDiff(p, prob_hermite_poly, pts)
        @test isapprox(out_of_place_eval, eval_space, rtol=1e-12)
        @test isapprox(out_of_place_diff, diff_space, rtol=1e-12)
    end

    # Use identity from https://math.stackexchange.com/questions/4751256/first-derivative-of-legendre-polynomial
    @testset "Legendre" begin
        legendre_poly = LegendrePolynomial()
        diff_legendre = (x,n,p_n,p_nm1) -> n*(p_nm1 - x*p_n)/(1-x^2)
        ref_eval_space = zeros(p+1, N_pts)
        Evaluate!(ref_eval_space, legendre_poly, pts)
        ref_diff_space = zeros(p+1, N_pts)
        for k in 2:p+1
            ref_diff_space[k,:] = diff_legendre.(pts, k-1, ref_eval_space[k,:], ref_eval_space[k-1,:])
        end
        eval_space = zeros(p+1, N_pts)
        diff_space = zeros(p+1, N_pts)
        EvalDiff!(eval_space, diff_space, legendre_poly, pts)

        @test 0 == @allocated(EvalDiff!(eval_space, diff_space, legendre_poly, pts))

        @test isapprox(eval_space, ref_eval_space, rtol=1e-12)
        @test isapprox(diff_space, ref_diff_space, rtol=1e-12)
        out_of_place_eval, out_of_place_diff = EvalDiff(p, legendre_poly, pts)
        @test isapprox(out_of_place_eval, eval_space, rtol=1e-12)
        @test isapprox(out_of_place_diff, diff_space, rtol=1e-12)
    end

    @testset "Jacobi (allocations)" begin
        jacobi_poly = JacobiPolynomial(0.5, 0.75)
        eval_space = zeros(p+1, N_pts)
        diff_space = zeros(p+1, N_pts)
        EvalDiff!(eval_space, diff_space, jacobi_poly, pts)
        @test 0 == @allocated(EvalDiff!(eval_space, diff_space, jacobi_poly, pts))
    end
end