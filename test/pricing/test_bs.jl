@testset "lognormal" begin
    θ₁ = FinNetValu.BlackScholesParams(0.0, 1.0, 1.0)
    θ₂ = FinNetValu.BlackScholesParams(0.0, 1.0, [1.0, 2.0])
    @test FinNetValu.Aτ(1.0, θ₁, [1.0, 2.0]) == exp.([0.5, 1.5])
    @test FinNetValu.Aτ(1.0, θ₂, [1.0, 2.0]) == exp.([0.5, 2.0])
    @test_throws DimensionMismatch FinNetValu.Aτ(1.0, θ₂, [1.0, 2.0, 3.0])
end

@testset "discount" begin
    θ₁ = FinNetValu.BlackScholesParams(0.0, 1.0, 1.0)
    θ₂ = FinNetValu.BlackScholesParams([0.0, 1.0], [1.0, 2.0], 1.0)
    @test FinNetValu.discount(θ₁) == exp(0)
    @test FinNetValu.discount(θ₂) == exp.([0, -2])
end

@testset "Black Scholes" begin
    r, τ, σ = 0.1, 1.2, 1.2
    θ = FinNetValu.BlackScholesParams(r, τ, σ)
    K = 100.
    S₀ = 110.
    @test FinNetValu.d₋(S₀, K, r, τ, σ) ≈ FinNetValu.d₊(S₀, K, r, τ, σ) - σ * √(τ)
    @test FinNetValu.callprice(S₀, K, θ) >= max(S₀ - K, 0)
    @test FinNetValu.putprice(S₀, K, θ) >= 0 ## TODO: Fixme ...
    ## Check for put-call parity
    @test FinNetValu.callprice(S₀, K, θ) - FinNetValu.putprice(S₀, K, θ) ≈ S₀ - FinNetValu.discount(θ) * K
    @test FinNetValu.putdualΔ(S₀, K, θ) ≈ FinNetValu.discount(θ) + FinNetValu.calldualΔ(S₀, K, θ)
end
