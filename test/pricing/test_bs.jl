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
