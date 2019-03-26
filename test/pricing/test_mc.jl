using Distributions

@testset "statistics" begin
    chain = [FinNetValu.Sample(1), FinNetValu.Sample(2), FinNetValu.Sample(3)]
    @test FinNetValu.expectation(identity, chain) == 2
    @test FinNetValu.expectation(x -> x^2, chain) == (1 + 4 + 9) / 3
end

@testset "sampling" begin
    ## Note: Test should pass with 99% probability
    N = 1000
    mu = 1.0
    sigma = 1.0
    z99 = 2.576
    @test (mu -  z99 * sigma / sqrt(N)) < FinNetValu.expectation(x -> x, FinNetValu.MonteCarloSampler(Distributions.Normal(mu, sigma)), N) < (mu + z99 * sigma / sqrt(N))
end
