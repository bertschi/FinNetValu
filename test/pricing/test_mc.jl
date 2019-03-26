using Distributions

@testset "statistics" begin
    chain = [FinNetValu.Sample(1), FinNetValu.Sample(2), FinNetValu.Sample(3)]
    @test FinNetValu.expectation(identity, chain) == 2
    @test FinNetValu.expectation(x -> x^2, chain) == (1 + 4 + 9) / 3
end

function hit99(estimate, mu, sigma, N)
    z99 = 2.576
    (mu -  z99 * sigma / sqrt(N)) < estimate < (mu + z99 * sigma / sqrt(N))
end

@testset "sampling" begin
    ## Note: Test should pass with 99% probability
    N = 1000
    mu = 1.0
    sigma = 1.0
    @test hit99(FinNetValu.expectation(x -> x, FinNetValu.MonteCarloSampler(Distributions.Normal(mu, sigma)), N), mu, sigma, N)
    @test hit99(FinNetValu.expectation(x -> x, FinNetValu.sample(FinNetValu.MonteCarloSampler(Distributions.Normal(mu, sigma)), N)), mu, sigma, N)
end
