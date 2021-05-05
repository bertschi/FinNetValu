using Distributions
using LinearAlgebra

fixdebtequity(solver, net, a) =
    FinNetValu.debtequity(net, FinNetValu.fixvalue(solver, net, a), a)

@testset "rv" begin
    ## Asymmetric example from paper
    L = [0    4.94 2.47 5.59 0 0;
         6    0    0    2    0 0;
         0    13   0    0    0 0;
         0    0    0    0    0 8;
         12   0    0    0    0 0;
         2.79 6.21 0    0    0 0]
    N = size(L, 1)
    e = [1, 1, 11.51, 1.4, 12.5, 2]
    for i ∈ 1:10
        α, β = rand(rng, Uniform(0, 1), 2)
        ## Note: Original model does not allow external debt
        netRV = FinNetValu.RogersVeraartModel(zeros(N), L, α, β)
        netRVOrig = FinNetValu.RVOrigModel(L, α, β)

        x = rand(rng, Uniform(0, 1)) .* e
        rv = fixdebtequity(solver, netRV, x)
        rvo = fixdebtequity(solver, netRVOrig, x)
        rvp = fixdebtequity(FinNetValu.GCVASolver(), netRVOrig, x)
        @test rv ≈ rvo
        @test rv ≈ rvp
    end
end

@testset "rveq" begin
    N = 8
    Mˢ = rand(rng, Uniform(0, 1/N), N, N)
    Mᵈ = rand(rng, Uniform(0, 1/N), N, N)
    for i ∈ 1:N
        Mˢ[i,i] = 0.0
        Mᵈ[i,i] = 0.0
    end
    d = ones(N)
    xos = FinNetValu.XOSModel(Mˢ, Mᵈ, I, d)
    rve = FinNetValu.RVEqModel(Mˢ, Mᵈ, I, d, 1.0, 1.0, 1.0)
    a = d .+ rand(rng, Uniform(-0.1, 0.1), N)
    @test fixdebtequity(solver, xos, a) ≈ fixdebtequity(solver, rve, a)

    ## Next check original and new version
    α, β = rand(rng, Uniform(0, 1), 2)
    rve = FinNetValu.RVEqModel(zeros(N, N), Mᵈ, I, d, α, β, β)
    xe = fixdebtequity(solver, rve, a)
    ## Note: Original model needs special bank to hold all external debt
    L = (Mᵈ .* d)'
    rvo = FinNetValu.RVOrigModel(vcat(zeros(1, N+1),
                                      hcat(d .- FinNetValu.rowsums(L), L)),
                                 α, β)
    xo = fixdebtequity(FinNetValu.GCVASolver(), rvo, vcat(0.1, a))
    @test xe.equity ≈ xo.equity[2:end]
    @test xe.debt ≈ xo.debt[2:end]
end
