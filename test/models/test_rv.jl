using Distributions

# Base.isapprox(x::FinNetValu.ModelState, y::FinNetValu.ModelState) =
#     x.equity ≈ y.equity && x.debt ≈ y.debt

@testset "rv" begin
    ## Asymmetric example from paper
    L = [0    4.94 2.47 5.59 0 0;
         6    0    0    2    0 0;
         0    13   0    0    0 0;
         0    0    0    0    0 8;
         12   0    0    0    0 0;
         2.79 6.21 0    0    0 0]
    N = size(L, 1)
    Lᵉ = zeros(N)
    e = [1, 1, 11.51, 1.4, 12.5, 2]
    for i ∈ 1:10
        α, β = rand(Uniform(0, 1), 2)
        netRV = FinNetValu.RogersVeraartModel(Lᵉ, L, α, β)
        netRVOrig = FinNetValu.RVOrigModel(L, α, β)

        x = rand(Uniform(0, 1)) .* e
        rv = FinNetValu.fixvalue(FinNetValu.PicardIteration(1e-12, 1e-12), netRV, x)
        rvo = FinNetValu.fixvalue(FinNetValu.PicardIteration(1e-12, 1e-12), netRVOrig, x)
        rvp = FinNetValu.fixvalue(FinNetValu.GCVASolver(), netRVOrig, x)
        @test rv ≈ rvo
        @test rv ≈ rvp
    end
end
