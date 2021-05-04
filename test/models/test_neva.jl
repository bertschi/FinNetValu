using LinearAlgebra
using Distributions

Base.isapprox(x::FinNetValu.ModelState, y::FinNetValu.ModelState) =
    x.equity ≈ y.equity && x.debt ≈ y.debt

@testset "neva" begin
    L = [0. 1 2;
         1 0 1;
         2 1 0]
    Lᵉ = [1., 0.2, 2.]
    N = length(Lᵉ)
    
    ## Check model equivalences
    d = FinNetValu.rowsums(L) .+ Lᵉ
    Mᵈ = (L ./ d)'

    @testset "same model I" begin
        netXOS = FinNetValu.XOSModel(zeros(N, N), Mᵈ, I, d)
        netEN = FinNetValu.EisenbergNoeModel(Lᵉ, L)
        netRV11 = FinNetValu.RogersVeraartModel(Lᵉ, L, 1.0, 1.0)
        a = rand(Uniform(-0.1, 0.1), N) .+ Lᵉ
        x = [FinNetValu.fixvalue(solver, net, a)
             for solver ∈ [FinNetValu.PicardIteration(1e-12, 1e-12),
                            FinNetValu.NLSolver(m = 0, xtol = 1e-12)],
                 net ∈ [netXOS, netEN, netRV11] ]
        
        for i ∈ 2:length(x)
            @test x[1] ≈ x[i]
        end
    end

    @testset "same model II" begin
        netF = FinNetValu.FurfineModel(Lᵉ, L, 0.0)
        netRV00 = FinNetValu.RogersVeraartModel(Lᵉ, L, 0.0, 0.0)
        for i ∈ 1:10
            solver = (i > 5) ? FinNetValu.PicardIteration(1e-12, 1e-12) : FinNetValu.NLSolver(m = 0, xtol = 1e-12)
            a = rand(Uniform(-0.1, 0.1), N) .+ Lᵉ
            @test FinNetValu.fixvalue(solver, netF, a) ≈ FinNetValu.fixvalue(solver, netRV00, a)
        end
    end
end
