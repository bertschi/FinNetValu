using ForwardDiff
using LinearAlgebra

@testset "xos" begin
    net = FinNetValu.XOSModel([0.0 0.2 0.3 0.1;
                               0.2 0.0 0.2 0.1;
                               0.1 0.1 0.0 0.3;
                               0.1 0.1 0.1 0.0],
                              [0.0 0.0 0.1 0.0;
                               0.0 0.0 0.0 0.1;
                               0.1 0.0 0.0 0.1;
                               0.0 0.1 0.0 0.0],
                              LinearAlgebra.I,
                              [0.8, 0.8, 0.8, 0.8])
    a = [2.0, 0.5, 0.6, 0.6]
    N = 4
    @test FinNetValu.numfirms(net) == N

    @test FinNetValu.valuation(net, zeros(2*N), a) == [1.2, 0., 0., 0., 0.8, 0.5, 0.6, 0.6]

    fix(net, a) = FinNetValu.fixvalue(FinNetValu.PicardIteration(1e-8, 1e-8), net, a;
                                      finalize = false)
    x = fix(net, a)
    y = FinNetValu.finalizestate(net, x, a)
    @test length(x) == 2*N
    @test all(FinNetValu.solvent(net, x))
    @test all(FinNetValu.solvent(net, y))
    @test y.debt == net.d
    @test x == vcat(y.equity, y.debt)

    @test x ≈ FinNetValu.fixvalue(FinNetValu.NLSolver(m = 0), net, a;
                                  finalize = false)
    
    J = FinNetValu.fixjacobian(net, x, a)
    @test size(J) == (2*N, N)
    @test isapprox(J, ForwardDiff.jacobian(a -> fix(net, a), a); rtol = 1e-4)
    @test J == vcat(FinNetValu.equityview(net, J),
                    FinNetValu.debtview(net, J))
end
