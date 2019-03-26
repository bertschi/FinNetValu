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
    x = FinNetValu.fixvalue(net, a)
    N = 4
    @test FinNetValu.numfirms(net) == N
    @test length(x) == 2*N
    @test all(FinNetValu.solvent(net, x))
    @test FinNetValu.debtview(net, x) == net.d
    @test x == vcat(FinNetValu.equityview(net, x),
                    FinNetValu.debtview(net, x))

    @test size(FinNetValu.fixjacobian(net, a)) == (2*N, N)
    @test ForwardDiff.jacobian(a -> FinNetValu.fixvalue(net, a), a) ≈ FinNetValu.fixjacobian(net, a)
end
