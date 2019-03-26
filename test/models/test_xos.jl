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
    
    x = FinNetValu.fixvalue(net, a)
    @test length(x) == 2*N
    @test all(FinNetValu.solvent(net, x))
    @test FinNetValu.debtview(net, x) == net.d
    @test x == vcat(FinNetValu.equityview(net, x),
                    FinNetValu.debtview(net, x))

    J = FinNetValu.fixjacobian(net, a)
    @test size(J) == (2*N, N)
    @test isapprox(J, ForwardDiff.jacobian(a -> FinNetValu.fixvalue(net, a; xtol = 1e-8), a); rtol = 1e-4)
    @test J == vcat(FinNetValu.equityview(net, J),
                    FinNetValu.debtview(net, J))
end
