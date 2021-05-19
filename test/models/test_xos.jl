using ForwardDiff
using LinearAlgebra
using SparseArrays

@testset "xos general" begin
    A1 = [0 0; 0 0]
    A2 = [1 0; 0 1]
    A3 = [1 2 3; 5 4 2; 7 2 5]
    A4 = [.1 .2 .3; .3 .4 .2; .1 .2 .5]
    A5 = [.5 .2 .3; .3 .4 .3; .2 .2 .3]
    A6 = [.5 .9 .3; .3 .4 .3; .2 .2 .3]
    A7 = [0 1 1; 1 0 1; 1 1 0]
    spA1 = spzeros(4, 2)
    spA1[1,1] = 1; spA1[1,2] = 2; spA1[2,1] = 2; spA1[2,2] = 1
    spA2 = spzeros(2, 4)
    spA2[1,1] = 3; spA2[1,2] = 2; spA2[2,1] = 2; spA2[2,2] = 1
    spA3 = spzeros(2, 4)
    spA3[1,1] = .3; spA3[1,2] = .2; spA3[2,1] = .2; spA3[2,2] = .1
    spA4 = spzeros(4, 2)
    spA4[1,1] = .3; spA4[1,2] = .2; spA4[2,1] = .2; spA4[2,2] = .1; spA4[3,1] = 0.5; spA4[3,2] = .7
    spFull_3 = spzeros(3,3)
    spFull_3[1,2] = 1; spFull_3[1,3] = 1; spFull_3[2,1] = 1; spFull_3[2,3] = 1; spFull_3[3,1] = 1; spFull_3[3,2] = 1;

    spA5 = spzeros(3,3)
    spA5[1,2] = 1.0; spA5[2,1] = 1.0
    a1 = [1, 2, 3, 2, 1]
    a2 = [1, 1, 1, 2, 3, 3, 2, 4, 4, 3]
    k1 = [2/5, 2/5, 1/5]
    k2 = [3/10, 2/10, 3/10, 2/10]

    @testset "rowsums" begin
        @test FinNetValu.rowsums(A1) == [0, 0]
        @test FinNetValu.rowsums(A2) == [1, 1]
        @test FinNetValu.rowsums(A3) == [6, 11, 14]
        @test FinNetValu.rowsums(spA1) == [3, 3, 0, 0]
        @test FinNetValu.rowsums(spA2) == [5, 3]
    end

    @testset "isleft_substochastic" begin
        @test FinNetValu.isleft_substochastic(A1) == true
        @test FinNetValu.isleft_substochastic(A2) == true
        @test FinNetValu.isleft_substochastic(A3) == false
        @test FinNetValu.isleft_substochastic(A4) == true
        @test FinNetValu.isleft_substochastic(A5) == true
        @test FinNetValu.isleft_substochastic(A6) == false
        @test FinNetValu.isleft_substochastic(spA1) == false
        @test FinNetValu.isleft_substochastic(spA2) == false
        @test FinNetValu.isleft_substochastic(spA3) == true
        @test FinNetValu.isleft_substochastic(spA4) == true
    end

    @testset "erdosrenyi" begin
        @test_throws ArgumentError FinNetValu.erdosrenyi(0, -1)
        @test_throws ArgumentError FinNetValu.erdosrenyi(0, 1)
        @test FinNetValu.erdosrenyi(1, 0) == spzeros(1,1)
        @test FinNetValu.erdosrenyi(2, 0) == spzeros(2,2)
        @test FinNetValu.erdosrenyi(3, 1) == spFull_3
        @test FinNetValu.erdosrenyi(3, 1, false) == spFull_3
    end

    @testset "rescale" begin
        @test FinNetValu.rescale(A5, 0.3) == FinNetValu.rescale(A5, [0.3, 0.3, 0.3])
        @test FinNetValu.rescale(A2, 1) == A2
        @test FinNetValu.rescale(A2, 0) == 0*A2
        @test_throws MethodError FinNetValu.rescale(A2, 0.5) == 0.5*A2
    end

    @testset "barabasialbert" begin
        @test FinNetValu.barabasialbert(3, 2) != false
        @test_throws ArgumentError FinNetValu.barabasialbert(2, 3)
        @test_throws ArgumentError FinNetValu.barabasialbert(2, 1)
        @test FinNetValu.barabasialbert(3, 2) == spFull_3
    end

    @testset "initm0graph" begin
        @test FinNetValu.initm0graph(3, 2) != false
        @test FinNetValu.initm0graph(3, 2) == spA5
        @test_throws ArgumentError FinNetValu.initm0graph(3, 1)
        @test_throws ArgumentError FinNetValu.initm0graph(3, 3)
    end

    @testset "attachmentweights" begin
        @test FinNetValu.attachmentweights(a1) != false
        @test FinNetValu.attachmentweights(a1) == k1
        @test FinNetValu.attachmentweights(a2) == k2
    end
end


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
