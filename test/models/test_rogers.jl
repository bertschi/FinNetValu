@testset "rogers" begin

    name = "InitTest"
    α = 1
    β = 1
    L = [0.0 0.1 0.7; 2.0 0.0 10.5; 0.0 8.4 0.0]
    A = [1.0; 8.0; 10.1]
    Lbar = sum(L, dims=[2])
    a1 = [1.0; 1.0; 1.0]
    a2 = [0.0; 0.0; 0.1]
    a3 = [1.0; 2.0; 3.0]


    net = FinNetValu.RogersModel(name, L, α, β)
    @testset "Initializations" begin
        @test net.name == name
        @test net.α == α
        @test net.β == β
        @test net.L == L
    end

    @testset "InterfaceFunctions" begin
        @test FinNetValu.numfirms(net) == 3
        @test all(FinNetValu.valuation(net, zeros(3), a1) .== zeros(3))
        @test all(FinNetValu.init(net,  a1) .== Lbar)
        @test all(FinNetValu.equityview(net, zeros(3)) .== zeros(3))
        @test all(FinNetValu.debtview(net, [1, 2, 3]) .== [1, 2, 3])
    end

    @testset "BasicProperties" begin
        @test all(sum(net.Π, dims=[2]) .∈ [[1.0, 0.0]])
        @test all(FinNetValu.valuation(net, 20 .* ones(3), a3) .≈ [-15.8,4.5,17.3])
        @test all(FinNetValu.valuation(net, ones(3), a3) .≈ [0.16,0.0,0.0])
    end
   
    @testset "balance" begin
        naive = [0.0; 0.0; 0.0]
        for i in 1:3
            for j in 1:3
                naive[i] += Lbar[j] * net.Π[j,i]
            end
        end
        tmp = transpose(Lbar) * net.Π
        @test all(naive .== transpose(tmp))
        @test all(naive .== transpose(net.Π) * Lbar)
    end

    @testset "GClearVecAlgo" begin
        @test all(FinNetValu.fixvalue(net, a1) .≈ [0.8 0.8; 12.5 9.5; 8.4 8.4])
        @test all(FinNetValu.fixvalue(net, a2) .≈ [0.8 0.8 0.8; 12.5 8.5 5.625; 8.4 8.4 5.525])
    end
end
