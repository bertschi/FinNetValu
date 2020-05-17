@testset "rogers" begin

    name = "InitTest"
    α = 1
    β = 1
    L = [0.0 0.1 0.7;
         2.0 0.0 10.5;
         0.0 8.4 0.0]
    A = [1.0, 8.0, 10.1]

    net = FinNetValu.RogersModel(name, L, A, α, β)
    @testset "Initializations" begin
        @test net.name == name
        @test net.α == α
        @test net.β == β
        @test net.L == L
        @test net.A == A
    end
end
