include("../src/FinNetValu.jl")

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@testset "utils" begin
    include("./utils/test_utils.jl")
end
