include("../src/FinNetValu.jl")
using Test

@testset "utils" begin
    include("./utils/test_utils.jl")
    include("./utils/test_nets.jl")
end
