include("../src/FinNetValu.jl")

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@testset "utils" begin
    include("./utils/test_utils.jl")
end


@testset "pricing" begin
    include("./pricing/test_mc.jl")
    include("./pricing/test_bs.jl")
end

@testset "models" begin
    include("./models/test_rogers.jl")
    include("./models/test_xos.jl")
end
