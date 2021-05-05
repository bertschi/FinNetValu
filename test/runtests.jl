include("../src/FinNetValu.jl")

using Random
using Test

rng = MersenneTwister(1234)
solver = FinNetValu.PicardIteration(1e-12, 1e-12)

@testset "utils" begin
    include("./utils/test_utils.jl")
    include("./utils/test_nets.jl")
end

@testset "pricing" begin
    include("./pricing/test_mc.jl")
    include("./pricing/test_bs.jl")
end

@testset "models" begin
    include("./models/test_xos.jl")
    include("./models/test_neva.jl")
    include("./models/test_rv.jl")
end
