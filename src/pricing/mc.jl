#############################
# Basic Monte-Carlo helpers #
#############################

using Distributions

"""
Base type for Monte-Carlo samplers.
"""
abstract type AbstractSampler end

"""
Samples holds `val` with `weight`.
"""
struct Sample{T}
    val::T
    weight::Float64

    function Sample(x::T) where T
        new{T}(x, 1.0)
    end
end

Chain{T} = Vector{Sample{T}} where T

"""
    samplesampler(sampler, N)

Draw `N` samples using the supplied `sampler`.
"""
function sample end

"""
    expectation(f, sampler, N)

Compute expectation of `f` using `N` samples from `sampler`. Faster
than mapping over samples as no samples are retained.
"""
function expectation end

"""
    expectation(f, samples)

Convenience method to compute expectation of `f` from supplied
samples.
"""
function expectation(f::Function, samples::Chain)
    N = length(samples)
    @assert N > 0
    μ = f(samples[1].val) .* samples[1].weight
    for i in 2:N
        μ += f(samples[i].val) .* samples[i].weight
    end
    μ ./ N
end

##############################
# Simple Monte-Carlo sampler #
##############################

struct MonteCarloSampler{T}
    p::T

    function MonteCarloSampler(p::Distribution)
        new{typeof(p)}(p)
    end
end

sample(s::MonteCarloSampler, N::Integer) = [Sample(rand(s.p)) for _ in 1:N]

function expectation(f::Function, s::MonteCarloSampler, N::Integer)
    @assert N > 0
    μ = f(rand(s.p))
    for _ in 2:N
        μ += f(rand(s.p))
    end
    μ ./ N
end

############################
# TODO: Importance sampler #
############################
