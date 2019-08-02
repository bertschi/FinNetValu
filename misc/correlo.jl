## Simulation studies for equity correlations in financial networks

using DataFrames
using DataFramesMeta
using Gadfly
import NLsolve
import LinearAlgebra
import ForwardDiff
using FinNetValu

## First we define some helper functions

"""
    calcΔ(net, a₀, θ, Z)

Uses pathwise differentiation to compute network Δ of `net` at Aτ(`a₀`, `θ`, `Z`).
"""
function calcΔ(net, a₀, θ, z)
    A = Aτ(a₀, θ, z)
    dVdA = fixjacobian(net, A)
    dAdg = ForwardDiff.jacobian(a -> Aτ(a, θ, z), A)
    discount(θ) .* dVdA * dAdg
end

"""
    netstat(net, a₀, θ, stat_fun; samples)

Computes risk-neutral expectations over `stat_fun` evaluated on
network `net` with parameters `a₀` and `θ` using Monte-Carlo
`samples`.
"""
function netstat(net, a₀, θ, stat_fun; samples = 25000)
    ## Use importance sampling focusing on default boundary
    function val(net, a)
        x = fixvalue(net, a)
        equityview(net, x) .+ debtview(net, x)
    end
    deff = NLsolve.nlsolve(a -> val(net, a) .- net.d, net.d).zero ## Find point where all firms just default
    μ = NLsolve.nlsolve(z -> Aτ(a₀, θ, z) .- deff, zeros(numfirms(net))).zero ## Find corresponding unconstrained asset values
    # q = MvNormal(μ, 3.0) ## Importance density ... does not seem to work very well?!?
    q = MvNormal(numfirms(net), 1.)
    w(z) = exp(logpdf(MvNormal(numfirms(net), 1.), z) - logpdf(q, z))
    function tmp()
        z = rand(q)
        FinNetValu.Sample(z, w(z))
    end
    Z = [tmp() for _ = 1:samples]
    expectation(z -> w(z) .* stat_fun(net, a₀, θ, z), Z), Z
    # expectation(z -> stat_fun(net, a₀, θ, z),
    #             MonteCarloSampler(MvNormal(numfirms(net), 1.)),
    #             samples)
    
end

valΔ(net, a₀, θ, z) = [discount(θ) .* fixvalue(net, Aτ(a₀, θ, z)),
                       calcΔ(net, a₀, θ, z)]

## Run data grid and compute correlations along the way
function calcΣ(net, x, Δ, a₀, θ)
    s = equityview(net, x)
    Δˢ = equityview(net, Δ)
    σ = θ.σ .* ones(numfirms(net))
    a₀ = a₀ .* ones(numfirms(net))
    L = cholesky(θ.Lᵨ).L
    Lˢ = diagm(0 => (1 ./ s)) * Δˢ * diagm(0 => a₀) * diagm(0 => σ) * L
    Lˢ * Lˢ'
end

function cordata()
end
