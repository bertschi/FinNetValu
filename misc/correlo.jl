## Simulation studies for equity correlations in financial networks

using DataFrames
using DataFramesMeta
## using Gadfly
import NLsolve
using LinearAlgebra
import ForwardDiff
using Distributions
using FinNetValu
using ProgressMeter
using Distributed

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
    # deff = NLsolve.nlsolve(a -> val(net, a) .- net.d, net.d).zero ## Find point where all firms just default
    # μ = NLsolve.nlsolve(z -> Aτ(a₀, θ, z) .- deff, zeros(numfirms(net))).zero ## Find corresponding unconstrained asset values
    # # q = MvNormal(μ, 3.0) ## Importance density ... does not seem to work very well?!?
    # q = MvNormal(numfirms(net), 1.)
    # w(z) = exp(logpdf(MvNormal(numfirms(net), 1.), z) - logpdf(q, z))
    # function tmp()
    #     z = rand(q)
    #     FinNetValu.Sample(z, w(z))
    # end
    # Z = [tmp() for _ = 1:samples]
    # expectation(z -> w(z) .* stat_fun(net, a₀, θ, z), Z), Z
    expectation(z -> stat_fun(net, a₀, θ, z),
                MonteCarloSampler(MvNormal(numfirms(net), 1.)),
                samples)
    
end

valΔ(net, a₀, θ, z) = [discount(θ) .* fixvalue(net, Aτ(a₀, θ, z)),
                       calcΔ(net, a₀, θ, z)]

## Run data grid and compute correlations along the way
function calcΣ(net, x, Δ, a₀, θ)
    s = equityview(net, x)
    Δˢ = equityview(net, Δ)
    σ = θ.σ .* ones(numfirms(net))
    a₀ = a₀ .* ones(numfirms(net))
    Lˢ = diagm(0 => (1 ./ s)) * Δˢ * diagm(0 => a₀) * diagm(0 => σ) * θ.Lᵨ
    Lˢ * Lˢ'
end

@everywhere function compute(a₀, σ₁, σ₂, md12, md21, ms12, ms21, ρ)
    net = XOSModel([0 ms12; ms21 0], [0 md12; md21 0], I, [1.0, 1.0])
    θ = BlackScholesParams(0.02, 1.5, [σ₁, σ₂], cholesky([1 ρ; ρ 1]).L)
    x, Δ = netstat(net, a₀, θ, valΔ)
    Σ = calcΣ(net, x, Δ, a₀, θ)
    Σ[1,2] / √(Σ[1,1] * Σ[2,2])
end

function cordata()
    df = reduce(DataFrames.crossjoin,
                [DataFrame(a = 10 .^ LinRange(-3, 1, 5)),
                 DataFrame(sigma1 = [0.4, 0.8]),
                 DataFrame(sigma2 = [0.4, 0.8]),
                 DataFrame(md12 = 0:0.4:0.8),
                 DataFrame(md21 = 0:0.4:0.8),
                 DataFrame(ms12 = 0),
                 DataFrame(ms21 = 0),
                 DataFrame(rho = -0.4:0.4:0.8)])
    
    # @byrow! df begin
    #     @newcol cor::Vector{Float64}
    #     :cor = compute(:a, :sigma1, :sigma2, :md12, :md21, :ms12, :ms21, :rho)
    #     next!(p)
    # end
    # p = Progress(nrow(df))
    # cor = progress_pmap(x -> compute(x...), zip(df.a, df.sigma1, df.sigma2, df.md12, df.md21, df.ms12, df.ms21, df.rho);
    #                     progress = p)
    cor = @with df pmap(compute, :a, :sigma1, :sigma2, :md12, :md21, :ms12, :ms21, :rho)
    @transform(df, cor = cor)
end
