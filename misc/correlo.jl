## Simulation studies for equity correlations in financial networks

using DataFrames
using DataFramesMeta
using Distributions
## using Gadfly
import NLsolve
using LinearAlgebra
import ForwardDiff
import CSV

import Pkg
Pkg.activate("/home/bertschi/GitRepos/FinNetValu")
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

function neff(w)
    sum(w)^2 / sum(x -> x^2, w)
end

function expectationIS(f, p, q, N)
    w(z) = exp(logpdf(p, z) - logpdf(q, z))
    function tmp()
        z = rand(q)
        FinNetValu.Sample(z, w(z))
    end
    Z = [tmp() for _ = 1:N]
    Nₑ = neff(map(x -> x.weight, Z))
    expectation(f, Z), Nₑ
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
    expectationIS(z -> stat_fun(net, a₀, θ, z),
                  MvNormal(numfirms(net), 1),
                  MvNormal(0.6 .* μ, max.(2, μ)), ## Importance density
                  samples)
end

function valΔ(net, a₀, θ, z)
    x = fixvalue(net, Aτ(a₀, θ, z))
    [discount(θ) .* x, calcΔ(net, a₀, θ, z), solvent(net, x)]
end

## Run data grid and compute correlations along the way
function calcΣ(net, x, Δ, a₀, θ)
    s = equityview(net, x)
    Δˢ = equityview(net, Δ)
    σ = θ.σ .* ones(numfirms(net))
    a₀ = a₀ .* ones(numfirms(net))
    Lˢ = LinearAlgebra.diagm(0 => (1 ./ s)) * Δˢ * LinearAlgebra.diagm(0 => a₀) * LinearAlgebra.diagm(0 => σ) * θ.Lᵨ
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
                [DataFrame(a0 = exp.(-3:0.1:1)),
                 ## DataFrame(a0 = exp.(-3:0.05:1)),
                 DataFrame(md12 = 0:0.2:0.8),
                 DataFrame(md21 = 0:0.2:0.8),
                 DataFrame(sigma = [0.2, 0.4]),
                 DataFrame(rho = -0.4:0.4:0.8)])
    function rhoS(idx, a₀, md12, md21, σ, ρ)
        @show idx, nrow(df)
        @show a₀, md12, md21, σ, ρ
        a₀ = a₀ .* ones(2)
        θ = BlackScholesParams(0, 1, [σ, σ], LinearAlgebra.cholesky([1 ρ; ρ 1]).L)
        ms12 = ms21 = 0.1
        net = XOSModel([0 ms12; ms21 0], [0 md12; md21 0], LinearAlgebra.I, [1.0, 1.0])
        @time tmp, Nₑ = netstat(net, a₀, θ, valΔ; samples = 125000) ## samples = 250000)
        x, Δ, ξ = tmp
        Σ = calcΣ(net, x, Δ, a₀, θ)
        (ρ = Σ[1,2] / √(Σ[1,1] * Σ[2,2]), ξ = ξ, Nₑ = Nₑ,
         s = equityview(net, x),
         r = debtview(net, x))
    end
    @linq df |>
        transform(tmp = Distributed.pmap(rhoS, 1:nrow(df), :a0, :md12, :md21, :sigma, :rho)) |>
        transform(rhoS = map(x -> x.ρ, :tmp),
                  Neff = map(x -> x.Nₑ, :tmp),
                  pd1 = map(x -> 1 - x.ξ[1], :tmp),
                  pd2 = map(x -> 1 - x.ξ[2], :tmp),
                  s1 = map(x -> x.s[1], :tmp),
                  s2 = map(x -> x.s[2], :tmp),
                  r1 = map(x -> x.r[1], :tmp),
                  r2 = map(x -> x.r[2], :tmp))
end

function cordata2()
    df = reduce(DataFrames.crossjoin,
                [DataFrame(a1 = exp.(-3:0.1:1)),
                 DataFrame(a2 = exp.(-3:0.1:1)),
                 DataFrame(md12 = 0:0.2:0.8),
                 DataFrame(md21 = 0:0.2:0.8),
                 DataFrame(sigma = 0.4),
                 DataFrame(rho = 0.0)])
    function rhoS(idx, a1, a2, md12, md21, σ, ρ)
        @show idx, nrow(df)
        @show a1, a2, md12, md21, σ, ρ
        a₀ = [a1, a2]
        θ = BlackScholesParams(0, 1, [σ, σ], LinearAlgebra.cholesky([1 ρ; ρ 1]).L)
        net = XOSModel(zeros(2, 2), [0 md12; md21 0], LinearAlgebra.I, [1.0, 1.0])
        tmp, Nₑ = netstat(net, a₀, θ, valΔ; samples = 250000)
        x, Δ, ξ = tmp
        Σ = calcΣ(net, x, Δ, a₀, θ)
        (ρ = Σ[1,2] / √(Σ[1,1] * Σ[2,2]), ξ = ξ, Nₑ = Nₑ)
    end
    @linq df |>
        transform(tmp = Distributed.pmap(rhoS, 1:nrow(df), :a1, :a2, :md12, :md21, :sigma, :rho)) |>
        transform(rhoS = map(x -> x.ρ, :tmp),
                  Neff = map(x -> x.Nₑ, :tmp),
                  pd1 = map(x -> 1 - x.ξ[1], :tmp),
                  pd2 = map(x -> 1 - x.ξ[2], :tmp))
end

## Run it
## cordata() |> CSV.write("/tmp/cordata.csv")
## cordata2() |> CSV.write("/tmp/cordata2.csv")
