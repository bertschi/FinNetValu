########################################
# Reproduce some of the Greeks figures #
########################################

using FinNetValu
using ForwardDiff
using Distributions
using Gadfly
using LinearAlgebra
using DataFrames
using SparseArrays

## Recompute Delta for wᵈ = 0.6 and <k> = 3

function delta(net, a₀, θ, Z)
    A = Aτ(a₀, θ, Z)
    x = fixvalue(net, A;
                 method = :anderson, m = 5)
    dVdA = fixjacobian(net, A, x)
    dAda₀ = ForwardDiff.jacobian(a -> Aτ(a, θ, Z), a₀)
    dVda₀ = discount(θ) * dVdA * dAda₀

    N = numfirms(net)
    deltaEq = sum(equityview(net, dVda₀)) / N
    deltaDb = sum(debtview(net, dVda₀)) / N
    [deltaEq, deltaDb, deltaEq + deltaDb]
end

function randnet(N, k, wᵈ)
    p = k / (N - 1)
    XOSModel(sparse(zeros(N, N)),
             sparse(rescale(erdosrenyi(N, p), wᵈ)),
             I, ones(N))
end

function main(N, numsamples)
    θ = BlackScholesParams(0.0, 1.0, 0.4)
    k = 3.0
    wᵈ = 0.6

    nextnet = calm(() -> randnet(N, k, wᵈ), 100)
    
    a₀ = range(0, length = 31, stop = 2.0)
    Δ = collect(expectation(z -> delta(nextnet(), a * ones(N), θ, z),
                            MonteCarloSampler(MvNormal(N, 1.0)),
                            numsamples)
                for a in a₀)
    a₀, Δ
end

# @time a₀, Δ = main(60, 1000)
# plot(stack(DataFrame(a₀= a₀,
#                      Equity = map(x -> x[1], Δ),
#                      Debt = map(x -> x[2], Δ),
#                      Value = map(x -> x[3], Δ)),
#            [:Equity, :Debt, :Value]),
#      x = :a₀, y = :value, color = :variable,
#      Geom.line)

"""
Compute Greeks for given `net` at initial asset values `a₀`
and random variate `Z` for Black-Scholes world `θ`.  
"""
function calc_greeks(net, a₀, θ, Z)
    @assert length(a₀) == numfirms(net)

    A = Aτ(a₀, θ, Z)
    x = fixvalue(net, A)
    dVdA = fixjacobian(net, A, x)

    r, tau, sig = 1:3
    a = 4:(numfirms(net)+3)
    dAdg = ForwardDiff.jacobian(g -> Aτ(g[a],
                                        BlackScholesParams(g[r], g[tau], g[sig]),
                                        Z),
                                vcat(θ.r, θ.τ, θ.σ, a₀))
    dVdg = discount(θ) * dVdA * dAdg
    dVdg[:, tau] .+= - θ.r .* discount(θ) .* x
    dVdg[:, tau] .*= - 1
    dVdg[:, r]   .+= - θ.τ .* discount(θ) .* x

    dVdg
end

function greeks(N, θ, k, wᵈ, numsamples)
    nextnet = calm(() -> randnet(N, k, wᵈ), 100)
    a₀ = range(0, length = 31, stop = 2.0)
    g = collect(expectation(z -> calc_greeks(nextnet(), a * ones(N), θ, z),
                            MonteCarloSampler(MvNormal(N, 1.0)),
                            numsamples)
                for a in a₀)
    net = nextnet()
    a₀, map(dVdg -> sum(equityview(net, dVdg); dims = 1), g), map(dVdg -> sum(debtview(net, dVdg); dims = 1), g)
end
