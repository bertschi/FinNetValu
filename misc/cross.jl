## Investigate impact of cross-holding on equity and debt returns

module Cross

using FinNetValu
using Parameters
using LinearAlgebra
using Distributions
using ForwardDiff

θ = BlackScholesParams(0., 2.5, 0.5)
K = 2.

function callΔ(S₀, K, θ)
    @unpack r, τ, σ = θ
    FinNetValu.Φ.(FinNetValu.d₊.(S₀, K, r, τ, σ))
end

function risk(S₀)
    s = FinNetValu.callprice(S₀, K, θ)
    r = discount(θ) * K - FinNetValu.putprice(S₀, K, θ)
    
    Δₛ = callΔ(S₀, K, θ)
    Δᵣ = - (Δₛ - 1)

    r / s * Δₛ / Δᵣ
end

S₀ = 1.0:0.05:2.5
x = risk.(S₀)

ρ = 0.0
L = cholesky([1 ρ; ρ 1]).L
θ = BlackScholesParams(0., 1., 0.25, L)

function calcΔ(net, a₀, θ, Z)
    A = Aτ(a₀, θ, Z)
    dVdA = fixjacobian(net, A)
    dAdg = ForwardDiff.jacobian(a -> Aτ(a, θ, Z), A)
    discount(θ) .* dVdA * dAdg
end

function risk2(a₀)
    net = XOSModel(zeros(2, 2), [0 0.6; 0.6 0], I, fill(K, 2))
    x = expectation(z -> discount(θ) .* fixvalue(net, Aτ(a₀, θ, z)),
                    MonteCarloSampler(MvNormal(2, 1.0)),
                    25000)
    s = equityview(net, x)
    r = debtview(net, x)
    Δ = expectation(z -> calcΔ(net, a₀, θ, z),
                    MonteCarloSampler(MvNormal(2, 1.0)),
                    25000)
    Δₛ = equityview(net, Δ)
    Δᵣ = debtview(net, Δ)
    denom = Δₛ[2,1] * Δᵣ[1,2] - Δₛ[2,2] * Δᵣ[1,1]
    retₛ = (Δₛ[1,1] * Δᵣ[1,2] - Δₛ[1,2] * Δᵣ[1,1]) / denom * s[2] / s[1]
    retᵣ = (Δₛ[1,2] * Δₛ[2,1] - Δₛ[1,1] * Δₛ[2,2]) / denom * r[1] / s[1]
    retₛ, retᵣ
end

end # module
