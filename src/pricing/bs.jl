#################################
# Black-Scholes pricing helpers #
#################################

using Parameters
using LinearAlgebra

""" 
Black-Scholes parameters `r`, `τ` and `σ`. In the multivariate
case, Brownian motions can be correlated with `Lᵨ` specifying the
Cholesky factor of their correlation matrix.
"""

struct BlackScholesParams{T1,T2,T3}
    r::T1
    τ::T1
    σ::T2
    Lᵨ::T3
end

BlackScholesParams(r, τ, σ) = BlackScholesParams(r, τ, σ, I)

"""
    Aτ(a₀, θ, Z)

Compute log normal asset prices at maturity `τ` by transforming
standard normal variates `Z` assuming an initial price `a₀` and
Black-Scholes parameters `θ`.
"""
function Aτ(a₀, θ::BlackScholesParams, Z::AbstractVector)
    @unpack r, τ, σ, Lᵨ = θ
    LZ = Lᵨ * Z
    @. a₀ * exp( (r - 0.5 * σ^2) * τ + sqrt(τ) * σ * LZ )
end

"""
    discount(θ)

Compute discount factor ``e^{- r \\tau}`` from Black-Scholes
parameters `θ`.
"""
function discount(θ::BlackScholesParams)
    @unpack r, τ = θ
    @. exp( - r * τ )
end

"""
    d₊(S₀, K, r, τ, σ)

Computes the d₊ value from the Black-Scholes formula.
"""
function d₊(S₀, K, r, τ, σ)
    @. (log(S₀) - log(K) + (r + σ^2 / 2) * τ) / (σ * √(τ))
end

"""
    d₊(S₀, K, r, τ, σ)

Computes the d₋ value from the Black-Scholes formula.
"""
function d₋(S₀, K, r, τ, σ)
    @. (log(S₀) - log(K) + (r - σ^2 / 2) * τ) / (σ * √(τ))
end

function Φ(x)
    cdf(Normal(0, 1), x)
end

"""
    callprice(S₀, K, θ)

Analytic Black-Scholes price of a call option with strike `K`. `S₀`
denotes the current price of the underlying and parameters of the
Black-Scholes world are collected in `θ`.
"""
function callprice(S₀, K, θ::BlackScholesParams{T1,T2,UniformScaling{T3}}) where {T1,T2,T3}
    @unpack r, τ, σ = θ
    S₀ .* Φ.(d₊.(S₀, K, r, τ, σ)) .- discount(θ) .* K .* Φ.(d₋.(S₀, K, r, τ, σ))
end

function calldualΔ(S₀, K, θ::BlackScholesParams{T1,T2,UniformScaling{T3}}) where {T1,T2,T3}
    @unpack r, τ, σ = θ
    .- discount(θ) .* Φ.(d₋.(S₀, K, r, τ, σ))
end

"""
    putprice(S₀, K, θ)

Analytic Black-Scholes price of a put option with strike `K`. `S₀`
denotes the current price of the underlying and parameters of the
Black-Scholes world are collected in `θ`.  
"""
function putprice(S₀, K, θ::BlackScholesParams{T1,T2,UniformScaling{T3}}) where {T1,T2,T3}
    @unpack r, τ, σ = θ
    discount(θ) .* K .* Φ.(.- d₋.(S₀, K, r, τ, σ)) .- S₀ .* Φ.(.- d₊.(S₀, K, r, τ, σ))
end

function putdualΔ(S₀, K, θ::BlackScholesParams{T1,T2,UniformScaling{T3}}) where {T1,T2,T3}
    @unpack r, τ, σ = θ
    discount(θ) .* Φ.(.- d₋.(S₀, K, r, τ, σ))
end
