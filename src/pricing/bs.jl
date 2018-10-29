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
function Aτ(a₀, θ::BlackScholesParams, Z)
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
