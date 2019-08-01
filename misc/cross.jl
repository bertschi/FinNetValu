## Investigate impact of cross-holding on equity and debt returns

module Cross

using FinNetValu
using Parameters
using LinearAlgebra
using Distributions
using ForwardDiff
using LinearAlgebra
using DataFrames
using DataFramesMeta
using CSV
using NLsolve

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

function ret(net, θ, a₀)
    @assert numfirms(net) == 2
    ## Control variate x to reduce variance
    val(a, x) = x .+ expectation(z -> discount(θ) .* fixvalue(net, Aτ(a, θ, z)) .- x,
                                 MonteCarloSampler(MvNormal(numfirms(net), 1.0)),
                                 25000)
    v₀ = val(a₀, discount(θ) .* fixvalue(net, Aτ(a₀, θ, zero(a₀))))
    ret = []
    for α = 0:0.1:(2 * pi)
        a = a₀ .+ 0.05 .* a₀ .* [sin(α), cos(α)]
        push!(ret, (val(a, v₀) .- v₀) ./ v₀)
    end
    ret
end

function demo()
    net = XOSModel(zeros(2, 2), [0 0.6; 0.6 0], I, fill(K, 2))
    θ = BlackScholesParams(0., 1., 0.25)
    a₀ = [0.6, 0.9, 1.2]
    df = DataFrame(a = zeros(0),
                   s1 = zeros(0), s2 = zeros(0),
                   r1 = zeros(0), r2 = zeros(0))
    for a in a₀
        for x in ret(net, θ, fill(a, 2))
            push!(df, vcat(a, x))
        end
    end
    df
end

# @time df = demo()
# df |> CSV.write("/tmp/ret.csv")

## Create larger data grid
function cordata()
    df = reduce(DataFrames.crossjoin,
                [DataFrame(a0 = 10 .^ (-5:0.2:1)),
                 DataFrame(r = 0.0),
                 DataFrame(sigma1 = 0.5),
                 DataFrame(sigma2 = 0.5),
                 DataFrame(wd = 0.6),
                 DataFrame(rho = -0.95)])
    function netcor(a₀, r, σ, wᵈ, ρ)
        L = cholesky([1 ρ; ρ 1]).L
        θ = BlackScholesParams(r, 1.0, σ, L)
        d = fill(1.0, 2)
        net = XOSModel(zeros(2, 2), [0 wᵈ; wᵈ 0], I, d)
        ## Use importance sampling which focuses on default boundary
        μ = nlsolve(z -> Aτ(a₀, θ, z) .- (1 - wᵈ) .* d,
                    [0.0, 0.0]).zero
        q = MvNormal(μ, 3.0)
        w(z) = exp(logpdf(MvNormal(numfirms(net), 1.), z) - logpdf(q, z))
        x, Δ = expectation(z -> [w(z) .* discount(θ) .* fixvalue(net, Aτ(a₀, θ, z)),
                                 w(z) .* calcΔ(net, a₀, θ, z)],
                           MonteCarloSampler(q),
                           25000)
        ## Return L factor, s and Δ separately
        s = equityview(net, x)
        Lˢ = diagm(0 => (1 ./ s)) * equityview(net, Δ) * diagm(0 => a₀) * diagm(0 => σ) * L
        Lˢ, s, equityview(net, Δ)
    end
    idx = 0
    N = size(df, 1)
    @byrow! df begin
        idx += 1
        @show idx / N
        @newcol sigmaS1::Vector{Float64}
        @newcol sigmaS2::Vector{Float64}
        @newcol cor::Vector{Float64}
        @newcol corfa::Vector{Float64}
        @newcol corfb::Vector{Float64}
        Lˢ, s, Δ = netcor([:a0, :a0], :r, [:sigma1, :sigma2], :wd, :rho)
        Σ = Lˢ * Lˢ'
        :sigmaS1 = √(Σ[1,1])
        :sigmaS2 = √(Σ[2,2])
        :cor = Σ[1,2] / √(Σ[1,1] * Σ[2,2])
        ## Check correlation formulas
        x = Lˢ[1,1] * Lˢ[2,1] + Lˢ[1,2] * Lˢ[2,2]
        :corfa = sign(x) * √(1 / (1 + ( (Lˢ[1,1] * Lˢ[2,2] - Lˢ[1,2] * Lˢ[2,1]) / x )^2))
        y = Δ[1,1] * Δ[2,1] * :sigma1^2 * :a0^2 + (Δ[1,1] * Δ[2,2] + Δ[1,2] * Δ[2,1]) * :a0 * :a0 * :sigma1 * :sigma2 * :rho + Δ[2,2] * Δ[1,2] * :a0^2 * :sigma2^2
        :corfb = sign(y) * √(1 / (1 + ( ((Δ[1,1] * Δ[2,2] - Δ[1,2] * Δ[2,1]) * :a0 * :a0 * :sigma1 * :sigma2 * √(1 - :rho^2)) / y )^2))
        # if x != (y / (s[1] * s[2]))
        #     @show :sigma1, :sigma2
        #     @show x, (y / (s[1] * s[2]))
        #     @show Lˢ[1,1] * Lˢ[2,2] - Lˢ[1,2] * Lˢ[2,1]
        #     @show ((Δ[1,1] * Δ[2,2] - Δ[1,2] * Δ[2,1]) * :a0 * :a0 * :sigma1 * :sigma2 * √(1 - :rho^2)) / (s[1] * s[2])
        #     ## Check all L entries
        #     @show Lˢ[1,1], (Δ[1,1] * :a0 * :sigma1 + Δ[1,2] * :a0 * :sigma2 * :rho) / s[1]
        # end
    end
end

## Check upper limit with equity cross holdings
function upperlimit()
    df = reduce(DataFrames.crossjoin,
                [DataFrame(a0 = 10 .^ (-1:0.2:1)),
                 DataFrame(r = 0.0),
                 DataFrame(sigma1 = 0.4),
                 DataFrame(sigma2 = 0.4),
                 DataFrame(ws12 = [0.2, 0.6]),
                 DataFrame(ws21 = [0.2, 0.6]),
                 DataFrame(rho = [0., 0.4, 0.8])])
    function netcor(a₀, r, σ, ws12, ws21, ρ)
        L = cholesky([1 ρ; ρ 1]).L
        θ = BlackScholesParams(r, 1.0, σ, L)
        d = fill(1.0, 2)
        net = XOSModel([0 ws12; ws21 0], zeros(2, 2), I, d)
        ## Use importance sampling which focuses on default boundary
        μ = nlsolve(z -> Aτ(a₀, θ, z) .- d,
                    [0.0, 0.0]).zero
        q = MvNormal(μ, 3.0)
        w(z) = exp(logpdf(MvNormal(numfirms(net), 1.), z) - logpdf(q, z))
        x, Δ = expectation(z -> [w(z) .* discount(θ) .* fixvalue(net, Aτ(a₀, θ, z)),
                                 w(z) .* calcΔ(net, a₀, θ, z)],
                           MonteCarloSampler(q),
                           25000)
        ## Return L factor, s and Δ separately
        s = equityview(net, x)
        Lˢ = diagm(0 => (1 ./ s)) * equityview(net, Δ) * diagm(0 => a₀) * diagm(0 => σ) * L
        Lˢ, s, equityview(net, Δ)
    end
    idx = 0
    N = size(df, 1)
    @byrow! df begin
        idx += 1
        @show idx / N
        @newcol sigmaS1::Vector{Float64}
        @newcol sigmaS2::Vector{Float64}
        @newcol cor::Vector{Float64}
        @newcol corfa::Vector{Float64}
        @newcol corfb::Vector{Float64}
        @newcol corlim::Vector{Float64}
        Lˢ, s, Δ = netcor([:a0, :a0], :r, [:sigma1, :sigma2], :ws12, :ws21, :rho)
        Σ = Lˢ * Lˢ'
        :sigmaS1 = √(Σ[1,1])
        :sigmaS2 = √(Σ[2,2])
        :cor = Σ[1,2] / √(Σ[1,1] * Σ[2,2])
        ## Check correlation formulas
        x = Lˢ[1,1] * Lˢ[2,1] + Lˢ[1,2] * Lˢ[2,2]
        :corfa = sign(x) * √(1 / (1 + ( (Lˢ[1,1] * Lˢ[2,2] - Lˢ[1,2] * Lˢ[2,1]) / x )^2))
        y = Δ[1,1] * Δ[2,1] * :sigma1^2 * :a0^2 + (Δ[1,1] * Δ[2,2] + Δ[1,2] * Δ[2,1]) * :a0 * :a0 * :sigma1 * :sigma2 * :rho + Δ[2,2] * Δ[1,2] * :a0^2 * :sigma2^2
        :corfb = sign(y) * √(1 / (1 + ( ((Δ[1,1] * Δ[2,2] - Δ[1,2] * Δ[2,1]) * :a0 * :a0 * :sigma1 * :sigma2 * √(1 - :rho^2)) / y )^2))
        :corlim = √(1 / (1 + ( (1 - :ws12 * :ws21) * √(1 - :rho^2) / (:ws21 + (1 + :ws12 * :ws21) * :rho + :ws12) )^2))
    end
end

end # module
