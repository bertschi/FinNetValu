################################################################################
## Using AD to compute impact of network changes on insurance cost
##
## Also experiments with optimizing networks to achieve low insurance
## cost or high shareholder values ...
################################################################################

using FinNetValu
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Distributions
using Optim
using Flux
using Plots

"""
    createXOS(L, dᵉ)

Create an XOS network from given liability matrix `L` and external
liability vector `dᵉ`.  
"""
function createXOS(L::AbstractMatrix, dᵉ::AbstractVector)
    d = vec(sum(L; dims = 2)) .+ dᵉ
    Mᵈ = (L ./ d)'
    N = length(d)
    XOSModel(spzeros(N, N), Mᵈ, I, d)
end

"""
    insurancecost(net::FinancialModel, a::AbstractVector)

Compute per euro cost of total insurance on external debt.
"""
function insurancecost(net::XOSModel, a::AbstractVector)
    r = debtview(net, fixvalue(net, a))
    α = 1 .- vec(sum(net.Mᵈ; dims = 1)) ## Fraction of external debt
    dᵉ = α .* net.d
    sum(dᵉ .- α .* r) / sum(dᵉ)
end

"""
    shareholdervalue(net::FinancialModel, a::AbstractVector)

Compute average shareholder value to external investors.
"""
function shareholdervalue(net::XOSModel, a::AbstractVector)
    x = fixvalue(net, a)
    α = 1 .- vec(sum(net.Mˢ; dims = 1)) ## Fraction to external shareholders
    s = equityview(net, x)
    ## λ = (s .+ debtview(net, x)) ./ (s .+ 1e-6)
    ## mean(α .* s) ## - 1e-3 * sum(.- log.(50. .- λ)) ## Log barrier for λ < 50
    mean(log.(α .* s .+ 1e-25)) ## External investors with log utility
end

function loss(fun, net, a₀, θ, n)
    expectation(z -> fun(net, Aτ(a₀, θ, z)),
                MonteCarloSampler(MvNormal(numfirms(net), 1)),
                n)
end

function demo(fun, N, n)
    L = rand(Uniform(0, 2), N, N)
    dᵉ = rand(Uniform(0, 1), N)
    
    a₀ = 0.8
    θ = BlackScholesParams(0.0, 1.0, 0.25)

    dVdL = reshape(ForwardDiff.gradient(L -> loss(fun, createXOS(L, dᵉ), a₀, θ, n),
                                        L),
                   N, N)
    dVdL
end

function optidemo(fun, iterations; maximize = true)
    ## 8 large and 16 small banks
    dᵉ = vcat(fill(10., 8), fill(1., 16))
    N = length(dᵉ)
    a₀ = dᵉ * 1.1
    θ = BlackScholesParams(0.0, 1.0, 0.25)
    
    ## Optimize fun via stochastic gradient
    unL = rand(Uniform(-2., -1.), N, N)
    
    ## opt = ADAM(1e-2)
    opt = RMSProp(1e-1)
    for i = 1:iterations
        if (i - 1) % 25 == 0
            println(string("Step ", i - 1, ":"))
            println(loss(fun, createXOS(softplus.(unL), dᵉ), a₀, θ, 2500))
        end
        dVdunL = reshape(ForwardDiff.gradient(unL -> ifelse(maximize, -1., 1.) * loss(fun, createXOS(softplus.(unL), dᵉ), a₀, θ, 10),
                                              unL),
                         N, N)
        Flux.Tracker.update!(opt, unL, dVdunL)
    end
    softplus.(unL)
end

function bfgsdemo(fun; maximize = true)
    ## 8 large and 16 small banks
    dᵉ = vcat(fill(10., 8), fill(1., 16))
    N = length(dᵉ)
    a₀ = dᵉ * 1.1
    θ = BlackScholesParams(0.0, 1.0, 0.25)
    
    ## Optimize fun via BFGS
    opt = optimize(unL -> ifelse(maximize, -1., 1.) * loss(fun, createXOS(softplus.(unL), dᵉ), a₀, θ, 250),
                   rand(Normal(-2., 1), N, N),
                   BFGS(),
                   Optim.Options(show_trace = true,
                                 show_every = 1))
    softplus.(opt.minimizer)
end
