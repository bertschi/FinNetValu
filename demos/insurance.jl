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

function optidemo(fun, iterations; maximize = true, λ₁ = 1e-6)
    ## 8 large and 16 small banks
    dᵉ = vcat(fill(10., 8), fill(1., 16))
    N = length(dᵉ)
    a₀ = dᵉ * 1.1
    θ = BlackScholesParams(0.0, 1.0, 0.25)
    
    ## Optimize fun via stochastic gradient
    unL = rand(Normal(-2., 1.), N * (N - 1))
    function makeL(unL)
        idx = 0
        L = Matrix{eltype(unL)}(undef, N, N)
        for i = 1:N
            for j = 1:N
                if i != j
                    L[i,j] = softplus(unL[idx += 1])
                else
                    L[i,j] = zero(eltype(unL))
                end
            end
        end
        L
    end
    
    ## opt = ADAM(1e-2)
    opt = RMSProp(1e-1)
    for i = 1:iterations
        if (i - 1) % 25 == 0
            println(string("Step ", i - 1, ":"))
            println(loss(fun, createXOS(makeL(unL), dᵉ), a₀, θ, 2500))
        end
        dVdunL = ForwardDiff.gradient(unL -> ifelse(maximize, -1., 1.) * loss(fun, createXOS(makeL(unL), dᵉ), a₀, θ, 10) + λ₁ * sum(softplus.(unL)),
                                      unL)
        Flux.Tracker.update!(opt, unL, dVdunL)
    end
    makeL(unL)
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

## Here, we numerically test our gradient calculations (see XOS/grads.tex)
δ(i, j) = i == j

function compgradL(L, dᵉ, a)
    net = createXOS(L, dᵉ)
    x = fixvalue(net, a)
    r = debtview(net, x)
    ξ = solvent(net, x)
    
    N = numfirms(net)
    dgdr = Matrix{Float64}(undef, N, N)
    for i = 1:N
        for k = 1:N
            if ξ[i]
                dgdr[i,k] = 0.
            else
                dgdr[i,k] = net.Mᵈ[i,k]
            end
        end
    end
    W = (I - dgdr) \ I
    drdL = Array{Float64,3}(undef, (N, N, N))
    for i = 1:N
        for k = 1:N
            for l = 1:N
                drdL[i,k,l] = W[i,k] * ξ[k] + r[k] / net.d[k] * (W[i,l] * (1. - ξ[l]) - sum(W[i,:] .* (1. .- ξ) .* net.Mᵈ[:,k]))
            end
        end
    end

    ## Compare with direct attempt
    drdL_AD = reshape(ForwardDiff.jacobian(L -> debtview(net, fixvalue(createXOS(L, dᵉ), a)), L),
                      N, N, N)
    ## and the formula by Demange
    dVdL_Dema = Array{Float64,2}(undef, (N, N))
    for k = 1:N
        for l = 1:N
            dVdL_Dema[k,l] = r[k] / net.d[k] * ( 1. - sum(W[:,k]) * (1 - ξ[k]) + sum(W[:,l]) * (1 - ξ[l]) ) 
        end
    end
    drdL, drdL_AD, dVdL_Dema
end

function testgrad(N)
    L = rand(Uniform(0.5, 1.5), N, N)
    dᵉ = rand(Uniform(0., 1.), N)
    a = rand(Uniform(0.5, 1.), N)

    drdL, drdL_AD, dVdL_Dema = compgradL(L, dᵉ, a)
    println("Jacobian matrices:")
    println(drdL)
    println(drdL_AD)
    
    println("Value gradients:")
    println(reshape(sum(drdL; dims = 1), N, N))
    println(reshape(sum(drdL_AD; dims = 1), N, N))
    println(dVdL_Dema)
    @assert reshape(sum(drdL; dims = 1), N, N) ≈ dVdL_Dema
    
    drdL, drdL_AD, dVdL_Dema
end

function gradADfull(N)
    Mˢ = rescale(rand(Uniform(0., 1.), N, N),
                 rand(Uniform(0.2, 0.6), N))
    L = rand(Uniform(0., 1.), N, N)
    dᵉ = rand(Uniform(0., 1.), N)
    a = rand(Uniform(0.5, 1.), N)

    function net(L)
        d = vec(sum(L; dims = 2)) .+ dᵉ
        Mᵈ = (L ./ d)'
        XOSModel(Mˢ, Mᵈ, I, d)
    end

    ## Compute drdL via AD
    reshape(ForwardDiff.jacobian(L -> debtview(net(L), fixvalue(net(L), a)), L),
            N, N, N)
end

function testgradC(N)
    ## Test gradient calculations for insurance cost
    L = rand(Uniform(0.5, 1.5), N, N)
    dᵉ = rand(Uniform(0., 1.), N)
    a = rand(Uniform(0.5, 1.), N)

    net = createXOS(L, dᵉ)
    x = fixvalue(net, a)
    r = debtview(net, x)
    ξ = solvent(net, x)
    
    N = numfirms(net)
    dgdr = Matrix{Float64}(undef, N, N)
    for i = 1:N
        for k = 1:N
            if ξ[i]
                dgdr[i,k] = 0.
            else
                dgdr[i,k] = net.Mᵈ[i,k]
            end
        end
    end
    W = (I - dgdr) \ I
    dCdL = Matrix{Float64}(undef, N, N)
    extfrac = (1 .- vec(sum(net.Mᵈ; dims = 1)))
    for k = 1:N
        for l = 1:N
            dCdL[k,l] = r[k] / net.d[k] * sum(extfrac .* (W[:, k] .* (1. - ξ[k]) .- W[:, l] .* (1. - ξ[l])))
        end
    end
    dCdL_AD = ForwardDiff.gradient(L -> insurancecost(createXOS(L, dᵉ), a) .* sum(dᵉ),
                                   L)
    dCdL, dCdL_AD
end
    
## Try with NEVA model which allows for default costs
function insurancecost(net::NEVAModel, a::AbstractVector)
    e = fixvalue(net, a)
    ## Valuation adjustment
    v = vec(net.𝕍(net, e, a))
    ## leads to insurance cost
    dᵉ = net.l .- vec(sum(net.A'; dims = 2))
    sum((1. .- v) .* dᵉ) / sum(dᵉ)
end

function optineva(lossfun; maximize = false, β = 1., σ = 1., algo = BFGS())
    ## 8 large and 16 small banks
    dᵉ = vcat(fill(10., 8), fill(1., 16))
    N = length(dᵉ)
    a₀ = dᵉ * 1.1
    θ = BlackScholesParams(0.0, 1.0, σ)
    
    ## Optimize fun via BFGS
    L = zeros(N, N)
    function makeL(unL)
        idx = 0
        for i = 1:N
            for j = 1:N
                if i != j
                    L[i,j] = exp(unL[idx += 1])
                end
            end
        end
        L
    end
    opt = optimize(unL -> ifelse(maximize, -1., 1.) * lossfun(ExAnteEN_BS_Model(dᵉ, makeL(unL), β, θ), a₀),
                   rand(Normal(-2., 1), N*(N-1)),
                   algo,
                   Optim.Options(show_trace = true,
                                 show_every = 1,
                                 iterations = 100))
    println(opt)
    makeL(opt.minimizer)
end

# Try this ...
# L = optineva(insurancecost; σ = 0.25, β = 0.7, algo = ConjugateGradient())

################################################################################
## Investigate the loss distribution with and without debt cross-holdings
################################################################################

import NLsolve

function costdist_net(xosmodel, dᵉ, L::AbstractMatrix)
    N, M = size(L)
    @assert N == M "Liabilitiy matrix must be square!"

    d = vec(sum(L; dims = 2)) .+ dᵉ
    Mᵈ = (L ./ d)'
    xosmodel(zeros(N, N), Mᵈ, I, d)
end

function costdist_fun(net, a₀, θ::BlackScholesParams, z::AbstractVector)
    a = Aτ(a₀, θ, z)
    insurancecost(net, a), sum(1 .- solvent(net, fixvalue(net, a)))
end

function costdist_sim(net, a₀::Union{AbstractFloat,AbstractVector}, θ::BlackScholesParams)
    pz = MvNormal(numfirms(net), 1)
    map(z -> FinNetValu.Sample(costdist_fun(net, a₀, θ, z)),
        [ rand(pz) for _ = 1:1000 ])
end

function costdist_sim_is(net::XOSModel, a₀::Union{AbstractFloat,AbstractVector}, θ::BlackScholesParams)
    ## Use importance sampling focusing on default boundary
    function val(net, a)
        x = fixvalue(net, a)
        equityview(net, x) .+ debtview(net, x)
    end
    deff = NLsolve.nlsolve(a -> val(net, a) .- net.d, net.d).zero ## Find point where all firms just default
    μ = NLsolve.nlsolve(z -> Aτ(a₀, θ, z) .- deff, zeros(numfirms(net))).zero ## Find corresponding unconstrained asset values
    qz = MvNormal(0.5 .* μ, max.(2.0, μ)) ## Importance density
    pz = MvNormal(numfirms(net), 1)     ## Sampling density
    w(z) = exp(logpdf(pz, z) - logpdf(qz, z))

    map(z -> FinNetValu.Sample(costdist_fun(net, a₀, θ, z), w(z)),
        [ rand(qz) for _ = 1:100000 ])
end

function full_graph(N)
    [if (i == j) 0.0 else 1.0 end
     for i = 1:N, j = 1:N]
end

function costdist_demo(lij, sim, model)
    # ## 8 large and 16 small banks
    # dᵉ = vcat(fill(10., 8), fill(1., 16))
    dᵉ = fill(1., 25)
    N = length(dᵉ)
    
    L = lij .* full_graph(N)
    net = model(dᵉ, L)

    sim(net, 2.2 .* dᵉ, BlackScholesParams(0.0, 1.0, 0.4))
end

function scatter_res(samples)
    scatter(map(x -> x.val[2], samples),
            map(x -> x.val[1], samples),
            markeralpha = map(x -> x.weight, samples),
            markersize = 1)
end

function box_res(samples)
    boxplot(map(x -> x.val[2], samples),
            map(x -> x.val[1], samples))
end

function costdist_main(model)
    function run(model, lij)
        res = costdist_demo(lij, costdist_sim, model)
        Plots.plot(scatter_res(res), histogram(map(x -> x.val[2], res)))
    end
    Plots.plot([run(model, lij) for lij in [0.0, 8.0]]...,
               layout = (2,1))
end

## For comparison create XOSModel with default cost

struct XOSModelCost{T1,T2,T3,U,V} <: FinNetValu.FinancialModel
    N::Int64
    Mˢ::T1
    Mᵈ::T2
    Mᵉ::T3
    d::U
    β::V

    function XOSModelCost(Mˢ::T1, Mᵈ::T2, Mᵉ::T3, d::AbstractVector, β::Real) where {T1,T2,T3}
        @assert FinNetValu.isleft_substochastic(Mˢ)
        @assert FinNetValu.isleft_substochastic(Mᵈ)
        @assert all(d .>= 0)
        new{T1,T2,T3,typeof(d),typeof(β)}(length(d), Mˢ, Mᵈ, Mᵉ, d, β)
    end
end

FinNetValu.numfirms(net::XOSModelCost) = net.N

recovery(d, x, β) = if (x < d) β * x else d end

function FinNetValu.valuation!(y, net::XOSModelCost, x, a)
    tmp = net.Mᵉ * a .+ net.Mˢ * equityview(net, x) .+ net.Mᵈ * debtview(net, x)
    equityview(net, y) .= max.(zero(eltype(x)), tmp .- net.d)
    debtview(net, y)   .= recovery.(net.d, tmp, net.β)
end

function FinNetValu.valuation(net::XOSModelCost, x, a)
    tmp = net.Mᵉ * a .+ net.Mˢ * equityview(net, x) .+ net.Mᵈ * debtview(net, x)
    vcat(max.(zero(eltype(x)), tmp .- net.d),
         recovery.(net.d, tmp, net.β))
end

function FinNetValu.solvent(net::XOSModelCost, x)
    equityview(net, x) .> zero(eltype(x))
end

function FinNetValu.init(net::XOSModelCost, a)
    vcat(max.(a .- net.d, 0), net.d)
end

FinNetValu.equityview(net::XOSModelCost, x::AbstractVector) = view(x, 1:numfirms(net))
FinNetValu.equityview(net::XOSModelCost, x::AbstractMatrix) = view(x, 1:numfirms(net), :)

FinNetValu.debtview(net::XOSModelCost, x::AbstractVector) = begin N = numfirms(net); view(x, (N+1):(2*N)) end
FinNetValu.debtview(net::XOSModelCost, x::AbstractMatrix) = begin N = numfirms(net); view(x, (N+1):(2*N), :) end


## Should be able to derive this!!
function insurancecost(net::XOSModelCost, a::AbstractVector)
    r = debtview(net, fixvalue(net, a))
    α = 1 .- vec(sum(net.Mᵈ; dims = 1)) ## Fraction of external debt
    dᵉ = α .* net.d
    sum(dᵉ .- α .* r) / sum(dᵉ)
end

## Example call with default cost
# β = 0.0
# model(ms,md,me,d) = XOSModelCost(ms,md,me,d,β)
# costdist_main((d,L) -> costdist_net(model, d, L))
