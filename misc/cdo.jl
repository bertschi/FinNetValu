## Investigate CDO type tranching of losses on network debt

using FinNetValu
using Distributions
using Random
using LinearAlgebra
using DataFrames
using DataFramesMeta
using CSV

rng = MersenneTwister(1234)

function symringnet(N::Integer, w::Real)
    ## Symmetric ring network on N nodes with total edge weights w
    A = diagm(1 => fill(w/2, N-1),
              -1 => fill(w/2, N-1))
    A[1, N] = w/2
    A[N, 1] = w/2
    return A
end
    
function fullnet(N::Integer, w::Real)
    ## Fully connected network on N nodes with total edge weights w
    return w ./ (N-1) .* (ones(N, N) .- I(N))
end

"""
    insuranceloss(net::FinancialModel, a::AbstractVector)

Compute per euro loss on total external debt.
"""
function insuranceloss(net::Union{XOSModel,RVEqModel}, x::FinNetValu.DefaultModelState)
    α = 1 .- vec(sum(net.Mᵈ; dims = 1)) ## Fraction of external debt
    dᵉ = α .* nominaldebt(net)
    sum(dᵉ .- α .* x.debt) / sum(dᵉ)
end

function demo(net::DefaultModel, a₀, θ::BlackScholesParams)
    Z = [ rand(rng, Normal(0, 1), numfirms(net))
          for i ∈ 1:25000 ]
    solver = PicardIteration(1e-12, 1e-12)
    function data(z)
        a = Aτ(a₀, θ, z)
        x = debtequity(net, fixvalue(solver, net, a), a)
        (loss = insuranceloss(net, x),
         default = sum(1 .- solvent(net, x)))
    end
    return map(data, Z)
end

function make_net(Mˢ::AbstractMatrix, L::AbstractMatrix, dᵉ::AbstractVector, α::Real, β::Real)
    d = vec(sum(L; dims = 2)) .+ dᵉ
    Mᵈ = (L ./ d)'
    N = length(d)
    return RVEqModel(Mˢ, Mᵈ, I(N), d, α, β, β)
    ## XOSModel(Mˢ, Mᵈ, I(N), d)
end

df = crossjoin(DataFrame(equity = [0, 0.25]),
               DataFrame(eqtype = [symringnet, fullnet]),
               DataFrame(debt = [0, 1/3, 1]),
               DataFrame(dbtype = [symringnet, fullnet]),
               DataFrame(alpha_beta = [0, 1/2, 1]),
               DataFrame(r = 0.0, tau = 1.0, sigma = 0.15,
                         a0 = 1.2, N = 20))
df = @eachrow df begin
    @newcol loss::Vector{Vector{Float64}}
    @newcol default::Vector{Vector{Float64}}
    data = @time demo(make_net(:eqtype(:N, :equity), :dbtype(:N, :debt), ones(:N), :alpha_beta, :alpha_beta),
                      :a0, BlackScholesParams(:r, :tau, :sigma))
    :loss = map(x -> x.loss, data)
    :default = map(x -> x.default, data)
end
df = flatten(df, [:loss, :default])

df |> CSV.write("/tmp/foo.csv")

function shockdata(net::DefaultModel, a::AbstractVector)
    solver = PicardIteration(1e-12, 1e-12)
    x = debtequity(net, fixvalue(solver, net, a), a)
    (loss = insuranceloss(net, x),
     default = sum(1 .- solvent(net, x)))
end

df = crossjoin(DataFrame(equity = [0, 0.25]),
               DataFrame(eqtype = [symringnet, fullnet]),
               DataFrame(debt = [0, 1/3, 1]),
               DataFrame(dbtype = [symringnet, fullnet]),
               DataFrame(alpha_beta = [0, 1/2, 1]),
               DataFrame(a0 = 1.1, N = 20),
               DataFrame(shock = range(0, 1, length = 101)))
df = @eachrow df begin
    @newcol loss::Vector{Float64}
    @newcol default::Vector{Float64}
    a = fill(:a0, :N)
    a[1] = (1 - :shock) * a[1]
    data = shockdata(make_net(:eqtype(:N, :equity), :dbtype(:N, :debt), ones(:N), :alpha_beta, :alpha_beta),
                     a)
    :loss = data.loss
    :default = data.default
end

df |> CSV.write("/tmp/baz.csv")
