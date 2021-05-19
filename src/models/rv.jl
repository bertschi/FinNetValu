## Additional implementation of Rogers & Veraart model as in original
## paper and extended to allow for equity cross-holdings

import Base.show

"""
    RVOrigModel(L, α, β)

Rogers & Veraart model as in the original paper with total nominal
liability matrix `L` and parameters `α` and `β`.
"""
struct RVOrigModel{T,U} <: DefaultModel
    L::T
    l::Vector{U}  ## called L̄ in the paper
    Π::Matrix{U}
    α::U
    β::U

    function RVOrigModel(L::AbstractMatrix,α::Real,β::Real)
        U = eltype(L)
        l = rowsums(L)
        div0(x, y) = if y == zero(y) y else x / y end
        Π = div0.(L, l)
        new{typeof(L),U}(L, l, Π, convert(U, α), convert(U, β))
    end
end

show(io::IO, net::RVOrigModel) = print(io, "Original RV model on N = ", numfirms(net), " firms")

##############################################
# Implementation of FinancialModel interface #
##############################################

numfirms(net::RVOrigModel) = length(net.l)

nominaldebt(net::RVOrigModel) = net.l

function valuation(net::RVOrigModel, l::AbstractVector, e::AbstractVector)
    N = numfirms(net)
    function val(i)
        tmp = sum(l[j] * net.Π[j, i] for j ∈ 1:N)
        if net.l[i] <= e[i] + tmp
            net.l[i]
        else
            net.α * e[i] + net.β * tmp
        end
    end
    [val(i) for i ∈ 1:N]
end

init(sol::PicardIteration, net::RVOrigModel, e::AbstractVector) = nominaldebt(net)

init(sol::NLSolver, net::RVOrigModel, e::AbstractVector) = nominaldebt(net)

function solvent(net::RVOrigModel, l::AbstractVector)
    l .>= nominaldebt(net)
end

function debtequity(net::RVOrigModel, l::AbstractVector, e::AbstractVector)
    ξ = solvent(net, l)
    equity = ξ .* (e .+ net.Π' * l .- l)
    DefaultModelState(equity, l)
end

#################################################
# Greatest Clearing Vector Algorithm from paper #
#################################################

struct GCVASolver <: FixSolver end

init(sol::GCVASolver, net::RVOrigModel, e::AbstractVector) = nominaldebt(net)

function fixvalue(sol::GCVASolver, net::RVOrigModel, e::AbstractVector)
    N = numfirms(net)
    l̄ = nominaldebt(net)
    Λ = copy(init(sol, net, e))
    i₀ = fill(false, N)
    while true
        v = e .+ net.Π' * Λ .- l̄
        i = v .< zero(eltype(v))
        if i == i₀
            ## Done and break loop
            return Λ
        end
        s = .!(i)
        Λ[s] .= l̄[s]
        x = (I(sum(i)) .- net.β .* net.Π'[i, i]) \ (net.α .* e[i] .+ net.β .* net.Π'[i, s] * l̄[s])
        Λ[i] .= x
        i₀ = i
    end
end
    
"""
    RVEqModel(Mˢ, Mᵈ, Mᵉ, d, α, βˢ, βᵈ)

Rogers & Veraart model extended for cross-holdings of equity `Mˢ` and
parameterized in terms of debt cross-holding fractions `Mᵈ` and total
nominal debt `d` (a la XOSModel).  

As in the XOS model the portfolio matrix `Mᵉ` mixes external assets
and the RV parameters are slightly generalized to allow different
recoveries on assets (`α`), equities (`βˢ`) and debt (`βᵈ`)
respectively.
"""
struct RVEqModel{T1,T2,T3,U,T} <: DefaultModel
    N::Int64
    Mˢ::T1
    Mᵈ::T2
    Mᵉ::T3
    d::U
    α::T
    βˢ::T
    βᵈ::T

    function RVEqModel(Mˢ::AbstractMatrix, Mᵈ::AbstractMatrix, Mᵉ, d::AbstractVector, α::Real, βˢ::Real, βᵈ::Real)
        @assert isleft_substochastic(Mˢ)
        @assert isleft_substochastic(Mᵈ)
        @assert all(d .>= 0)
        α, βˢ, βᵈ = promote(α, βˢ, βᵈ)
        new{typeof(Mˢ),typeof(Mᵈ),typeof(Mᵉ),typeof(d), typeof(α)}(length(d), Mˢ, Mᵈ, Mᵉ, d, α, βˢ, βᵈ)
    end
end

function show(io::IO, net::RVEqModel)
    if any(net.Mˢ .> 0) & any(net.Mᵈ .> 0)
        msg = "equity and debt"
    elseif any(net.Mˢ .> 0)
        msg = "equity"
    elseif any(net.Mᵈ .> 0)
        msg = "debt"
    else
        msg = "no"
    end
    print(io, "RV model of N = ",
          numfirms(net), " firms with ",
          msg, " cross holdings.")
end

##############################################
# Implementation of FinancialModel interface #
##############################################

numfirms(net::RVEqModel) = net.N

nominaldebt(net::RVEqModel) = net.d

equityview(net::RVEqModel, x::AbstractVector) = view(x, 1:numfirms(net))
## equityview(net::RVEqModel, x::AbstractMatrix) = view(x, 1:numfirms(net), :)

debtview(net::RVEqModel, x::AbstractVector) = begin N = numfirms(net); view(x, (N+1):(2*N)) end
## debtview(net::RVEqModel, x::AbstractMatrix) = begin N = numfirms(net); view(x, (N+1):(2*N), :) end

function valuation(net::RVEqModel, x::AbstractVector, a::AbstractVector)
    N = numfirms(net)
    v = net.Mᵉ * a .+ net.Mˢ * equityview(net, x) .+ net.Mᵈ * debtview(net, x)
    vαβ = net.α .* net.Mᵉ * a .+ net.βˢ .* net.Mˢ * equityview(net, x) .+ net.βᵈ .* net.Mᵈ * debtview(net, x)
    equity = max.(zero(eltype(v)), v .- net.d)
    debt = ifelse.(v .< net.d, vαβ, net.d)
    vcat(equity, debt)
end

solvent(net::RVEqModel, x::AbstractVector) = debtview(net, x) .>= nominaldebt(net)

init(sol::NLSolver, net::RVEqModel, a::AbstractVector) =
    ## Note: We need to take equity cross-holdings into account and
    ## therefore initialize at α = β = 1, i.e. via the XOSModel
    fixvalue(sol, XOSModel(net.Mˢ, net.Mᵈ, net.Mᵉ, net.d), a)

init(sol::PicardIteration, net::RVEqModel, a::AbstractVector) =
    fixvalue(sol, XOSModel(net.Mˢ, net.Mᵈ, net.Mᵉ, net.d), a)

function debtequity(net::RVEqModel, x::AbstractVector, a::AbstractVector)
    DefaultModelState(equityview(net, x), debtview(net, x))
end
