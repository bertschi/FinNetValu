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
        Π = L ./ l
        new{typeof(L),U}(L, l, Π, convert(U, α), convert(U, β))
    end
end

show(io::IO, net::RVOrigModel) = print(io, "Original RV model on N = ", numfirms(net), " firms")

##############################################
# Implementation of FinancialModel interface #
##############################################

numfirms(net::RVOrigModel) = length(net.l)

nominaldebt(net::RVOrigModel) = net.l

function valuation(net::RVOrigModel, l, e)
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

init(sol::PicardIteration, net::RVOrigModel, e) = nominaldebt(net)

init(sol::NLSolver, net::RVOrigModel, e) = nominaldebt(net)

function solvent(net::RVOrigModel, l::AbstractVector)
    l .>= nominaldebt(net)
end

function finalizestate(net::RVOrigModel, l, e)
    ξ = solvent(net, l)
    equity = ξ .* (e .+ net.Π' * l .- l)
    ModelState(equity, l)
end

#################################################
# Greatest Clearing Vector Algorithm from paper #
#################################################

struct GCVASolver <: FixSolver end

init(sol::GCVASolver, net::RVOrigModel, e) = nominaldebt(net)

function fixvalue(sol::GCVASolver, net::RVOrigModel, e; finalize = true)
    N = numfirms(net)
    l̄ = nominaldebt(net)
    Λ = copy(init(sol, net, e))
    i₀ = fill(false, N)
    while true
        v = e .+ net.Π' * Λ .- l̄
        i = v .< 0.0
        if i == i₀
            return maybefinalize(finalize, net, Λ, e)
        end
        s = .!(i)
        Λ[s] .= l̄[s]
        x = (I(sum(i)) .- net.β .* net.Π'[i, i]) \ (net.α .* e[i] .+ net.β .* net.Π'[i, s] * l̄[s])
        Λ[i] .= x
        i₀ = i
    end
end
    
"""
    RVEqModel(Mˢ, Mᵈ, d)

Rogers & Veraart model extended for cross-holdings of equity `Mˢ` and
parameterized in terms of debt cross-holding fractions `Mᵈ` and total
nominal debt `d` (a la XOSModel).
"""
struct RVEqModel{T1,T2,U} <: DefaultModel
    N::Int64
    Mˢ::T1
    Mᵈ::T2
    d::U

    function RVEqModel(Mˢ::AbstractMatrix, Mᵈ::AbstractMatrix, d::AbstractVector)
        @assert isleft_substochastic(Mˢ)
        @assert isleft_substochastic(Mᵈ)
        @assert all(d .>= 0)
        new{typeof(Mˢ),typeof(Mᵈ),typeof(d)}(length(d), Mˢ, Mᵈ, d)
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

