import Base.show
using LinearAlgebra

"""
    XOSModel(N, Mˢ, Mᵈ, Mᵉ, d)

Financial network of firms with investment portfolios `Mˢ`, `Mᵈ`
and `Mᵉ` of holding fractions in counterparties equity, debt and
external assets respectively. Nominal debt `d` is due at maturity.

Note that `Mˢ`, `Mᵈ` are required to be left substochastic matrices
and `d` must be a non-negative vector.
"""
struct XOSModel{T1,T2,T3,U} <: FinancialModel
    N::Int64
    Mˢ::T1
    Mᵈ::T2
    Mᵉ::T3
    d::U

    function XOSModel(Mˢ::T1, Mᵈ::T2, Mᵉ::T3, d::AbstractVector) where {T1,T2,T3}
        @assert isleft_substochastic(Mˢ)
        @assert isleft_substochastic(Mᵈ)
        @assert all(d .>= 0)
        new{T1,T2,T3,typeof(d)}(length(d), Mˢ, Mᵈ, Mᵉ, d)
    end
end

function show(io::IO, net::XOSModel)
    if any(net.Mˢ .> 0) & any(net.Mᵈ .> 0)
        msg = "equity and debt"
    elseif any(net.Mˢ .> 0)
        msg = "equity"
    elseif any(net.Mᵈ .> 0)
        msg = "debt"
    else
        msg = "no"
    end
    print(io, "XOS model of N = ",
          numfirms(net), " firms with ",
          msg, " cross holdings.")
end

##############################################
# Implementation of FinancialModel interface #
##############################################

numfirms(net::XOSModel) = net.N

nominaldebt(net::XOSModel) = net.d

equity(net::XOSModel, x::AbstractVector) = view(x, 1:numfirms(net))
equity(net::XOSModel, x::AbstractMatrix) = view(x, 1:numfirms(net), :)

debt(net::XOSModel, x::AbstractVector) = begin N = numfirms(net); view(x, (N+1):(2*N)) end
debt(net::XOSModel, x::AbstractMatrix) = begin N = numfirms(net); view(x, (N+1):(2*N), :) end

function valuation!(y, net::XOSModel, x, a)
    tmp = net.Mᵉ * a .+ net.Mˢ * equity(net, x) .+ net.Mᵈ * debt(net, x)
    equity(net, y) .= max.(zero(eltype(x)), tmp .- net.d)
    debt(net, y)   .= min.(net.d, tmp)
end

function valuation(net::XOSModel, x, a)
    tmp = net.Mᵉ * a .+ net.Mˢ * equity(net, x) .+ net.Mᵈ * debt(net, x)
    vcat(max.(zero(eltype(x)), tmp .- net.d),
         min.(net.d, tmp))
end

function fixjacobian(net::XOSModel, a, x = fixvalue(net, a))
    ## Uses analytical formulas for speedup
    ξ = solvent(net, x)
    eins = one(eltype(ξ))
    dVdx = vcat(hcat(Diagonal(ξ) * net.Mˢ, Diagonal(ξ) * net.Mᵈ),
                hcat(Diagonal(eins .- ξ) * net.Mˢ, Diagonal(eins .- ξ) * net.Mᵈ))
    dVda = vcat(Diagonal(ξ), Diagonal(1.0 .- ξ)) * net.Mᵉ
    (I - dVdx) \ Matrix(dVda) ## Note: RHS needs to be dense
end

function solvent(net::XOSModel, x)
    equity(net, x) .> zero(eltype(x))
end

function init(sol::NLSolver, net::XOSModel, a)
    vcat(max.(a .- net.d, 0), net.d)
end

function init(sol::PicardIteration, net::XOSModel, a)
    vcat(max.(a .- net.d, 0), net.d)
end
