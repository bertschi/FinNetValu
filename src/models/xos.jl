import Base.show
using LinearAlgebra

"""
    XOSModel(Mˢ, Mᵈ, Mᵉ, d)

Financial network of firms with investment portfolios `Mˢ`, `Mᵈ`
and `Mᵉ` of holding fractions in counterparties equity, debt and
external assets respectively. Nominal debt `d` is due at maturity.

Note that `Mˢ`, `Mᵈ` are required to be left substochastic matrices
and `d` must be a non-negative vector.
"""
struct XOSModel{T1,T2,T3,U} <: DefaultModel
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

equityview(net::XOSModel, x::AbstractVector) = view(x, 1:numfirms(net))
equityview(net::XOSModel, x::AbstractMatrix) = view(x, 1:numfirms(net), :)

debtview(net::XOSModel, x::AbstractVector) = begin N = numfirms(net); view(x, (N+1):(2*N)) end
debtview(net::XOSModel, x::AbstractMatrix) = begin N = numfirms(net); view(x, (N+1):(2*N), :) end

function valuation!(y::AbstractVector, net::XOSModel, x::AbstractVector, a::AbstractVector)
    tmp = net.Mᵉ * a .+ net.Mˢ * equityview(net, x) .+ net.Mᵈ * debtview(net, x)
    equityview(net, y) .= max.(zero(eltype(x)), tmp .- net.d)
    debtview(net, y)   .= min.(net.d, tmp)
end

function valuation(net::XOSModel, x::AbstractVector, a::AbstractVector)
    tmp = net.Mᵉ * a .+ net.Mˢ * equityview(net, x) .+ net.Mᵈ * debtview(net, x)
    vcat(max.(zero(eltype(x)), tmp .- net.d),
         min.(net.d, tmp))
end

function fixjacobian(net::XOSModel, x::AbstractVector, a::AbstractVector)
    ## Uses analytical formulas for speedup
    ξ = solvent(net, x)
    eins = one(eltype(ξ))
    dVdx = vcat(hcat(Diagonal(ξ) * net.Mˢ, Diagonal(ξ) * net.Mᵈ),
                hcat(Diagonal(eins .- ξ) * net.Mˢ, Diagonal(eins .- ξ) * net.Mᵈ))
    dVda = vcat(Diagonal(ξ), Diagonal(1.0 .- ξ)) * net.Mᵉ
    (I - dVdx) \ Matrix(dVda) ## Note: RHS needs to be dense
end

function solvent(net::XOSModel, x::AbstractVector)
    equityview(net, x) .> zero(eltype(x))
end

function init(sol::NLSolver, net::XOSModel, a::AbstractVector)
    vcat(max.(a .- net.d, 0), net.d)
end

function init(sol::PicardIteration, net::XOSModel, a::AbstractVector)
    vcat(max.(a .- net.d, 0), net.d)
end

function debtequity(net::XOSModel, x::AbstractVector, a::AbstractVector)
    DefaultModelState(equityview(net, x), debtview(net, x))
end
