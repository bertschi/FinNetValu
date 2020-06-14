import Base.show
using LinearAlgebra

"""
    RogersModel(L, A)

Financial network model with nominal internal liabilities `L`.
The fire sale parameters (defualt costs) are `α` for assets 
and `β` for debt holdings. 
The greatest clearing vector `Λ` takes to role of the fixed point `x`
in the XOS related models.
"""
struct RogersModel <: FinancialModel
    name
    α
    β
    L
    Π
    Lbar

    """
        RogersModel(L, A)

    Construct Rogers model with `N` firms (where `N` is determined
    from the size of `L`), internal liabilities `L`.
    The fire sale parameters (default costs) are `α` for assets 
    and `β` for debt holdings. 
    """
    function RogersModel(name::String, L, α, β)
        @assert all(L .>= 0)
        N = size(L, 1)
        Lbar = sum(L, dims=[2])[:,1]
        Π = L ./ map(x-> if x == 0.0 Inf else x end, Lbar)
        new(name, α, β, L, Π, Lbar)
    end
end

##############################################
# Implementation of FinancialModel interface #
##############################################


valuation!(y, net::RogersModel, x, a) = y

#TODO: check if cutoff is needed: res[x .< sum(net.L, dims=[2])[:,1]] .= zeros(eltype(x))
function valuation(net::RogersModel, Λ, a) 
    @assert all(size(Λ) .== size(a))
    res = transpose(net.Π) * Λ .+ a .- Λ
    res[Λ .< net.Lbar] .= zeros(eltype(res))
    return res
end

init(net::RogersModel, a) = sum(net.L, dims=[2])[:,1]

function fixvalue(net::RogersModel, a)
    @assert all(size(net.Lbar) .== size(a))
    Λ = repeat(init(net, a), 1, length(a)+1)
    iSet = zeros(eltype(Λ), size(Λ,1)...)
    ΠT = transpose(net.Π)

    μmax = 1
    for μ in 1:length(a)
        v = ΠT * Λ[:, μ] .+ a .- net.Lbar
        lastInsolventSet = iSet
        iSet = v .< zero(eltype(v)) 
        sSet = v .>= zero(eltype(v))
        all(lastInsolventSet .== iSet) && break
        rhs = net.α .* a[iSet] + net.β .* ΠT[iSet,sSet] * net.Lbar[sSet] 
        μmax = μ + 1
        Λ[iSet, μmax] .= (I - ΠT[iSet,iSet])\rhs
    end

    return Λ[:,1:μmax]
end

function solvent(net::RogersModel, Λ)
    v .>= zero(eltype(v))
end

numfirms(net::RogersModel) = size(net.L, 1)

##########################
# Model specific methods #
##########################

equityview(net::RogersModel, x::AbstractVector) = zeros(numfirms(net))

debtview(net::RogersModel, x::AbstractVector) = view(x, 1:length(x))
