import Base.show

"""
    NEVAModel(Lᵉ, L, 𝕍ᵉ, 𝕍)

Financial network model with nominal external `Lᵉ` and internal
liabilities `L`. Nominal values of external assets and equity are
weighted by the valuation functions `𝕍ᵉ` and `𝕍` respectively.
"""
struct NEVAModel <: FinancialModel
    name
    N
    A
    l
    𝕍ᵉ
    𝕍

    """
        NEVAModel(N, Lᵉ, L, 𝕍ᵉ, 𝕍)

    Construct NEVA model with `N` firms, external `Lᵉ` and internal
    liabilities `L`. Values of external assets and liabilities of
    counterparties are adjusted by the valuation functions `𝕍ᵉ` and
    `𝕍` respectively."""

    function NEVAModel(name::String, N::Integer, Lᵉ, L, 𝕍ᵉ::Function, 𝕍::Function)
        @assert all(Lᵉ .>= 0)
        @assert all(L .>= 0)
        A = copy(L')
        l = rowsums(L) .+ Lᵉ
        new(name, N, A, l, 𝕍ᵉ, 𝕍)
    end
end

function show(io::IO, net::NEVAModel)
    print(io, net.name, " model with N = ", numfirms(net), " firms.")
end

############################
# Convenience constructors #
############################

function NEVAModel(name::String, Lᵉ::AbstractVector, L, 𝕍ᵉ::Function, 𝕍::Function)
    NEVAModel(name, length(Lᵉ), Lᵉ, L, 𝕍ᵉ, 𝕍)
end

function NEVAModel(name::String, Lᵉ, L::AbstractMatrix, 𝕍ᵉ::Function, 𝕍::Function)
    @assert size(L, 1) == size(L, 2)
    NEVAModel(name, size(L, 1), Lᵉ, L, 𝕍ᵉ, 𝕍)
end

function NEVAModel(name::String, Lᵉ::AbstractVector, L::AbstractMatrix, 𝕍ᵉ::Function, 𝕍::Function)
    @assert length(Lᵉ) == size(L, 1) == size(L, 2)
    NEVAModel(name, length(Lᵉ), Lᵉ, L, 𝕍ᵉ, 𝕍)
end

##############################################
# Implementation of FinancialModel interface #
##############################################

numfirms(net::NEVAModel) = net.N

function valuation!(y, net::NEVAModel, x, a)
    y .= a .* net.𝕍ᵉ(net, x, a) .+ rowsums(net.A .* net.𝕍(net, x, a)) .- net.l
end

function valuation(net::NEVAModel, x, a)
    a .* net.𝕍ᵉ(net, x, a) .+ rowsums(net.A .* net.𝕍(net, x, a)) .- net.l
end

function solvent(net::NEVAModel, x)
    x .> zero(eltype(x))
end

function init(net::NEVAModel, a)
    ## Initialize between boundaries m <= M
    a .- net.l
end

##########################
# Model specific methods #
##########################

bookequity(net::NEVAModel, a) = a .+ rowsums(net.A) .- net.l

##########################################
# Constructors for different models from #
# arxiv???                               #
##########################################

function EisenbergNoeModel(Lᵉ::AbstractVector, L::AbstractMatrix)
    pbar = vec(sum(L; dims = 2))
    function val(net, e, a)
        fillrows(@. (e >= 0) + max(e + pbar, 0) / pbar * (e < 0))
    end
    NEVAModel("Eisenberg & Noe",
              Lᵉ,
              L,
              constantly(one(eltype(L))),
              val)
end

function FurfineModel(Lᵉ::AbstractVector, L::AbstractMatrix, R::Real)
    @assert 0 <= R <= 1
    val(net, e, a) = fillrows(@. (e >= 0) + R * (e < 0))
    NEVAModel("Furfine",
              Lᵉ,
              L,
              constantly(one(eltype(L))),
              val)
end

function LinearDebtRankModel(Lᵉ::AbstractVector, L::AbstractMatrix)
    function val(net, e, a)
        fillrows(max.(e, 0) ./ bookequity(net, a))
    end
    NEVAModel("Linear Debt Rank",
              Lᵉ,
              L,
              constantly(one(eltype(L))),
              val)
end
