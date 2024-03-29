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
    E₀
    
    """
        NEVAModel(N, Lᵉ, L, 𝕍ᵉ, 𝕍)

    Construct NEVA model with `N` firms, external `Lᵉ` and internal
    liabilities `L`. Values of external assets and liabilities of
    counterparties are adjusted by the valuation functions `𝕍ᵉ` and
    `𝕍` respectively. The initial equity is computed via `E₀`."""

    function NEVAModel(name::String, N::Integer, Lᵉ, L, 𝕍ᵉ::Function, 𝕍::Function, E₀::Function)
        @assert all(Lᵉ .>= 0)
        @assert all(L .>= 0)
        A = copy(L')
        l = rowsums(L) .+ Lᵉ
        new(name, N, A, l, 𝕍ᵉ, 𝕍, E₀)
    end
end

function show(io::IO, net::NEVAModel)
    print(io, net.name, " model with N = ", numfirms(net), " firms.")
end

############################
# Convenience constructors #
############################

function NEVAModel(name::String, Lᵉ::AbstractVector, L, 𝕍ᵉ::Function, 𝕍::Function, E₀::Function)
    NEVAModel(name, length(Lᵉ), Lᵉ, L, 𝕍ᵉ, 𝕍, E₀)
end

function NEVAModel(name::String, Lᵉ, L::AbstractMatrix, 𝕍ᵉ::Function, 𝕍::Function, E₀::Function)
    @assert size(L, 1) == size(L, 2)
    NEVAModel(name, size(L, 1), Lᵉ, L, 𝕍ᵉ, 𝕍, E₀)
end

function NEVAModel(name::String, Lᵉ::AbstractVector, L::AbstractMatrix, 𝕍ᵉ::Function, 𝕍::Function, E₀::Function)
    @assert length(Lᵉ) == size(L, 1) == size(L, 2)
    NEVAModel(name, length(Lᵉ), Lᵉ, L, 𝕍ᵉ, 𝕍, E₀)
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

bookequity(net::NEVAModel, a) = a .+ rowsums(net.A) .- net.l

function init(net::NEVAModel, a)
    net.E₀(net, a)
end

##########################################
# Constructors for different models from #
# arxiv:1606.05164                       #
##########################################

valueEN(e::Real, pbar::Real) = if (e > 0) 1. else max((e + pbar) / pbar, 0.) end

"""
    EisenbergNoeModel(Lᵉ, L)

Creates an instance of the NEVAModel with valuation functions

```math
\\begin{align}
\\mathbb{V}^e_i(E_i) &= 1 \\quad \\forall i \\\\
\\mathbb{V}_{ij}(E_j) &= \\unicode{x1D7D9}_{E_j \\geq 0} + \\left(\\frac{E_j + \\bar{p}_j}{\\bar{p}_j}\\right)^+ \\unicode{x1D7D9}_{E_j < 0} \\quad \\forall i, j
\\end{align}
```

where ``\\bar{p}_j = \\sum_k L_{jk}``.

This valuation was shown to correspond to the model by Eisenberg & Noe.
"""
function EisenbergNoeModel(Lᵉ::AbstractVector, L::AbstractMatrix)
    pbar = vec(sum(L; dims = 2))
    function val(net, e, a)
        # Note: rowvector gets broadcasted correctly as 𝕍(Eⱼ)
        transpose(valueEN.(e, pbar))
    end
    NEVAModel("Eisenberg & Noe",
              Lᵉ,
              L,
              constantly(one(eltype(L))),
              val,
              bookequity)
end

valueFurfine(e::Real, R::Real) = if (e > 0) 1. else R end

"""
    FurfineModel(Lᵉ, L, R)

Creates an instance of the NEVAModel with valuation functions

```math
\\begin{align}
\\mathbb{V}^e_i(E_i) &= 1 \\quad \\forall i \\\\
\\mathbb{V}_{ij}(E_j) &= \\unicode{x1D7D9}_{E_j \\geq 0} + R \\unicode{x1D7D9}_{E_j < 0} \\quad \\forall i, j
\\end{align}
```
with recovery `R`.

This valuation was shown to correspond to the model by Furfine.
"""
function FurfineModel(Lᵉ::AbstractVector, L::AbstractMatrix, R::Real)
    @assert 0 <= R <= 1
    function val(net, e, a)
        transpose(valueFurfine.(e, R))
    end
    NEVAModel("Furfine",
              Lᵉ,
              L,
              constantly(one(eltype(L))),
              val,
              bookequity)
end

valueLR(e::Real, ebook::Real) = if (e > ebook) 1. elseif (e > 0) e / ebook else 0. end

"""
    LinearDebtRankModel(Lᵉ, L, M)

Creates an instance of the NEVAModel with valuation functions

```math
\\begin{align}
\\mathbb{V}^e_i(E_i) &= 1 \\quad \\forall i \\\\
\\mathbb{V}_{ij}(E_j) &= \\frac{E_j^+}{M_j} \\quad \\forall i, j
\\end{align}
```
where the bookequities are externally fixed as `M`.

This valuation was shown to correspond to the linear DebtRank model.
"""
function LinearDebtRankModel(Lᵉ::AbstractVector, L::AbstractMatrix, M::AbstractVector)
    function val(net, e, a)
        transpose(valueLR.(e, M))
    end
    NEVAModel("Linear Debt Rank",
              Lᵉ,
              L,
              constantly(one(eltype(L))),
              val,
              constantly(M))
end

"""
    ExAnteEN_BS_Model(Lᵉ, L, β, θ)

Creates an instance of the NEVAModel with valuation functions

```math
\\begin{align}
\\mathbb{V}^e_i(E_i) &= 1 \\quad \\forall i \\\\
\\mathbb{V}_{ij}(E_j) &= 1 - p_j^D(E_j) + β ρ_j(E_j) \\quad \\forall i, j
\\end{align}
```
where ``p_j^D(E_j)`` and ``ρ_j(E_j)`` denote the risk neutral probability
of default and endogenous recovery respectively.

This valuation can be considered as a ex-ante version of the Eisenberg & Noe model.
"""
function ExAnteEN_BS_Model(Lᵉ::AbstractVector, L::AbstractMatrix, β, θ::BlackScholesParams)
    pbar = vec(sum(L; dims = 2))
    function val(net, e, a)
        K = a .- e
        function fun(K, a, pbar, β)
            if (K <= 0)
                1.0
            else
                # Compute probability of default
                pd = putdualΔ(a, K, θ)
                # Compute expected shortfall when defaulted
                es1 = putprice(a, K, θ)
                # and again shifted
                es2 = if ((K - pbar) <= 0) 0.0 else putprice(a, K - pbar, θ) end
                # TODO: Document this way of writing everything in put prices!
                (1 - pd) + β / pbar * ((pd * pbar - es1) + es2)
            end
        end
        # Note: rowvector gets broadcasted correctly as 𝕍(Eⱼ)
        transpose(fun.(K, a, pbar, β))
    end
    NEVAModel("Ex-ante Eisenberg & Noe (Black-Scholes)",
              Lᵉ,
              L,
              constantly(one(eltype(L))),
              val,
              bookequity)
end
