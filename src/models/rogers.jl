import Base.show
using LinearAlgebra

"""
    RogersModel(L, A)

Financial network model with nominal internal liabilities `L`.
Nominal values of external assets and equity are given by `A`.
The fire sale parameters (defualt costs) are `α` for assets 
and `β` for debt holdings. 
"""
struct RogersModel <: FinancialModel
    name
    α
    β
    L
    Π
    A

    """
        RogersModel(L, A)

    Construct Rogers model with `N` firms (where `N` is determined
    from the size of `L`), internal liabilities `L`.
    Values of external assets are given by `A`. 
    The fire sale parameters (default costs) are `α` for assets 
    and `β` for debt holdings. 
    """
    function RogersModel(name::String, L, A, α, β)
        @assert all(L .>= 0)
        N = size(L, 1)
        Lbar = sum(L, dims=[2])
        Π = L ./ map(x-> if x == 0.0 Inf else x end, Lbar)
        new(name, α, β, L, Π, A)
    end
end
