import Base.show

"""
    XOSModel(N, Mˢ, Mᵈ, Mᵉ, d)

Financial network of `N` firms with investment portfolios `Mˢ`, `Mᵈ`
and `Mᵉ` of holding fractions in counterparties equity, debt and
external assets respectively. Nominal debt `d` is due at maturity.
"""
struct XOSModel <: FinancialModel
    N
    Mˢ
    Mᵈ
    Mᵉ
    d

    function XOSModel(N::Integer, Mˢ, Mᵈ, Mᵉ, d)
        @assert isleft_substochastic(Mˢ)
        @assert isleft_substochastic(Mᵈ)
        @assert all(d .>= 0)
        new(N, Mˢ, Mᵈ, Mᵉ, d)
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

function valuation!(y, net::XOSModel, x, a)
    ei = _eqidx(net)
    di = _dbidx(net)
    tmp = net.Mᵉ * a .+ net.Mˢ * x[ei] .+ net.Mᵈ * x[di]
    y[ei] .= max.(zero(eltype(x)), tmp .- net.d)
    y[di] .= min.(net.d, tmp)
end

function solvent(net::XOSModel, x)
    equity(net, x) .> zero(eltype(x))
end

function init(net::XOSModel, a)
    vcat(max.(a .- net.d, 0), net.d)
end

##########################
# Model specific methods #
##########################

_eqidx(net::XOSModel) = 1:numfirms(net)
function _dbidx(net::XOSModel)
    N = numfirms(net)
    (N+1):(2*N)
end

equity(net::XOSModel, x) = x[_eqidx(net)]
debt(net::XOSModel, x) = x[_dbidx(net)]
