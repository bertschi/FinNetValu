using NLsolve
using ForwardDiff
using LinearAlgebra

"""
Base type of all financial network models.
"""
abstract type FinancialModel end

"""
    valuation!(y, net, x, a)

Updates the state `y` of a financial `net` based on the current state
`x` and external asset values `a`.

Note: For convenience, e.g. when using autodiff or computing averages
      etc., you should consider representing states as (abstract)
      vectors.
"""
function valuation! end

function valuation!(y, net::FinancialModel, x, a)
    y .= valuation(net, x, a)
end

"""
    valuation(net, x, a)

Same as `valuation!`, but non destructive.
"""
function valuation end

"""
    init(solver, net, a)

Small wrapper to generate initial state `x₀` suitable for fixed
point `solver`.
"""
function init end

"""
    valuefunc(net, a)

Small wrapper to generate function f!(y, x) suitable for fixed point
solvers. Fixed point `x` should fulfill x = valuation(net, x, a).
"""
function valuefunc(net::FinancialModel, a)
    function f!(y, x)
        valuation!(y, net, x, a)
        y
    end
    f!
end

"""
Base type of fixed-point solution methods.
"""
abstract type FixSolver end

"""
    fixvalue(solver, net, a)

Solve for self-consistent fixed point value using method `solver` of
model `net` for external asset values `a`.

Note: Solver options should be part of `solver` type.
"""
function fixvalue end

struct NLSolver <: FixSolver
    opts::NamedTuple
end

function fixvalue(sol::NLSolver, net::FinancialModel, a)
    sol = fixedpoint(valuefunc(net, a),
                     init(sol, net, a);
                     sol.opts...)
    sol.zero
end

struct PicardIteration{T <: Real} <: FixSolver
    atol::T
    rtol::T
end

function fixvalue(sol::PicardIteration, net::FinancialModel, a)
    x = init(sol, net, a)
    y = valuation(net, x, a)
    while !isapprox(x, y;
                    atol = sol.atol, rtol = sol.rtol)
        x, y = y, valuation(net, y, a)
    end
    y
end

"""
    fixjacobian(net, a [, x])

Compute the Jacobian matrix of `fixvalue(net, a)` using the implicit
function theorem and autodiff (currently via ForwardDiff).

Note: `x` is assumed to solve x = valuation(net, x, a) if provided.
"""
function fixjacobian(net::FinancialModel, a)
    fixjacobian(net, a, fixvalue(net, a))
end

function fixjacobian(net::FinancialModel, a, x::AbstractVector)
    dVdx = ForwardDiff.jacobian(x -> valuation(net, x, a), x)
    dVda = ForwardDiff.jacobian(a -> valuation(net, x, a), a)
    (I - dVdx) \ dVda
end

"""
    numfirms(net)

Number of firms in the financial `net`.
"""
function numfirms end

"""
    nominaldebt(net)

Returns a vector of nominal debts of each firm in the financial
network `net`.
"""
function nominaldebt end

"""
    solvent(net, x)

Returns a Boolean vector indicating the solvency of each firm in the
financial network `net` in state `x`.
"""
function solvent end

"""
    equity(net, x)

Returns a vector of equity values of each firm in the financial
network `net` in state `x`.

Note: Can also be defined to extract equity part from Jacobian matrix
      (as of fixjacobian).
"""
function equity end

"""
    debt(net, x)

Returns a vector of debt values of each firm in the financial
network `net` in state `x`.

Note: Can also be defined to extract equity part from Jacobian matrix
      (as of fixjacobian).
"""
function debt end
