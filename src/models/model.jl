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
"""
function valuation! end

"""
    valuation(net, x, a)

Same as `valuation!`, but non destructive.
"""
function valuation end
# Note: needs separate implementation for autodiff
# function valuation(net::FinancialModel, x, a)
#     y = similar(x)
#     valuation!(y, net, x, a)
#     y
# end

"""
    init(net, a)

Small wrapper to generate initial conditions `x0` suitable for fixed
point solvers.
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
    fixvalue(net, a; kwargs...)

Solve for self-consistent fixed point value of model `net` for
external asset values `a`. `kwargs` are passed on to solver.
"""
function fixvalue(net::FinancialModel, a; m = 0, kwargs...)
    sol = fixedpoint(valuefunc(net, a),
                     init(net, a);
                     m = m, kwargs...)
    sol.zero
end

"""
    fixjacobian(net, a [, x])

Compute the Jacobian matrix of `fixvalue(net, a)` using the implicit
function theorem and autodiff (currently via ForwardDiff). Note
that `x` is assumed to solve the fixed point `x = valuation(net,
x, a)` which is also its default value.
"""
function fixjacobian(net::FinancialModel, a, x = fixvalue(net, a))
    dVdx = ForwardDiff.jacobian(x -> valuation(net, x, a), x)
    dVda = ForwardDiff.jacobian(a -> valuation(net, x, a), a)
    (I - dVdx) \ dVda
end

"""
    solvent(net, x)

Returns a Boolean vector indicating the solvency of each firm in the
financial network net with value x.
"""
function solvent end

"""
    numfirms(net)

Number of firms in the financial `net`.
"""
function numfirms end
