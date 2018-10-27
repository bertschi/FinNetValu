########################################
# Reproduce some of the Greeks figures #
########################################

using FinNetValu
using ForwardDiff
using Distributions
using Gadfly
using LinearAlgebra
using DataFrames

## TODO: Move to network helpers in library
function erdosrenyi(N::Integer, k::Real)
    p = k / N
    A = collect(rand() < p ? 1.0 : 0.0
                for i = 1:N, j = 1:N)
    for i = 1:N
        A[i, i] = 0.0
    end
    A
end

function rescale(A::Matrix, w::Real)
    B = similar(A)
    for j = 1:size(A, 1)
        s = sum(A[:, j])
        if s > 0
            B[:, j] = A[:, j] .* w ./ s
        else
            B[:, j] = A[:, j]
        end
    end
    B
end

## Recompute Delta for wᵈ = 0.6 and <k> = 3

function delta(net, a₀, θ, Z)
    A = Aτ(a₀, θ, Z)
    x = fixvalue(net, A)
    dVdA = fixjacobian(net, A, x)
    dAda₀ = ForwardDiff.jacobian(a -> Aτ(a, θ, Z), a₀)
    dVda₀ = discount(θ) * dVdA * dAda₀

    N = numfirms(net)
    deltaEq = sum(dVda₀[FinNetValu._eqidx(net), :]) / N
    deltaDb = sum(dVda₀[FinNetValu._dbidx(net), :]) / N
    [deltaEq, deltaDb, deltaEq + deltaDb]
end

θ = BlackScholesParams(0.0, 1.0, 0.4)
N = 60
net = XOSModel(zeros(N, N), rescale(erdosrenyi(N, 3.0), 0.4),
               I, ones(N))

a₀ = range(0, length = 31, stop = 2.0)
@time Δ = collect(expectation(z -> delta(net, a * ones(N), θ, z),
                              MonteCarloSampler(MvNormal(N, 1.0)),
                              1000)
                  for a in a₀)

plot(stack(DataFrame(a₀= a₀,
                     Equity = map(x -> x[1], Δ),
                     Debt = map(x -> x[2], Δ),
                     Value = map(x -> x[3], Δ)),
           [:Equity, :Debt, :Value]),
     x = :a₀, y = :value, color = :variable,
     Geom.line)
