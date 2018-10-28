using FinNetValu
using LinearAlgebra
using DataFrames
using Gadfly
using Parameters
using Distributions

suzuki = XOSModel([[0.0 0.2 0.3 0.1];
                   [0.2 0.0 0.2 0.1];
                   [0.1 0.1 0.0 0.3];
                   [0.1 0.1 0.1 0.0]],
                  [[0.0 0.0 0.1 0.0];
                   [0.0 0.0 0.0 0.1];
                   [0.1 0.0 0.0 0.1];
                   [0.0 0.1 0.0 0.0]],
                  LinearAlgebra.I,
                  [0.8, 0.8, 0.8, 0.8])

Aᵉ = [2.0, 0.5, 0.6, 0.6]

println(suzuki)
println(fixvalue(suzuki, Aᵉ))

## Produce figures of default boundaries

function defaultregions(s12, s21, f12, f21)
    Mˢ = [0 s12; s21 0]
    Mᵈ = [0 f12; f21 0]
    net = XOSModel(Mˢ, Mᵈ, I, ones(2))

    Aᵉ = range(0, length = 101, stop = 2)
    df = DataFrame([Float64, Float64, Vector{Bool}],
                   [:A1, :A2, :Solvent],
                   0)
    for a1 in Aᵉ, a2 in Aᵉ
        x = fixvalue(net, [a1, a2])
        df = vcat(df,
                  DataFrame(A1 = a1, A2 = a2,
                            Solvent = string(solvent(net, x)))) 
    end
    df
end

df = defaultregions(0.2, 0.2, 0.4, 0.4)

plot(df,
     x = :A1, y = :A2, color = :Solvent,
     Geom.rectbin)

## Reproduce plot showing default spreads

function defaultspread(net, θ)
    function f(Z)
        A = Aτ(1.0, θ, Z)
        F = discount(θ) .* debtview(net, fixvalue(net, A))
    end
    F₀ = expectation(f, MonteCarloSampler(MvNormal(numfirms(net), 1.0)), 7500)
                     
    @unpack r, τ = θ
    @. 1 / τ * log( net.d / F₀ ) - r
end

B = [0.9, 0.9]
θ = BlackScholesParams(0.0, 0.2, 1.0,
                       cholesky([1 0.9; 0.9 1]).L)

exampleA = XOSModel(zeros(2, 2), zeros(2, 2), I, B)
exampleB = XOSModel([0 0.2; 0.2 0], zeros(2, 2), I, B)

## TODO: Does not look like plot from Suzuki paper!
plot(x -> defaultspread(exampleB, reconstruct(θ, τ = x))[1],
     0.0, 10.0)
                        
