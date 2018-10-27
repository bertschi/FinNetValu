## Run examples from Barucca paper

using FinNetValu
using DataFrames
using Gadfly

Lᵉ = [9., 4., 2.]
A = [0 0.5 0;
     0 0 0.5;
     0.5 0 0]
Aᵉ = [10., 5., 3.]

modelEN  = EisenbergNoeModel(Lᵉ, A')
modelF   = FurfineModel(Lᵉ, A', 0.0)
modelLDR = LinearDebtRankModel(Lᵉ, A')

function runshocks(model, Aᵉ, shocks)
    E₀ = fixvalue(model, Aᵉ)
    ΔE(α)  = E₀ .- fixvalue(model, (1 - α) .* Aᵉ)
    ΔAᵉ(α) = α .* Aᵉ
    [sum(ΔE(α) .- ΔAᵉ(α)) / sum(model.A) for α in shocks]
end

α = range(0, length = 101, stop = 0.6)

df = DataFrame(α = α,
               EisenbergNoe = runshocks(modelEN, Aᵉ, α),
               Furfine = runshocks(modelF, Aᵉ, α),
               LinearDebtRank = runshocks(modelLDR, Aᵉ, α))

plot(stack(df,
           [:EisenbergNoe, :Furfine, :LinearDebtRank],
           variable_name = :model),
     x = :α, y = :value, color = :model,
     Geom.line)
