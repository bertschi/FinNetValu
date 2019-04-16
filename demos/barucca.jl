## Run examples from Barucca paper

using FinNetValu
using DataFrames
using Plots
using StatsPlots

function 𝕍demo()
    modelEN = EisenbergNoeModel([0.0], 2 .* ones(1,1))
    modelFurfine = FurfineModel([0.0], 2 .* ones(1,1), 0)
    modelLinearDR = LinearDebtRankModel([0.0], 2 .* ones(1,1))
    modelExAnteEN = FinNetValu.ExAnteEN_BS_Model([0.0], 2 .* ones(1, 1), 1., BlackScholesParams(0.0, 1.0, 1.0))
    
    equities = range(-3, length = 251, stop = 3)
    𝕍_EN = [modelEN.𝕍(modelEN, e, nothing)[1] for e in equities]
    𝕍_Furfine = [modelFurfine.𝕍(modelFurfine, e, nothing)[1] for e in equities]
    𝕍_LinearDR = [modelLinearDR.𝕍(modelLinearDR, e, [2.5])[1] for e in equities] # Note: A = 2.5 leads to book equity M = 2.5
    𝕍_ExAnteEN = [modelExAnteEN.𝕍(modelExAnteEN, e, [1.0])[1] for e in equities]
    
    plt = plot(equities, 𝕍_EN, label = "EN",
               xlabel = "equity of the borrower",
               ylabel = "interbank valuation function")
    plot!(plt, equities, 𝕍_Furfine, label = "Furfine")
    plot!(plt, equities, 𝕍_LinearDR, label = "Linear DR")
    plot!(plt, equities, 𝕍_ExAnteEN, label = "Ex-ante EN")
end

## TODO: Investigate why figure is not replicated!
function shockdemo()
    Lᵉ = [9., 4., 2.]
    A = [0 0.5 0;
         0 0 0.5;
         0.5 0 0]
    Aᵉ = [10., 5., 3.]
    
    modelEN  = EisenbergNoeModel(Lᵉ, A')
    modelF   = FurfineModel(Lᵉ, A', 0.0)
    modelLDR = LinearDebtRankModel(Lᵉ, A')
    modelNEVA = FinNetValu.ExAnteEN_BS_Model(Lᵉ, A', 1.0, BlackScholesParams(0.0, 1.0, 0.1))

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
                   LinearDebtRank = runshocks(modelLDR, Aᵉ, α),
                   NEVA = runshocks(modelNEVA, Aᵉ, α))

    @df df plot(:α, [:EisenbergNoe, :Furfine, :LinearDebtRank, :NEVA])
    df
end

function shockdemoNEVA()
    ## Replicate right panel of figure 2
    Lᵉ = [9., 4., 2.]
    A = [0 0.5 0;
         0 0 0.5;
         0.5 0 0]
    Aᵉ = [10., 5., 3.]

    modelNEVA = FinNetValu.ExAnteEN_BS_Model(Lᵉ, A', 1.0, BlackScholesParams(0.0, 1.0, 0.1))

    α = range(0, length = 101, stop = 0.6)
    df = DataFrame(α = Vector{Float64}(),
                   bankA = Vector{Float64}(), bankB = Vector{Float64}(), bankC = Vector{Float64}())
    for α in α
        a =  (1 - α) .* Aᵉ
        m = bookequity(modelNEVA, a)
        e = fixvalue(modelNEVA, a)
        val = modelNEVA.𝕍(modelNEVA, m, a) .- modelNEVA.𝕍(modelNEVA, e, a)
        push!(df, vcat(α, val'))
    end
    df
end
