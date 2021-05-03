module FinNetValu

include("utils/utils.jl")
include("utils/nets.jl")
include("pricing/bs.jl")
include("pricing/mc.jl")
include("models/model.jl")
include("models/neva.jl")
include("models/xos.jl")

# Generic financial model interface
export fixvalue, fixjacobian, valuation!, valuation
export numfirms, nominaldebt, solvent, equity, debt
# Model constructors
export XOSModel, NEVAModel, EisenbergNoeModel, FurfineModel, LinearDebtRankModel, ExAnteEN_BS_Model
# Solution methods
export NLSolver, PicardIteration

# Pricing helpers
export BlackScholesParams, Aτ, discount
# Monte-Carlo helpers
export MonteCarloSampler, sample, expectation

# Functional utils
export constantly, calm
# Network utils
export erdosrenyi, rescale, barabasialbert


end # module
