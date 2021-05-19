module FinNetValu

include("utils/utils.jl")
include("utils/nets.jl")
include("pricing/bs.jl")
include("pricing/mc.jl")
include("models/model.jl")
include("models/neva.jl")
include("models/xos.jl")
include("models/rv.jl")

# Generic financial model interface
export fixvalue, fixjacobian, valuation!, valuation
export numfirms, nominaldebt, solvent, debtequity
# Model types and constructors
export
    FinancialModel, DefaultModel,
    XOSModel,
    NEVAModel, EisenbergNoeModel, RogersVeraartModel, FurfineModel, LinearDebtRankModel, ExAnteEN_BS_Model,
    RVOrigModel, RVEqModel
# Solution methods
export
    NLSolver, PicardIteration,
    GCVASolver

# Pricing helpers
export BlackScholesParams, Aτ, discount
# Monte-Carlo helpers
export MonteCarloSampler, sample, expectation

# Functional utils
export constantly, calm
# Network utils
export erdosrenyi, rescale, barabasialbert


end # module
