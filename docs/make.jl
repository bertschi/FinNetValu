push!(LOAD_PATH, "../src/")

using Documenter, FinNetValu

makedocs(sitename = "FinNetValu.jl",
         format = Documenter.HTML())

deploydocs(
    repo = "github.com/bertschi/FinNetValu.git",
    devbranch = "master",
    devurl = "stable"
)
