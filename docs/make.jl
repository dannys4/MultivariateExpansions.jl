using MultivariateExpansions
using Documenter

DocMeta.setdocmeta!(MultivariateExpansions, :DocTestSetup, :(using MultivariateExpansions); recursive=true)

makedocs(;
    modules=[MultivariateExpansions],
    authors="Daniel Sharp <dannys4@mit.edu> and contributors",
    sitename="MultivariateExpansions.jl",
    format=Documenter.HTML(;
        canonical="https://dannys4.github.io/MultivariateExpansions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dannys4/MultivariateExpansions.jl",
    devbranch="main",
)
