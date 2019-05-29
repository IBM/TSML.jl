using Documenter, DocumenterTools, DocumenterLaTeX
using TSML
using TSML.DecisionTreeLearners
using Test

const ROOT = joinpath(@__DIR__, "..")

# Documenter package docs
doc = makedocs(
   debug = true,
   root = ROOT,
   build = "pdf/build",
   modules = [TSML, DecisionTreeLearners],
   clean = false,
   format = LaTeX(platform = "docker"),
   sitename = "TSML: Time Series Machine Learning Toolbox",
   authors = "Paulito P. Palmes, Joern Ploennigs, Niall Brady",
   pages = Any[
      "HOME" => "index.md",
      "Tutorial" => Any[
      	 "tutorial/aggregators.md",
      	 "tutorial/pipeline.md",
      	 "tutorial/statistics.md",
         "tutorial/monotonic_plotting_pdf.md",
      	 "tutorial/tsclassifier.md"
       ],
      "Manual" => Any[
         "Date Processing" => "man/dateproc.md",
         "Value Processing" => "man/valueproc.md",
         "Aggregation" => "man/aggregation.md",
         "Imputation" => "man/imputation.md",
      ],
      "ML Library" => Any[
      	 "Decision Tree" => "lib/decisiontree.md",
      	 "Types and Functions" => "lib/functions.md"
      ]
   ]  
);

# hack to only deploy the actual pdf-file
mkpath(joinpath(ROOT, "pdf", "build", "pdfdir"))
let files = readdir(joinpath(ROOT, "pdf", "build"))
  for f in files
     if startswith(f, "Documenter.jl") && endswith(f, ".pdf")
        mv(joinpath(ROOT, "pdf", "build", f),
           joinpath(ROOT, "pdf", "build", "pdfdir", f))
     end
  end
end


deploydocs(
   repo = "github.com/IBM/TSML.jl.git",
   root = ROOT,
   target = "pdf/build/pdfdir",
   branch = "master",
   forcepush = true,
)
