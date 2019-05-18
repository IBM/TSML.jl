using Documenter

using TSML

makedocs(
  source = "src",
  build = "build",
  modules = [TSML],
  clean = true,
  sitename = "TSML Documentation",
  doctest = false,
  pages = Any[
    "HOME" => "index.md",
    "Tutorial" => Any[
       "tutorial/aggregators.md",
       "tutorial/pipeline.md",
       "tutorial/statistics.md",
       "tutorial/monotonic.md",
       "tutorial/tsclassifier.md"
    ],
    "Manual" => Any[
      "Date Processing" => "man/dateproc.md",
      "Value Processing" => "man/valueproc.md",
      "Aggregation" => "man/aggregation.md",
      "Imputation" => "man/imputation.md",
      "Monotonic Detection" => "man/monotonic.md",
      "TS Classification" => "man/tsclassification.md",
      "CLI Wrappers" => "man/cli.md"
    ],
    "Library" => Any[
      "Decision Tree" => "lib/decisiontree.md"
      #"Scikit Learners" => "lib/sklearn.md",
      #"Caret Learners" => "lib/caretlearn.md"
    ]
  ],
  format = Documenter.HTML(
     prettyurls = get(ENV, "CI", nothing) == "true"
  )
)

deploydocs(
    repo   = "github.com/IBM/TSML.jl.git",
)
