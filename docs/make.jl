using Documenter, TSML

using TSML.DecisionTreeLearners

makedocs(modules = [TSML,DecisionTreeLearners],
   clean = false,
   sitename = "TSML Documentation",
   pages = Any[
      "HOME" => "index.md",
      "Tutorial" => Any[
              "tutorial/aggregators.md",
              "tutorial/pipeline.md",
              "tutorial/statistics.md",
              "tutorial/tsdetectors.md"
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
