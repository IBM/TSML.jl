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
	sitename = "TSML Documentation",
	authors = "Paulito P. Palmes",
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
