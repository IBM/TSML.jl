module TSML
using Reexport

# reexport common functions to Main
@reexport using CSV
@reexport using Dates
@reexport using DataFrames
@reexport using Random
@reexport using Statistics


include("types.jl")
using .TSMLTypes

include("utils.jl")
using .Utils

include("transformers.jl")
using .TSMLTransformers

include("baseline.jl")
using .BaselineAlgos

include("mlbase.jl")
using .MLBaseWrapper

include("decisiontree.jl")
using .DecisionTreeLearners

include("statifier.jl")
using .Statifiers

include("monotonicer.jl")
using .Monotonicers

include("cliwrapper.jl")
using .CLIWrappers

include("tsclassifier.jl")
using .TSClassifiers

include("outliernicer.jl")
using .Outliernicers

include("plotter.jl")
using .Plotters

include("timescaledb.jl")
using .TimescaleDBs

include("demo.jl")
using .TSMLDemo

include("argparse.jl")
using .ArgumentParsers

include("ensemble.jl")
using .EnsembleMethods

end # module
