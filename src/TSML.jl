module TSML
using Reexport

# reexport common functions to Main
@reexport using CSV
@reexport using Dates
@reexport using DataFrames
@reexport using Random


include("types.jl")
@reexport using .TSMLTypes

include("utils.jl")
@reexport using .Utils

include("transformers.jl")
@reexport using .TSMLTransformers

include("baseline.jl")
@reexport using .BaselineAlgos

include("mlbase.jl")
@reexport using .MLBaseWrapper

include("decisiontree.jl")
@reexport using .DecisionTreeLearners

include("statifier.jl")
@reexport using .Statifiers

include("monotonicer.jl")
@reexport using .Monotonicers

include("cliwrapper.jl")
@reexport using .CLIWrappers
export tsmlrun

include("tsclassifier.jl")
@reexport using .TSClassifiers

include("outliernicer.jl")
@reexport using .Outliernicers

include("plotter.jl")
@reexport using .Plotters

include("timescaledb.jl")
@reexport using .TimescaleDBs

include("demo.jl")
@reexport using .TSMLDemo
export tsml_demo

include("argparse.jl")
@reexport using .ArgumentParsers

include("ensemble.jl")
@reexport using .EnsembleMethods

end # module
