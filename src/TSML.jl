module TSML


include("types.jl")
using .TSMLTypes

include("utils.jl")
using .Utils

include("transformers.jl")
using .TSMLTransformers


include("baseline.jl")
using .BaselineAlgos

include("decisiontree.jl")
using .DecisionTreeLearners

include("statifier.jl")
using .Statifiers

include("monotonicer.jl")
using .Monotonicers

include("cliwrapper.jl")
using .CLIWrappers
export tsmlrun

include("tsclassifier.jl")
using .TSClassifiers

include("outliernicer.jl")
using .Outliernicers

include("plotter.jl")
using .Plotters

include("demo.jl")
using .TSMLDemo
export tsml_demo

end # module
