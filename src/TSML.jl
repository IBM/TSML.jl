module TSML

include("system.jl")
using .System

include("types.jl")
using .TSMLTypes

include("utils.jl")
using .Utils

include("dataproc.jl")
using .DataProc

include("transformers.jl")
using .TSMLTransformers

include("baseline.jl")
using .BaselineAlgos

if LIB_SKL_AVAILABLE # from System module
    include("scikitlearn.jl")
    using .SKLearners
end

if LIB_CRT_AVAILABLE # from System module
    include("caret.jl")
    using .CaretLearners
end

include("multilearner.jl")
using .MultiLearners

include("decisiontree.jl")
using .DecisionTreeLearners

include("datareader.jl")
using .DataReaders

include("datawriter.jl")
using .DataWriters

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

end # module
