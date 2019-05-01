module TSML

#if  "LOAD_SK_CARET" in keys(ENV) &&  ENV["LOAD_SK_CARET"] == "true" # to disable precompile for binary libs
#    __precompile__(false)
#elseif "LOAD_SK_CARET" in keys(ENV) &&  ENV["LOAD_SK_CARET"] == "false"
#    __precompile__(true) # no binary libs
#else
#    __precompile__(false) # assume default has binary libs
#end

__precompile__(false) # assume default has binary libs

export greet
export testall
export mrun,prun # from DataProc
export mergedict
export multirun
export matrifyrun, dateifierrun
export datevalgatorrun, datevalizerrun, datevalnnerrun
using Dates

greet() = print("Hello World!")

include("system.jl")
using .System

include("types.jl")
using .TSMLTypes
#export typerun


include("utils.jl")
using .Utils

include("dataproc.jl")
using .DataProc
#export mrun
#export prun

include("transformers.jl")
using .TSMLTransformers

include("baseline.jl")
using .BaselineAlgos
#export baselinerun

if LIB_SKL_AVAILABLE # from System module
    include("scikitlearn.jl")
    using .SKLearners
    #export skkrun
end

if LIB_CRT_AVAILABLE # from System module
    include("caret.jl")
    using .CaretLearners
    #export caretrun
end

include("multilearner.jl")
using .MultiLearners

include("decisiontree.jl")
using .DecisionTreeLearners

include("datareader.jl")
using .DataReaders
#export datareaderrun

include("datawriter.jl")
using .DataWriters
#export datawriterrun

include("statifier.jl")
using .Statifiers
#export fullstat, statifierrun

include("monotonicer.jl")
using .Monotonicers
#export monotonicerrun

include("cliwrapper.jl")
using .CLIWrappers
export tsmlrun


end # module
