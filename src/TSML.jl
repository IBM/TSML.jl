module TSML
__precompile__(false)

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
export typerun


include("utils.jl")
using .Utils

include("dataproc.jl")
using .DataProc
export mrun
export prun

include("transformers.jl")
using .TSMLTransformers
export trfrun

include("baseline.jl")
using .BaselineAlgos
export baselinerun

if LIB_SKL_AVAILABLE # from System module
    include("scikitlearn.jl")
    using .SKLearners
    export skkrun
end

if LIB_CRT_AVAILABLE # from System module
    include("caret.jl")
    using .CaretLearners
    export caretrun
end

include("multilearner.jl")
using .MultiLearners

function testall()
    typerun()
    mrun()
    prun()
    trfrun()
    baselinerun()
    skkrun()
end

include("decisiontree.jl")
using .DecisionTreeLearners
   

end # module
