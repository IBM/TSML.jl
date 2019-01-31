module TSML
__precompile__(false)

export greet
export testall
export mrun,prun # from DataProc 
export mergedict


greet() = print("Hello World!")

include("types.jl")
using .TSMLTypes
export typerun
export  Transformer,TSLearner,fit!,transform!


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

include("scikitlearn.jl")
using .SKLearners
export skkrun

include("caret.jl")
using .CaretLearners
export caretrun

function testall()
    typerun()
    mrun()
    prun()
    trfrun()
    baselinerun()
    skkrun()
end
   

end # module
