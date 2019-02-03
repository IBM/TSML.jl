module BaselineAlgos

using TSML.TSMLTypes
using TSML.Utils

import TSML.TSMLTypes.fit!
import TSML.TSMLTypes.transform!

export baselinerun
export Baseline,
       fit!,
       transform!


using StatsBase: mode
using RDatasets

mutable struct Baseline <: TSLearner
    model
    args

    function Baseline(args=Dict())
        default_args = Dict(
            :output => :class,
            :strat => mode
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(bsl::Baseline,x::Matrix,y::Vector)
    bsl.model = bsl.args[:strat](y)
end

function transform!(bsl::Baseline,x::Matrix)
    fill(bsl.model,size(x,1))
end

mutable struct Identity <: Transformer
    model
    args

    function Identity(args=Dict())
        default_args = Dict{Symbol,Any}()
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(idy::Identity,x::Matrix,y::Vector)
    nothing
end

function transform!(idy::Identity,x::Matrix)
    return x
end



function baselinerun()
    iris=dataset("datasets","iris") 
    instances=iris[:,1:4] |> Matrix
    labels=iris[:,5] |> Vector
    bl = Baseline()
    show(fit!(bl,instances,labels));println()
    show(transform!(bl,instances));println()
    idy = Identity()
    show(fit!(idy,instances,labels));println()
    show(transform!(idy,instances));println()
end

end
