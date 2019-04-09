module BaselineAlgos

using TSML.TSMLTypes
using TSML.Utils

import TSML.TSMLTypes.fit!
import TSML.TSMLTypes.transform!

export Baseline,Identity
       fit!,
       transform!


using StatsBase: mode

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

end
