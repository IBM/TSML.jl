module BaselineAlgos

using DataFrames

using ..AbsTypes
using ..Utils

import ..AbsTypes: fit!, transform!
export fit!,transform!

export Baseline,Identity

using StatsBase: mode


"""
    Baseline(
       default_args = Dict(
          :output => :class,
          :strat => mode
       )
    )

Baseline model that returns the mode during classification.
"""
mutable struct Baseline <: Learner
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

"""
    fit!(bsl::Baseline,x::Matrix,y::Vector)

Get the mode of the training data.
"""
function fit!(bsl::Baseline,x::DataFrame,y::Vector=[])
    bsl.model = bsl.args[:strat](y)
end

"""
    transform!(bsl::Baseline,x::Matrix)

Return the mode in classification.
"""
function transform!(bsl::Baseline,x::DataFrame)::Vector{<:Any}
    fill(bsl.model,size(x,1))
end

"""
    Identity(args=Dict())

Returns the input as output.
"""
mutable struct Identity <: Transformer
    model
    args

    function Identity(args=Dict())
        default_args = Dict{Symbol,Any}()
        new(nothing,mergedict(default_args,args))
    end
end

"""
    fit!(idy::Identity,x::Matrix,y::Vector)
    
Does nothing.
"""
function fit!(idy::Identity,x::DataFrame,y::Vector=[])
    nothing
end

"""
    transform!(idy::Identity,x::Matrix)

Return the input as output.
"""
function transform!(idy::Identity,x::DataFrame)::DataFrame
    return x
end

end
