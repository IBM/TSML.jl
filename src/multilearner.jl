module MultiLearners

export MultiLearner, transform!, fit!
export multirun

using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils: mergedict
using TSML.System # import LIB_SKL_AVAILABLE

if LIB_SKL_AVAILABLE
    using TSML.SKLearners
end

mutable struct MultiLearner <: TSLearner
    model
    args
    data

    function MultiLearner(args=Dict())
        default_args=Dict(
            :output => :numeric,
            :learner => "RandomForestRegressor",
            :impl_args => Dict()
        )
        new(nothing,mergedict(default_args,args),nothing)
    end
end

function fit!(mlt::MultiLearner, x::T,y::Vector) where {T<:Union{Matrix,Vector}}
    @info "fit: to be implemented"
    @info "matrify x"
    @info "fit(matrified x, y)"
end

function transform!(mlt::MultiLearner,x::T) where {T<:Union{Matrix,Vector}}
    @info "transform: to be implemented"
end

function multirun()
    model = MultiLearner()
    println(model)
end

end
