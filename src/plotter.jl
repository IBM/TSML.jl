module Plotters

import Plots
using GR
using Interact
using DataFrames

export fit!,transform!
export Plotter

import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload

using TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML.Utils

"""
    Plotter()

Plots a TS by default but performs interactive plotting if specified during instance creation.
"""
mutable struct Plotter <: Transformer
  model
  args
  function Plotter(args=Dict())
    default_args = Dict(
        :interactive => false
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(pltr::Plotter, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  typeof(features) <: DataFrame || error("Outliernicer.fit!: data should be a dataframe: Date,Val ")
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  pltr.model = pltr.args
end

"""
Convert missing into NaN for plotting discontinuity
"""
function transform!(pltr::Plotter, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  features != [] || return DataFrame()
  typeof(features) <: DataFrame || error("Outliernicer.fit!: data should be a dataframe: Date,Val ")
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  sum(names(features) .== (:Date,:Value))  == 2 || error("wrong column names")
  # covert missing to NaN
  df = deepcopy(features)
  df[:Value] = Array{Union{Missing, Float64,eltype(features[:Value])},1}(missing,nrow(df))
  df[:Value] .= features[:Value]
  ndxmissing = findall(x->ismissing(x),df[:Value])
  df[:Value][ndxmissing] .= NaN

  Plots.gr()
  if pltr.args[:interactive] == true
    @info "interactive plot: use notebook interface"
    interactiveplot(df)
  else
    @info "static plot"
    display(Plots.plot(df[:Date],df[:Value]))
  end
end

function interactiveplot(df::Union{Vector,Matrix,DataFrame})
  mlength = length(df[:Value])
  @manipulate for min in slider(1:mlength,label="min",value=1),max in slider(1:mlength,label="max",value=mlength)
    Plots.plot(df[:Date][min:max],df[:Value][min:max])
  end
end


end
