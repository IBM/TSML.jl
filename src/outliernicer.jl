module Outliernicers

using Dates
using DataFrames
using Random
using Statistics
using StatsBase: iqr, quantile, sample

export fit!,transform!
export Outliernicer

import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload

using TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML.Utils

"""
    Outliernicer(Dict())

Detects outliers below or above (q25-iqr,q75+iqr)
and replace them with missing so that ValNNer can
use nearest neighbors to replace the missings.
"""
mutable struct Outliernicer <: Transformer
  model
  args

  function Outliernicer(args=Dict())
    default_args = Dict(
        :dateinterval => Dates.Hour(1),
        :nnsize => 1,
        :missdirection => :symmetric
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(st::Outliernicer, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  typeof(features) <: DataFrame || error("Outliernicer.fit!: data should be a dataframe: Date,Val ")
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  st.model = st.args
end

function transform!(st::Outliernicer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  features != [] || return DataFrame()
  typeof(features) <: DataFrame || error("Outliernicer.fit!: data should be a dataframe: Date,Val ")
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  sum(names(features) .== (:Date,:Value))  == 2 || error("wrong column names")
  mfeatures=deepcopy(features)
  rvals = mfeatures[:Value]
  # compute the outlier range
  # setup to store both missing and numbers
  mvals = Array{Union{Missing,eltype(rvals)},1}(missing,length(rvals))
  mvals .= rvals
  crvals = skipmissing(rvals) # stat of non-missing
  miqr = iqr(crvals)
  q25,q75 = quantile(crvals,[0.25,0.75])
  lower=q25-miqr; upper=q75+miqr
  missindx = findall(x -> !ismissing(x) && (x > upper || x < lower),rvals) 
  mvals[missindx] .= missing
  mfeatures[:Value] = mvals
  # use ValNNer to replace missings
  valnner = DateValNNer(st.args)
  fit!(valnner,mfeatures)
  resdf = transform!(valnner,mfeatures)
  resdf[:Value] = collect(skipmissing(resdf[:Value])) 
  resdf
end

end
