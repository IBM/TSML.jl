@reexport module Monotonicers

using Dates
using DataFrames
using Statistics

export fit!,transform!,ismonotonic,dailyflips
export Monotonicer

using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils


# Transforms instances with nominal features into one-hot form
# and coerces the instance matrix to be of element type Float64.
mutable struct Monotonicer <: Transformer
  model
  args

  function Monotonicer(args=Dict())
    default_args = Dict(
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(st::Monotonicer, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  typeof(features) <: DataFrame || error("Monotonicer.fit!: data should be a dataframe: Date,Val ")
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  st.model = st.args
end

function transform!(st::Monotonicer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  features != [] || return DataFrame()
  typeof(features) <: DataFrame || error("Monotonicer.fit!: data should be a dataframe: Date,Val ")
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  sum(names(features) .== (:Date,:Value))  == 2 || error("wrong column names")
  mfeatures=features
  # double check monotonic 
  # based on flips and daily flips
  if ismonotonic(features.Value)
    mfeatures = antimonotonize(features)
  end
  # daily mono reset everynight at most 2x
  ndailyflips = dailyflips(mfeatures)
  if ndailyflips < 0.5
    antimono = antimonotonize(mfeatures)
  elseif 0.5 <= ndailyflips <= 2.0
    antimono = antimonotonizedaily(mfeatures)
  else
    antimono = mfeatures
  end
  return antimono
end

function antimonotonizedaily(dat::DataFrame)
  # get diffs, negative values represent reset
  # replace negative values by succeeding neighbor
  df = deepcopy(dat)
  df.Value .= [ -1.0; diff(dat.Value) ]
  valuelength = length(df.Value)
  # find locations of negatives
  negatives = findall(x->x<0,df.Value)
  # if last element is negative, zero it
  if negatives[end] == valuelength
    df.Value[negative[end]] = 0.0
    pop!(negatives)
  end
  # replace negatives with immediate neighbor
  for val in reverse(negatives)
    df.Value[val] = df.Value[val+1]
  end
  return df
end

function antimonotonize(data::Union{Vector,DataFrame,Matrix})
  dat = deepcopy(data)
  typeof(dat) <: DataFrame || error("input must be a dataframe")
  ncol(dat) == 2 || error("input must have two columns")
  sum(names(dat) .== (:Date,:Value))  == 2 || error("wrong column names")
  vals = dat.Value
  fvals = diff(vals)
  dvals = [fvals[1];fvals]
  dat.Value = dvals
  return dat
end

function dailyflips(dat::DataFrame)
  dates = dat.Date
  values = dat.Value
  datemin = minimum(dates)
  datemax = maximum(dates)
  ndays = round(datemax-datemin,Dates.Day(1))
  nflips::Float64 = sum((skipmissing(values) |> collect |> diff) .< 0 )
  avg::Float64 = nflips/ndays.value
end

function ismonotonic(dat::Vector{T}) where T
  #check if monotonic increasing
  cdata=dat |> skipmissing |> collect
  maxflips = minimum([10;div(length(cdata),2)]) # flip cutoff to identify monotonic
  nflips = length(cdata) - sum(cdata .== accumulate(max,cdata))
  nflips < maxflips ? true : false
end

end
