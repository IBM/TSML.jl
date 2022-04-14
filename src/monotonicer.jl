module Monotonicers

using Random
using Dates
using DataFrames
using Statistics

using ..AbsTypes
using ..Utils
import ..AbsTypes: fit, fit!, transform, transform!

export fit, fit!, trasnform, transform!
export Monotonicer,ismonotonic,dailyflips

# Transforms instances with nominal features into one-hot form
# and coerces the instance matrix to be of element type Float64.

"""
    Monotonicer()

Monotonic filter to detect and normalize two types of dataset: 
- daily monotonic 
- entirely non-decreasing/non-increasing data

Example: 

    fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
    csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
    valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
    valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
    stfier = Statifier(Dict(:processmissing=>true))
    mono = Monotonicer(Dict())
    
    mypipeline = @pipeline csvfilter |> valgator |> mono |> stfier
    result = fit_transform!(mypipeline)


Implements: `fit!`, `transform!`

"""
mutable struct Monotonicer <: Transformer
   name::String
   model::Dict{Symbol,Any}

  function Monotonicer(args=Dict())
    default_args = Dict(
       :name => "mntncr"
    )
    cargs=nested_dict_merge(default_args,args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],cargs)
 end
end

"""
    fit!(st::Monotonicer,features::T, labels::Vector=[])

A function that checks if `features` are two-column data of  Dates and Values
"""
function fit!(st::Monotonicer, features::DataFrame, labels::Vector=[])::Nothing
   ncol(features) == 2 || throw(ArgumentError("dataframe must have 2 columns: Date, Val"))
   return nothing
end

function fit(st::Monotonicer, features::DataFrame, labels::Vector=[])::Monotonicer
   fit!(st,features,labels)
   return deepcopy(st)
end

"""
    transform!(st::Monotonicer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Normalize monotonic or daily monotonic data by taking the diffs and counting the flips.
"""
function transform!(st::Monotonicer, features::DataFrame)
   features != DataFrame() || return DataFrame()
   ncol(features) == 2 ||  throw(ArgumentError("dataframe must have 2 columns: Date, Val"))
   sum(names(features) .== ("Date","Value"))  == 2 || throw(ArgumentError("wrong column names"))
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

function transform(st::Monotonicer, features::DataFrame)::DataFrame
   return transform!(st,features)
end

function antimonotonizedaily(dat::DataFrame)
  # get diffs, negative values represent reset
  # replace negative values by succeeding neighbor
  df = deepcopy(dat)
  df[:,:Value] .= [ -1.0; diff(dat.Value) ]
  valuelength = length(df.Value)
  # find locations of negatives
  negatives = findall(x->x<0,df.Value)
  # if last element is negative, zero it
  if negatives[end] == valuelength
    df.Value[negatives[end]] = 0.0
    pop!(negatives)
  end
  # replace negatives with immediate neighbor
  for val in reverse(negatives)
    df.Value[val] = df.Value[val+1]
  end
  return df
end

function antimonotonize(data::DataFrame)
  dat = deepcopy(data)
  typeof(dat) <: DataFrame || error("input must be a dataframe")
  ncol(dat) == 2 || error("input must have two columns")
  sum(names(dat) .== ("Date","Value"))  == 2 || error("wrong column names")
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

function ismonotonic(dat::Vector)
  #check if monotonic increasing
  cdata=dat |> skipmissing |> collect
  maxflips = minimum([10;div(length(cdata),2)]) # flip cutoff to identify monotonic
  nflips = length(cdata) - sum(cdata .== accumulate(max,cdata))
  nflips < maxflips ? true : false
end

end
