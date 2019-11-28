@reexport module Schemalizers

using Random
using Dates
using DataFrames
using JuliaDB: ML, table

export fit!,transform!
export Schemalizer

import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload

using TSML.TSMLTypes
using TSML.Utils

"""
    Schemalizer(
       Dict(
           :mlschema => ML.Schema()
       )
    )

Plots a TS by default but performs interactive plotting if specified during instance creation.

Example:


Implements: `fit!`, `transform!`
"""
mutable struct Schemalizer <: Transformer
  model
  args
  function Schemalizer(args=Dict())
    default_args = Dict(
        :mlschema => ML.Schema(),
        :hints => Dict{Symbol,Any}()
    )
    margs=mergedict(default_args, args)
    new(nothing,margs)
  end
end

"""
    fit!(sch::Schemalizer, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}

Check validity of `features`: Date,Val data or just Vals Matrix
"""
function fit!(sch::Schemalizer, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  typeof(features) <: DataFrame || error("Schemalizer.fit!: data should be a dataframe")
  isempty(features) && error("Schemalizer.fit: data format not recognized.")
  isempty(sch.args[:mlschema]) && (sch.args[:mlschema] = ML.schema(table(features),hints=sch.args[:hints]))
  sch.model = sch.args
end

"""
    transform!(sch::Schemalizer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Normalized continous features and hot-bit encode categorical features
"""
function transform!(sch::Schemalizer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  isempty(features) && (return DataFrame())
  typeof(features) <: DataFrame || error("Schemalizer.transform!: data should be a dataframe")
  ML.featuremat(sch.args[:mlschema],table(features))' |> DataFrame
end


function generatedf()
  Random.seed!(123)
  gdate = DateTime(2015,1,1):Dates.Minute(30):DateTime(2015,1,2) |> collect
  len = length(gdate)
  sex = rand(['m','f'],len)
  x1=rand(1:100,len)
  x2=rand(1:100,len)
  x3=rand(1:1000,len)
  DataFrame(date=gdate,sex=sex,f1=x1,f2=x2,f3=x3)
end

function schemadriver()
  df = generatedf()
  m = Schemalizer(Dict(:hints=>
      Dict(:sex => ML.Categorical)))
  fit!(m,df)
  transform!(m,df)
end

end
