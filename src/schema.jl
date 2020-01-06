module Schemalizers

using Random
using Dates
using DataFrames
using JuliaDB: ML, table

using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils

export fit!,transform!
export Schemalizer, ML, table


"""
    Schemalizer(
       Dict(
           :schema => ML.Schema()
       )
    )

Automatically converts continuous features into zscore and categorical values into hot-bin encoding.

Example:

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
    
    df = generatedf()
    m = Schemalizer(Dict(:schema=>Dict(:sex => ML.Categorical)))
    fit!(m,df)
    res=transform!(m,df)
    @assert isapprox(sum(Matrix(res.^2)),193,atol=1e-2)
    m = Schemalizer(Dict())
    fit!(m,df)
    res = transform!(m,df)
    @assert isapprox(sum(Matrix(res.^2)),144,atol=1e-2)


Implements: `fit!`, `transform!`
"""
mutable struct Schemalizer <: Transformer
  model
  args
  function Schemalizer(args=Dict())
    default_args = Dict(
        :schema => ML.Schema(),
    )
    margs=mergedict(default_args, args)
    new(nothing,margs)
  end
end

"""
    fit!(sch::Schemalizer, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}

Check validity of `features`: Date,Val data or just Vals Matrix
"""
function fit!(sch::Schemalizer, features::DataFrame, labels::Vector=[])
  isempty(features) && error("Schemalizer.fit: data format not recognized.")
  if isempty(sch.args[:schema]) 
    sch.args[:schema] = ML.schema(table(features))  
  else 
    sch.args[:schema] = ML.schema(table(features),hints=sch.args[:schema])
  end
  sch.model = sch.args
end

"""
    transform!(sch::Schemalizer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Normalized continous features and hot-bit encode categorical features
"""
function transform!(sch::Schemalizer, features::DataFrame) 
  isempty(features) && (return DataFrame())
  ML.featuremat(sch.args[:schema],table(features))' |> DataFrame
end


end
