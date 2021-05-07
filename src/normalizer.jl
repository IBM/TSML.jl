module Normalizers

import StatsBase
using StatsBase: zscore, ZScoreTransform,UnitRangeTransform
using Dates
using DataFrames: DataFrame
using Statistics
import MultivariateStats
using Random

using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!,transform, transform!
export Normalizer

const MV=MultivariateStats

const gmethods = [:zscore,:unitrange,:sqrt,:log,:pca,:ppca,:fa]

"""
    Normalizer(Dict(
       :method => :zscore
    ))


Transforms continuous features into normalized form such as zscore, unitrange, square-root, log, pca, ppca
with parameter: 

- `:method` => `:zscore` or `:unitrange` or `:sqrt` or `:log` or `pca` or `ppca` or `fa`
- `:zscore` => standard z-score with centering and scaling
- `:unitrange` => unit range normalization with centering and scaling
- `:sqrt` => square-root transform
- `:pca` => principal component analysis transform
- `:ppca` => probabilistic pca
- `:fa` => factor analysis
- `:log` => log transform

Example:
    
    function generatedf()
        Random.seed!(123)
        gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
        gval1 = rand(length(gdate))
        gval2 = rand(length(gdate))
        gval3 = rand(length(gdate))
        X = DataFrame(Date=gdate,Value1=gval1,Value2=gval2,Value3=gval3)
        X
    end

    X = generatedf()
    norm = Normalizer(Dict(:method => :zscore))
    fit!(norm,X)
    res=transform!(norm,X)

Implements: `fit!`, `transform!`
"""
mutable struct Normalizer <: Transformer
   name::String
   model::Dict{Symbol,Any}

   function Normalizer(args=Dict())
      default_args = Dict(
         :name => "nrmlzr",
         :method => :zscore
      )
      cargs=nested_dict_merge(default_args,args)
      cargs[:method] in gmethods || throw(ArgmentError("invalid method")) 
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

function Normalizer(st::Symbol)
   Normalizer(Dict(:method=>st))
end

"""
    fit!(st::Statifier, features::T, labels::Vector=[])

Validate argument features other than dates are continuous.
"""
function fit!(norm::Normalizer, features::DataFrame, labels::Vector=[])::Nothing
   # check features are in correct format and no categorical values
   (infer_eltype(features[:,1]) <: DateTime && infer_eltype(Matrix(features[:,2:end])) <: Real) || 
      (infer_eltype(Matrix(features)) <: Real) || 
      throw(ArgmentError("Normalizer.fit!: make sure features are purely float values or float values with Date on first column"))
      return nothing
end

function fit(norm::Normalizer, features::DataFrame, labels::Vector=[])::Normalizer
   fit!(norm,features,labels)
   return deepcopy(norm)
end

"""
    transform!(norm::Normalizer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Compute statistics.
"""
function transform!(norm::Normalizer, pfeatures::DataFrame)::DataFrame
   pfeatures != DataFrame() || return DataFrame()
   res = Array{Float64,2}(undef,0,0)
   if (infer_eltype(pfeatures[:,1]) <: DateTime && infer_eltype(Matrix(pfeatures[:,2:end])) <: Real)
      res = processnumeric(norm,Matrix{Float64}(pfeatures[:,2:end])) |> x->DataFrame(x,:auto)
   elseif infer_eltype(Matrix(pfeatures)) <: Real
      features = pfeatures |> Array{Float64}
      res = processnumeric(norm,features) |> x->DataFrame(x,:auto)
   else
      error("Normalizer.transform!: make sure features are purely float values or float values with Date on first column")
   end
   res
end

function transform(norm::Normalizer, pfeatures::DataFrame)::DataFrame
   return transform!(norm,pfeatures)
end

function processnumeric(norm::Normalizer,features::Matrix)
  if norm.model[:method] == :zscore
    ztransform(features)
  elseif norm.model[:method] == :unitrange
    unitrtransform(features)
  elseif norm.model[:method] == :pca
    pca(features)
  elseif norm.model[:method] == :ppca
    ppca(features)
  elseif norm.model[:method] == :ica
    ica(features)
  elseif norm.model[:method] == :fa
    fa(features)
  elseif norm.model[:method] == :sqrt
    sqrtf(features)
  elseif norm.model[:method] == :log
    logf(features)
  else
    error("arg's :method is mapped to unknown keyword")
  end
end

# apply sqrt transform
function sqrtf(X)
   sqrt.(X)
end


# apply log transform
function logf(X)
   log.(X)
end

# apply z-score transform
function ztransform(X)
   xp = X' |> collect |> Matrix{Float64}
   StatsBase.fit(ZScoreTransform, xp,dims=2; center=true, scale=true) |> dt -> StatsBase.transform(dt,xp)' |> collect
end

# unit-range
function unitrtransform(X)
   xp = X' |> collect |> Matrix{Float64}
   StatsBase.fit(UnitRangeTransform,xp,dims=2) |> dt -> StatsBase.transform(dt,xp)' |> collect
end

# pca
function pca(X)
   xp = X' |> collect |> Matrix{Float64}
   m = MV.fit(MV.PCA,xp)
   MV.transform(m,xp)' |> collect
end

function ica(X,kk::Int=0)
   k = kk
   if k == 0
      k = size(X)[2]
   end
   xp = X' |> collect |> Matrix{Float64}
   m = MV.fit(MV.ICA,xp,k)
   MV.transform(m,xp)' |> collect
end


# ppca
function ppca(X)
   xp = X' |> collect |> Matrix{Float64}
   m = MV.fit(MV.PPCA,xp)
   MV.transform(m,xp)' |> collect
end

# fa
function fa(X)
   xp = X' |> collect |> Matrix{Float64}
   m = MV.fit(MV.FactorAnalysis,xp)
   MV.transform(m,xp)' |> collect
end

end
