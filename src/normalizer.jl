module Normalizers

using StatsBase
using Dates
using DataFrames: DataFrame
using Statistics
using MultivariateStats
using Random

using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils
import AutoMLPipeline.AbsTypes: fit!, transform!

export fit!,transform!
export Normalizer

const MV=MultivariateStats


"""
    Normalizer(Dict(
       :method => :zscore
    ))


Transforms continuous features into normalized form such as zscore, unitrange, square-root, log, pca, ppca
with parameter: 

- `:method` => `:zscore` or `:unitrange` or `:sqrt` or `:log` or `pca` or `ppca` or `fa`
- :zscore => standard z-score with centering and scaling
- :unitrange => unit range normalization with centering and scaling
- :sqrt => square-root transform
- :pca => principal component analysis transform
- :ppca => probabilistic pca
- :fa => factor analysis
- :log => log transform

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
  model
  args

  function Normalizer(args=Dict())
    default_args = Dict(
        :method => :zscore
    )
    new(nothing, mergedict(default_args, args))
  end
end

function Normalizer(st::Symbol)
  Normalizer(Dict(:method=>st))
end

"""
    fit!(st::Statifier, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}

Validate argument features other than dates are continuous.
"""
function fit!(norm::Normalizer, features::DataFrame, labels::Vector=[]) 
  # check features are in correct format and no categorical values
  (infer_eltype(features[:,1]) <: DateTime && infer_eltype(Matrix(features[:,2:end])) <: Real) || 
    (infer_eltype(Matrix(features)) <: Real) || 
    error("Normalizer.fit!: make sure features are purely float values or float values with Date on first column")
  norm.model = norm.args
end

"""
    transform!(norm::Normalizer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Compute statistics.
"""
function transform!(norm::Normalizer, pfeatures::DataFrame)
  pfeatures != DataFrame() || return DataFrame()
  res = Array{Float64,2}(undef,0,0)
  if (infer_eltype(pfeatures[:,1]) <: DateTime && infer_eltype(Matrix(pfeatures[:,2:end])) <: Real)
    res = processnumeric(norm,Matrix{Float64}(pfeatures[:,2:end])) |> DataFrame
  elseif infer_eltype(Matrix(pfeatures)) <: Real
    features = pfeatures |> Array{Float64}
    res = processnumeric(norm,features) |> DataFrame
  else
    error("Normalizer.transform!: make sure features are purely float values or float values with Date on first column")
  end
  res
end

function processnumeric(norm::Normalizer,features::Matrix)
  if norm.args[:method] == :zscore
    ztransform(features)
  elseif norm.args[:method] == :unitrange
    unitrtransform(features)
  elseif norm.args[:method] == :pca
    pca(features)
  elseif norm.args[:method] == :ppca
    ppca(features)
  elseif norm.args[:method] == :ica
    ica(features)
  elseif norm.args[:method] == :fa
    fa(features)
  elseif norm.args[:method] == :sqrt
    sqrtf(features)
  elseif norm.args[:method] == :log
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
  fit(ZScoreTransform, xp,dims=2; center=true, scale=true) |> dt -> StatsBase.transform(dt,xp)' |> collect
end

# unit-range
function unitrtransform(X)
  xp = X' |> collect |> Matrix{Float64}
  fit(UnitRangeTransform,xp,dims=2) |> dt -> StatsBase.transform(dt,xp)' |> collect
end

# pca
function pca(X)
  xp = X' |> collect |> Matrix{Float64}
  m = MV.fit(PCA,xp)
  MV.transform(m,xp)' |> collect
end


function ica(X,kk::Int=0)
  k = kk
  if k == 0
    k = size(X)[2]
  end
  xp = X' |> collect |> Matrix{Float64}
  m = MV.fit(ICA,xp,k)
  MV.transform(m,xp)' |> collect
end


# ppca
function ppca(X)
  xp = X' |> collect |> Matrix{Float64}
  m = MV.fit(PPCA,xp)
  MV.transform(m,xp)' |> collect
end

# fa
function fa(X)
  xp = X' |> collect |> Matrix{Float64}
  m = MV.fit(FactorAnalysis,xp)
  MV.transform(m,xp)' |> collect
end

end
