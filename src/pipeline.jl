module Pipelines

using DataFrames
using Random

using TSML.TSMLTypes
using TSML.BaseFilters
using TSML.BaseFilters: createtransformer
using TSML.Utils

import TSML.TSMLTypes: fit!, transform!
export fit!, transform!
export Pipeline, ComboPipeline, @pipeline, @pipelinex

mutable struct Pipeline <: Transformer
  name::String
  model::Dict
  args::Dict

  function Pipeline(args::Dict = Dict())
    default_args = Dict(
			:name => "Pipeline",
			# transformers as list to chain in sequence.
			:transformers => Vector{Transformer}(),
			# Transformer args as list applied to same index transformer.
			:transformer_args => Dict()
		       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function Pipeline(tf::Vector{T}) where {T<:Transformer}
  Pipeline(Dict(:transformers => tf))
end

function Pipeline(tf...)
  combo=nothing
  if eltype(tf) <: Transformer
    v=[x for x in tf] # convert tuples to vector
    combo = Pipeline(v)
  else
    error("argument setup error")
  end
  return combo
end

function fit!(pipe::Pipeline, features::DataFrame=DataFrame(), labels::Vector=[])
  instances=deepcopy(features)
  transformers = pipe.args[:transformers]
  transformer_args = pipe.args[:transformer_args]

  current_instances = instances
  new_transformers = Transformer[]

  # fit-transform all except last element
  # last element calls fit only
  trlength = length(transformers)
  for t_index in 1:(trlength - 1)
    transformer = createtransformer(transformers[t_index], transformer_args)
    push!(new_transformers, transformer)
    fit!(transformer, current_instances, labels)
    current_instances = transform!(transformer, current_instances)
  end
  transformer = createtransformer(transformers[trlength], transformer_args)
  push!(new_transformers, transformer)
  fit!(transformer, current_instances, labels)

  pipe.model = Dict(
      :transformers => new_transformers,
      :transformer_args => transformer_args
  )
end

function transform!(pipe::Pipeline, instances::DataFrame=DataFrame())
  transformers = pipe.model[:transformers]

  current_instances = deepcopy(instances)
  for t_index in 1:length(transformers)
    transformer = transformers[t_index]
    current_instances = transform!(transformer, current_instances)
  end

  return current_instances
end


mutable struct ComboPipeline <: Transformer
  name::String
  model::Dict
  args::Dict

  function ComboPipeline(args::Dict = Dict())
    default_args = Dict(
			:name => "combopipeline",
			# transformers as list to chain in sequence.
			:transformers => Vector{Transformer}(),
			# Transformer args as list applied to same index transformer.
			:transformer_args => Dict()
		       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function ComboPipeline(tfs::Vector{T}) where {T<:Transformer}
  ComboPipeline(Dict(:transformers => tfs))
end

function ComboPipeline(tfs...)
  combo=nothing
  if eltype(tfs) <: Transformer
    v=[eval(x) for x in tfs] # convert tuples to vector
    combo = ComboPipeline(v)
  else
    error("argument setup error")
  end
  return combo
end


function fit!(pipe::ComboPipeline, features::DataFrame=DataFrame(), labels::Vector=[])
  instances=deepcopy(features)
  transformers = pipe.args[:transformers]
  transformer_args = pipe.args[:transformer_args]

  new_transformers = Transformer[]
  new_instances = DataFrame()
  trlength = length(transformers)
  for t_index in eachindex(transformers)
    transformer = createtransformer(transformers[t_index], transformer_args)
    push!(new_transformers, transformer)
    fit!(transformer, instances, labels)
  end

  pipe.model = Dict(
      :transformers => new_transformers,
      :transformer_args => transformer_args
  )
end

function transform!(pipe::ComboPipeline, features::DataFrame=DataFrame())
  transformers = pipe.model[:transformers]
  instances = deepcopy(features)
  new_instances = DataFrame()
  for t_index in eachindex(transformers)
    transformer = transformers[t_index]
    current_instances = transform!(transformer, instances)
    new_instances = hcat(new_instances,current_instances,makeunique=true)
  end

  return new_instances
end

function processexpr(args)
  for ndx in eachindex(args)
    if typeof(args[ndx]) == Expr
      processexpr(args[ndx].args)
    elseif args[ndx] == :+
      args[ndx] = :Pipeline
    elseif args[ndx] == :*
      args[ndx] = :ComboPipeline
    end
  end
  return args
end

macro pipeline(expr)
  lexpr = :($(esc(expr)))
  res = processexpr(lexpr.args)
  lexpr.args = res
  lexpr
end

macro pipelinex(expr)
  lexpr = :($(esc(expr)))
  res = processexpr(lexpr.args)
  res
end


end
