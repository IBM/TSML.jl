@reexport module Utils

using Random: randperm
export mergedict, getiris 
export skipmean,skipmedian,skipstd
export aggregatorclskipmissing

export holdout,
       kfold,
       score,
       infer_eltype,
       nested_dict_to_tuples,
       nested_dict_set!,
       nested_dict_merge,
       create_transformer


using TSML: Transformer
using Statistics
using DataFrames
using CSV

import MLBase: Kfold

# Holdout method that partitions a collection
# into two partitions.
#
# @param n Size of collection to partition.
# @param right_prop Percentage of collection placed in right partition.
# @return Two partitions of indices, left and right.
function holdout(n, right_prop)
  shuffled_indices = randperm(n)
  partition_pivot = round(Int,right_prop * n)
  right = shuffled_indices[1:partition_pivot]
  left = shuffled_indices[partition_pivot+1:end]
  return (left, right)
end

# Returns k-fold partitions.
#
# @param num_instances Total number of instances.
# @param num_partitions Number of partitions required.
# @return Returns training set partition.
function kfold(num_instances, num_partitions)
  return collect(Kfold(num_instances, num_partitions))
end

# Score learner predictions against ground truth values.
#
# Available metrics:
# - :accuracy
#
# @param metric Metric to assess with.
# @param actual Ground truth values.
# @param predicted Predicted values.
# @return Score of learner.
function score(metric::Symbol, actual, predicted)
  if metric == :accuracy
    mean(actual .== predicted) * 100.0
  else
    error("Metric $metric not implemented for score.")
  end
end

# Returns element type of vector unless it is Any.
# If Any, returns the most specific type that can be
# inferred from the vector elements.
#
# @param vector Vector to infer element type on.
# @return Inferred element type.
function infer_eltype(vector::Vector)
  # Obtain element type of vector
  vec_eltype = eltype(vector)

  # If element type of Vector is Any and not empty,
  # and all elements are of the same type,
  # then return the vector elements' type.
  if vec_eltype == Any && !isempty(vector)
    all_elements_same_type = all(x -> typeof(x) == typeof(first(vector)), vector)
    if all_elements_same_type
      vec_eltype = typeof(first(vector))
    end
  end

  # Return inferred element type
  return vec_eltype
end

# Converts nested dictionary to list of tuples
#
# @param dict Dictionary that can have other dictionaries as values.
# @return List where elements are ([outer-key, inner-key, ...], value).
function nested_dict_to_tuples(dict::Dict)
  set = Set()
  for (entry_id, entry_val) in dict
    if typeof(entry_val) <: Dict
      inner_set = nested_dict_to_tuples(entry_val)
      for (inner_entry_id, inner_entry_val) in inner_set
        new_entry = (vcat([entry_id], inner_entry_id), inner_entry_val)
        push!(set, new_entry)
      end
    else
      new_entry = ([entry_id], entry_val)
      push!(set, new_entry)
    end
  end
  return set
end

# Set value in a nested dictionary.
#
# @param dict Nested dictionary to assign value.
# @param keys Keys to access nested dictionaries in sequence.
# @param value Value to assign.
function nested_dict_set!(dict::Dict, keys::Array{T, 1}, value) where {T}
  inner_dict = dict
  for key in keys[1:end-1]
    inner_dict = inner_dict[key]
  end
  inner_dict[keys[end]] = value
end

# Second nested dictionary is merged into first.
#
# If a second dictionary's value as well as the first
# are both dictionaries, then a merge is conducted between
# the two inner dictionaries.
# Otherwise the second's value overrides the first.
#
# @param first First nested dictionary.
# @param second Second nested dictionary.
# @return Merged nested dictionary.
function nested_dict_merge(first::Dict, second::Dict)
  target = copy(first)
  for (second_key, second_value) in second
    values_both_dict =
      typeof(second_value) <: Dict &&
      typeof(get(target, second_key, nothing)) <: Dict
    if values_both_dict
      target[second_key] = nested_dict_merge(target[second_key], second_value)
    else
      target[second_key] = second_value
    end
  end
  return target
end

# Create transformer
#
# @param prototype Prototype transformer to base new transformer on.
# @param options Additional options to override prototype's options.
# @return New transformer.
function create_transformer(prototype::Transformer, options=nothing)
  new_options = copy(prototype.options)
  if options != nothing
    new_options = nested_dict_merge(new_options, options)
  end

  prototype_type = typeof(prototype)
  return prototype_type(new_options)
end

# closure to create aggregator closure with skipmissing features
function aggregatorclskipmissing(fn::Function)
  function skipagg(x::Union{AbstractArray,DataFrame})
    if length(collect(skipmissing(x))) == 0
      return missing
    else
      return fn(skipmissing(x))
    end
  end
  return skipagg
end


function skipmean(x::T) where {T<:Union{AbstractArray,DataFrame}} 
  if length(collect(skipmissing(x))) == 0
    missing
  else
    mean(skipmissing(x))
  end
end

function skipmedian(x::T) where {T<:Union{AbstractArray,DataFrame}} 
  if length(collect(skipmissing(x))) == 0
    missing
  else
    median(skipmissing(x))
  end
end

function skipstd(x::T) where {T<:Union{AbstractArray,DataFrame}} 
  if length(collect(skipmissing(x))) == 0
    missing
  else
    std(skipmissing(x))
  end
end

function mergedict(first::Dict, second::Dict)
  target = copy(first)
  for (second_key, second_value) in second
    values_both_dict =
      typeof(second_value) <: Dict &&
      typeof(get(target, second_key, nothing)) <: Dict
    if values_both_dict
      target[second_key] = mergedict(target[second_key], second_value)
    else
      target[second_key] = second_value
    end
  end
  return target
end

function getiris()
  iris = CSV.read(joinpath(Base.@__DIR__,"../data","iris.csv"))
  return iris
end


end
