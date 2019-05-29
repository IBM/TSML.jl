module TSMLTransformers

using Dates
using DataFrames
using Statistics
using Random
using CSV

export fit!,transform!

using MLDataUtils: slidingwindow

export Transformer,TSLearner
export Imputer,Pipeline,SKLLearner,OneHotEncoder,Wrapper

export Matrifier,Dateifier
export DateValizer,DateValgator,DateValNNer
export CSVDateValReader, CSVDateValWriter

using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils

include("valdate.jl")

# Transforms instances with nominal features into one-hot form
# and coerces the instance matrix to be of element type Float64.
mutable struct OneHotEncoder <: Transformer
  model
  args

  function OneHotEncoder(args=Dict())
    default_args = Dict(
      # Nominal columns
      :nominal_columns => nothing,
      # Nominal column values map. Key is column index, value is list of
      # possible values for that column.
      :nominal_column_values_map => nothing
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(ohe::OneHotEncoder, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  instances=convert(Matrix,features)
  # Obtain nominal columns
  nominal_columns = ohe.args[:nominal_columns]
  if nominal_columns == nothing
    nominal_columns = find_nominal_columns(instances)
  end

  # Obtain unique values for each nominal column
  nominal_column_values_map = ohe.args[:nominal_column_values_map]
  if nominal_column_values_map == nothing
    nominal_column_values_map = Dict{Int, Any}()
    for column in nominal_columns
      nominal_column_values_map[column] = unique(instances[:, column])
    end
  end

  # Create model
  ohe.model = Dict(
    :nominal_columns => nominal_columns,
    :nominal_column_values_map => nominal_column_values_map
  )
end

function transform!(ohe::OneHotEncoder, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  instances=convert(Matrix,features)
  nominal_columns = ohe.model[:nominal_columns]
  nominal_column_values_map = ohe.model[:nominal_column_values_map]

  # Create new transformed instance matrix of type Float64
  num_rows = size(instances, 1)
  num_columns = (size(instances, 2) - length(nominal_columns))
  if !isempty(nominal_column_values_map)
    num_columns += sum(map(x -> length(x), values(nominal_column_values_map)))
  end
  transformed_instances = zeros(Float64, num_rows, num_columns)

  # Fill transformed instance matrix
  col_start_index = 1
  for column in 1:size(instances, 2)
    if !in(column, nominal_columns)
      transformed_instances[:, col_start_index] = instances[:, column]
      col_start_index += 1
    else
      col_values = nominal_column_values_map[column]
      for row in 1:size(instances, 1)
        entry_value = instances[row, column]
        entry_value_index = findfirst(isequal(entry_value),col_values)
        if entry_value_index == 0
          warn("Unseen value found in OneHotEncoder,
                for entry ($row, $column) = $(entry_value).
                Patching value to $(col_values[1]).")
          entry_value_index = 1
        end
        entry_column = (col_start_index - 1) + entry_value_index
        transformed_instances[row, entry_column] = 1
      end
      col_start_index += length(nominal_column_values_map[column])
    end
  end

  return transformed_instances
end

# Finds all nominal columns.
#
# Nominal columns are those that do not have Real type nor
# do all their elements correspond to Real.
function find_nominal_columns(features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  instances=convert(Matrix,features)
  nominal_columns = Int[]
  for column in 1:size(instances, 2)
    col_eltype = infer_eltype(instances[:, column])
    if !<:(col_eltype, Real)
      push!(nominal_columns, column)
    end
  end
  return nominal_columns
end

# Imputes NaN values from Float64 features.
mutable struct Imputer <: Transformer
  model
  args

  function Imputer(args=Dict())
    default_args = Dict(
      # Imputation strategy.
      # Statistic that takes a vector such as mean or median.
      :strategy => mean
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(imp::Imputer, instances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  imp.model = imp.args
end

function transform!(imp::Imputer, features::T)  where {T<:Union{Vector,Matrix,DataFrame}}
  instances=convert(Matrix,features)
  new_instances = copy(instances)
  strategy = imp.model[:strategy]

  for column in 1:size(instances, 2)
    column_values = instances[:, column]
    col_eltype = infer_eltype(column_values)

    if <:(col_eltype, Real)
      na_rows = map(x -> isnan(x), column_values)
      if any(na_rows)
        fill_value = strategy(column_values[.!na_rows])
        new_instances[na_rows, column] .= fill_value
      end
    end
  end

  return new_instances
end


# Chains multiple transformers in sequence.
mutable struct Pipeline <: Transformer
  model
  args

  function Pipeline(args=Dict())
    default_args = Dict(
      # Transformers as list to chain in sequence.
      :transformers => [OneHotEncoder(), Imputer()],
      # Transformer args as list applied to same index transformer.
      :transformer_args => nothing
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(pipe::Pipeline, features::T=[], labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  #instances=convert(Matrix,features)
  instances=copy(features)
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

function transform!(pipe::Pipeline, features::T=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  #instances = convert(Matrix,features)
  instances = copy(features)
  transformers = pipe.model[:transformers]

  current_instances = instances
  for t_index in 1:length(transformers)
    transformer = transformers[t_index]
    current_instances = transform!(transformer, current_instances)
  end

  return current_instances
end


# Wraps around an CombineML transformer.
mutable struct Wrapper <: Transformer
  model
  args

  function Wrapper(args=Dict())
    default_args = Dict(
      # Transformer to call.
      :transformer => OneHotEncoder(),
      # Transformer args.
      :transformer_args => nothing
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(wrapper::Wrapper, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  instances=convert(Matrix,features)
  transformer_args = wrapper.args[:transformer_args]
  transformer = createtransformer(
    wrapper.args[:transformer],
    transformer_args
  )

  if transformer_args != nothing
    transformer_args = mergedict(transformer.args, transformer_args)
  end
  fit!(transformer, instances, labels)

  wrapper.model = Dict(
    :transformer => transformer,
    :transformer_args => transformer_args
  )
end

function transform!(wrapper::Wrapper, instances::T) where {T<:Union{Vector,Matrix,DataFrame}}
  transformer = wrapper.model[:transformer]
  return transform!(transformer, instances)
end

# Create transformer
#
# @param prototype Prototype transformer to base new transformer on.
# @param options Additional options to override prototype's options.
# @return New transformer.
function createtransformer(prototype::Transformer, args=nothing)
  new_args = copy(prototype.args)
  if args != nothing
    new_args = mergedict(new_args, args)
  end

  prototype_type = typeof(prototype)
  return prototype_type(new_args)
end


end
