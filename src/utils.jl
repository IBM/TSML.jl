module Utils

export mergedict
export skipmean,skipmedian,skipstd

using Statistics
using DataFrames

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


end
