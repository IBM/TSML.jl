module Utils
export mergedict

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
