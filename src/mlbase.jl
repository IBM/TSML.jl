# MLBase transformers.
module MLBaseWrapper

using DataFrames
using TSML.TSMLTypes
import TSML.TSMLTypes.fit!
import TSML.TSMLTypes.transform!
using TSML.Utils

include("standardize.jl")

using .MStandardize # standardize,estimate,transform

export StandardScaler,
       fit!,
       transform!

# Standardizes each feature using (X - mean) / stddev.
# Will produce NaN if standard deviation is zero.
mutable struct StandardScaler <: Transformer
  model
  args

  function StandardScaler(args=Dict())
    default_args = Dict( 
      :center => true,
      :scale => true
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(st::StandardScaler, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  mfeatures = convert(Matrix,features)
  st_transform = estimate(Standardize, Array(mfeatures'); st.args...)
  st.model = Dict(
    :standardize_transform => st_transform
  )
end

function transform!(st::StandardScaler, features::T)  where {T<:Union{Vector,Matrix,DataFrame}}
  mfeatures = convert(Matrix,features)
  st_transform = st.model[:standardize_transform]
  transposed_instances = Array(mfeatures')
  return Array(transform(st_transform, transposed_instances)')
end

end # module
