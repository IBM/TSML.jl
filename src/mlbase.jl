# MLBase transformers.
module MLBaseWrapper

using Random
using DataFrames
using LinearAlgebra

using ..AbsTypes
using ..Utils
import ..AbsTypes: fit!, transform!

export fit!,transform!

export Standardize
export standardize, standardize!, estimate, transform,StandardScaler

"""
    StandardScaler(
       Dict( 
          :impl_args => Dict(
              :center => true,
              :scale => true
          )
       )
    )

Standardizes each feature using (X - mean) / stddev.
Will produce NaN if standard deviation is zero.
"""
mutable struct StandardScaler <: Transformer
   name::String
   model::Dict{Symbol,Any}

  function StandardScaler(args=Dict())
     default_args = Dict{Symbol,Any}( 
      :name => "stdsclr",
      :impl_args => Dict{Symbol,Any}(
         :center => true,
         :scale => true
      )
    )
    cargs=nested_dict_merge(default_args,args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],cargs)
 end
end

"""
    fit!(st::StandardScaler, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}

Compute the parameters to center and scale.
"""
function fit!(st::StandardScaler, features::DataFrame, labels::Vector=[]) 
   mfeatures = convert(Matrix{Float64},features)
   pfeatures = mfeatures' |> collect |> Matrix{Float64}
   impl_args = st.model[:impl_args]
   st_transform = estimate(Standardize, Array(mfeatures'); impl_args...)
   st.model[:standardize_transform] = st_transform
end

"""
    transform!(st::StandardScaler, features::T)  where {T<:Union{Vector,Matrix,DataFrame}}

Apply the computed parameters for centering and scaling to new data.
"""
function transform!(st::StandardScaler, features::DataFrame)
   mfeatures = convert(Matrix{Float64},features)
   st_transform = st.model[:standardize_transform]
   pfeatures = mfeatures' |> collect |> Matrix{Float64}
   transposed_instances = Array(pfeatures)
   pres = transform(st_transform, transposed_instances)
   return (pres' |> collect |> Array{Float64}) |> DataFrame
end

### Standardization

"""
    Standardize(d::Int, m::Vector{Float64}, s::Vector{Float64})

Standardization type.
"""
struct Standardize
    dim::Int
    mean::Vector{Float64}
    scale::Vector{Float64}

    function Standardize(d::Int, m::Vector{Float64}, s::Vector{Float64})
        lenm = length(m)
        lens = length(s)
        lenm == d || lenm == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lens == d || lens == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new(d, m, s)
    end
end

indim(t::Standardize) = t.dim
outdim(t::Standardize) = t.dim

function transform!(y::DenseArray{YT,1}, t::Standardize, x::DenseArray{XT,1}) where {YT<:Real,XT<:Real}
    d = t.dim
    length(x) == length(y) == d || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if !is(x, y)
                copy!(y, x)
            end
        else
            for i = 1:d
                @inbounds y[i] = x[i] * s[i]
            end
        end
    else
        if isempty(s)
            for i = 1:d
                @inbounds y[i] = x[i] - m[i]
            end
        else
            for i = 1:d
                @inbounds y[i] = s[i] * (x[i] - m[i])
            end
        end
    end
    return y
end

function transform!(y::DenseArray{YT,2}, t::Standardize, x::DenseArray{XT,2}) where {YT<:Real,XT<:Real}
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(x,2)
    size(y,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if !is(x, y)
                copy!(y, x)
            end
        else
            for j = 1:n
                xj = view(x, :, j)
                yj = view(y, :, j)
                for i = 1:d
                    @inbounds yj[i] = xj[i] * s[i]
                end
            end
        end
    else
        if isempty(s)
            for j = 1:n
                xj = view(x, :, j)
                yj = view(y, :, j)
                for i = 1:d
                    @inbounds yj[i] = xj[i] - m[i]
                end
            end
        else
            for j = 1:n
                xj = view(x, :, j)
                yj = view(y, :, j)
                for i = 1:d
                    @inbounds yj[i] = s[i] * (xj[i] - m[i])
                end
            end
        end
    end
    return y
end

transform!(t::Standardize, x::DenseArray{T,1}) where {T<:AbstractFloat} = transform!(x, t, x)
transform!(t::Standardize, x::DenseArray{T,2}) where {T<:AbstractFloat} = transform!(x, t, x)

transform(t::Standardize, x::DenseArray{T,1}) where {T<:Real} = transform!(Array{Float64}(undef,size(x)), t, x)
transform(t::Standardize, x::DenseArray{T,2}) where {T<:Real} = transform!(Array{Float64}(undef,size(x)), t, x)

# estimate a standardize transform

function estimate(::Type{Standardize}, X::DenseArray{T,2}; center::Bool=true, scale::Bool=true) where T<:Real
    d, n = size(X)
    n >= 2 || error("X must contain at least two columns.")

    m = Array{Float64}(undef,ifelse(center, d, 0))
    s = Array{Float64}(undef,ifelse(scale, d, 0))

    if center
        fill!(m, 0.0)
        for j = 1:n
            xj = view(X, :, j)
            for i = 1:d
                @inbounds m[i] += xj[i]
            end
        end
        rmul!(m, 1.0 / n)
    end

    if scale
        fill!(s, 0.0)
        if center
            for j = 1:n
                xj = view(X, :, j)
                for i = 1:d
                    @inbounds s[i] += abs2(xj[i] - m[i])
                end
            end
        else
            for j = 1:n
                xj = view(X, :, j)
                for i = 1:d
                    @inbounds s[i] += abs2(xj[i])
                end
            end
        end
        for i = 1:d
            @inbounds s[i] = sqrt((n - 1) / s[i])
        end
    end

    return Standardize(d, m, s)
end

# standardize

function standardize(X::DenseArray{T,2}; center::Bool=true, scale::Bool=true) where T<:Real
    t = estimate(Standardize, X; center=center, scale=scale)
    Y = transform(t, X)
    return (Y, t)
end

function standardize!(X::DenseArray{T,2}; center::Bool=true, scale::Bool=true) where T<:AbstractFloat
    t = estimate(Standardize, X; center=center, scale=scale)
    Y = transform!(t, X)
    return (Y, t)
end

end
