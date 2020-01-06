module Imputers

using IterTools
using Random
using Statistics
using StatsBase
using Tables: Tables, materializer, istable

import Base.Iterators: drop
import DataFrames: dropmissing

export impute, impute!, chain, chain!, drop, drop!, interp, interp!, ImputeError, locf, nocb 

"""
    ImputeError{T} <: Exception

Is thrown by `impute` methods when the limit of imputable values has been exceeded.

# Fields
* msg::T - the message to print.
"""
struct ImputeError{T} <: Exception
    msg::T
end

Base.showerror(io::IO, err::ImputeError) = println(io, "ImputeError: $(err.msg)")

include("context.jl")
include("imputors.jl")

#=
These default methods are required because @auto_hash_equals doesn't
play nice with Base.@kwdef
=#
function Base.hash(imp::T, h::UInt) where T <: Union{Imputor, AbstractContext}
    h = hash(Symbol(T), h)

    for f in fieldnames(T)
        h = hash(getfield(imp, f), h)
    end

    return h
end

function Base.:(==)(a::T, b::T) where T <: Union{Imputor, AbstractContext}
    result = true

    for f in fieldnames(T)
        if !isequal(getfield(a, f), getfield(b, f))
            result = false
            break
        end
    end

    return result
end

const global imputation_methods = (
    drop = DropObs,
    dropobs = DropObs,
    dropvars = DropVars,
    interp = Interpolate,
    interpolate = Interpolate,
    fill = Fill,
    locf = LOCF,
    nocb = NOCB,
    srs = SRS,
)

include("deprecated.jl")

for (f, v) in pairs(imputation_methods)
    typename = nameof(v)
    f! = Symbol(f, :!)

    @eval begin
        $f(data; kwargs...) = _impute(data, $typename, kwargs...)
        $f!(data; kwargs...) = _impute!(data, $typename, kwargs...)
        $f(; kwargs...) = data -> _impute(data, $typename, kwargs...)
        $f!(; kwargs...) = data -> _impute!(data, $typename, kwargs...)
    end
end

@doc """
    Impute.dropobs(data; dims=1, context=Context())

Removes missing observations from the `AbstractArray` or `Tables.table` provided.
See [DropObs](@ref) for details.

# Example
```
julia> using DataFrames; using Impute: Impute, Context

julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrames.DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> Impute.dropobs(df; dims=2, context=Context(; limit=1.0))
3×2 DataFrames.DataFrame
│ Row │ a       │ b       │
│     │ Float64 │ Float64 │
├─────┼─────────┼─────────┤
│ 1   │ 1.0     │ 1.1     │
│ 2   │ 2.0     │ 2.2     │
│ 3   │ 5.0     │ 5.5     │
```
""" dropobs

@doc """
    Impute.dropvars(data; dims=1, context=Context())

Finds variables with too many missing values in a `AbstractMatrix` or `Tables.table` and
removes them from the input data. See [DropVars](@ref) for details.

# Example
""" dropvars

@doc """
    Impute.interp(data; dims=1, context=Context())

Performs linear interpolation between the nearest values in an vector.
See [Interpolate](@ref) for details.

# Example
""" interp

@doc """
    Impute.fill(data; value=mean, dims=1, context=Context())

Fills in the missing data with a specific value. See [Fill](@ref) for details.

# Example
""" fill

@doc """
    Impute.locf(data; dims=1, context=Context())

Iterates forwards through the `data` and fills missing data with the last existing
observation. See [LOCF](@ref) for details.

# Example
""" locf

@doc """
    Impute.nocb(data; dims=1, context=Context())

Iterates backwards through the `data` and fills missing data with the next existing
observation. See [LOCF](@ref) for details.

# Example
""" nocb

@doc """
    Impute.srs(data; rng=Random.GLOBAL_RNG, context=Context())

Simple Random Sampling (SRS) imputation is a method for imputing both continuous and
categorical variables. Furthermore, it completes imputation while preserving the
distributional properties of the variables (e.g., mean, standard deviation).

# Example
""" srs

end  # module
