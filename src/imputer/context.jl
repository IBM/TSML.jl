"""
    AbstractContext

An imputation context records summary information about missing data for an imputation algorithm.
All `AbstractContext`s are callable with a function, which allows us to write code like:

```julia
context() do c
    # My imputation code using a clean context
end
```

This do-block will pass a fresh context to your code and apply the `on_complete` function on
the resulting data and context state. By default, `on_complete` will throw an
[ImputeError](@ref) if we have too many missing values.
"""
abstract type AbstractContext end

# We implement a version of copy for all contexts which reconstructs the context from the
# raw fields.
Base.copy(ctx::T) where {T <: AbstractContext} = T(fieldvalues(ctx)...)

"""
    ismissing!(ctx::AbstractContext, x) -> Bool

Uses `ctx.is_missing` to determine if x is missing. If x is a `NamedTuple` or an `AbstractArray`
then `ismissing!` will return true if `ctx.is_missing` returns true for any element.
The ctx.count is increased whenever whenever we return true and if `ctx.count / ctx.num`
exceeds our `ctx.limit` we throw an `ImputeError`

# Arguments
* `ctx::Context`: the contextual information about missing information.
* `x`: the value to check (may be an single values, abstract array or row)
"""
function ismissing!(ctx::AbstractContext, x)
    was_missing = if isa(x, NamedTuple)
        any(ctx.is_missing, Tuple(x))
    elseif isa(x, AbstractArray)
        any(ctx.is_missing, x)
    else
        ctx.is_missing(x)
    end

    missing_update!(ctx, was_missing)

    return was_missing
end

"""
    findfirst(ctx::AbstractContext, data::AbstractVector) -> Int

Returns the first non-missing index in `data`.

# Arguments
* `ctx::AbstractContext`: the context to pass into `ismissing!`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the first index in `data` that isn't missing
"""
function Base.findfirst(ctx::AbstractContext, data::AbstractVector)
    return findfirst(x -> !ismissing!(ctx, x), data)
end

"""
    findlast(ctx::AbstractContext, data::AbstractVector) -> Int

Returns the last non-missing index in `data`.

# Arguments
* `ctx::AbstractContext`: the context to pass into `ismissing!`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the last index in `data` that isn't missing
"""
function Base.findlast(ctx::AbstractContext, data::AbstractVector)
    return findlast(x -> !ismissing!(ctx, x), data)
end

"""
    findnext(ctx::AbstractContext, data::AbstractVector) -> Int

Returns the next non-missing index in `data`.

# Arguments
* `ctx::AbstractContext`: the context to pass into `ismissing!`
* `data::AbstractVector`: the data array to search

# Returns
* `Int`: the next index in `data` that isn't missing
"""
function Base.findnext(ctx::AbstractContext, data::AbstractVector, idx::Int)
    return findnext(x -> !ismissing!(ctx, x), data, idx)
end

mutable struct Context <: AbstractContext
    num::Int
    count::Int
    limit::Float64
    is_missing::Function
    on_complete::Function
end

"""
    Context

Records base information about the missing data and assume all observations are equally
weighted.

# Keyword Arguments
* `n::Int`: number of observations
* `count::Int`: number of missing values found
* `limit::Float64`: portion of total values allowed to be imputed (should be between 0.0 and 1.0).
* `is_missing::Function`: must return a Bool indicating if the value counts as missing
* `on_complete::Function`: a function to run when imputation is complete
"""
function Context(;
    limit::Float64=0.1,
    is_missing::Function=ismissing,
    on_complete::Function=complete
)
    return Context(0, 0, limit, is_missing, on_complete)
end

function Base.empty(ctx::Context)
    _ctx = copy(ctx)
    _ctx.num = 0
    _ctx.count = 0

    return _ctx
end

function missing_update!(ctx::Context, was_missing)
    ctx.num += 1

    if was_missing
        ctx.count += 1
    end
end

function complete(ctx::Context, data)
    missing_ratio = ctx.count / ctx.num

    if missing_ratio > ctx.limit
        throw(ImputeError(
            "More than $(ctx.limit * 100)% of values were missing ($missing_ratio)."
        ))
    end

    return data
end


mutable struct WeightedContext <: AbstractContext
    num::Int
    s::Float64
    limit::Float64
    is_missing::Function
    on_complete::Function
    wv::AbstractWeights
end

"""
    WeightedContext(wv; limit=1.0, is_missing=ismissing, on_complete=complete)

Records information about the missing data relative to a set of weights.
This context type can be useful if some missing observation are more important than others
(e.g., more recent observations in time series datasets)

# Arguments
* `wv::AbstractWeights`: a set of statistical weights to use when evaluating the importance
  of each observation. Will be accumulated during imputation.

# Keyword Arguments
* `num::Int`: number of observations
* `s::Float64`: sum of the weights of missing values
* `limit::Float64`: portion of total values allowed to be imputed (should be between 0.0 and 1.0).
* `is_missing::Function`: returns a Bool if the value counts as missing
* `on_complete::Function`: a function to run when imputation is complete
"""
function WeightedContext(
    wv::AbstractWeights;
    limit::Real=1.0,
    is_missing::Function=ismissing,
    on_complete::Function=complete
)
    return WeightedContext(0, 0.0, limit, is_missing, on_complete, wv)
end

function Base.empty(ctx::WeightedContext)
    _ctx = copy(ctx)
    _ctx.num = 0
    _ctx.s = 0.0

    return _ctx
end

function missing_update!(ctx::WeightedContext, was_missing)
    ctx.num += 1

    if was_missing
        ctx.s += ctx.wv[ctx.num]
    end
end

function complete(ctx::WeightedContext, data)
    missing_ratio = ctx.s / sum(ctx.wv)

    if missing_ratio > ctx.limit
        throw(ImputeError(
            "More than $(ctx.limit * 100)% of weighted values were missing ($missing_ratio)."
        ))
    end

    return data
end

#=
Define our callable methods for each context. Once we drop 1.0 we should be able to just
define this on the `AbstractContext`.
=#
for T in (Context, WeightedContext)
    @eval begin
        function (ctx::$T)(f::Function)
            _ctx = empty(ctx)
            return ctx.on_complete(_ctx, f(_ctx))
        end
    end
end
