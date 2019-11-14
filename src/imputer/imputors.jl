"""
    Imputor

An imputor stores information about imputing values in `AbstractArray`s and `Tables.table`s.
New imputation methods are expected to sutype `Imputor` and, at minimum,
implement the `impute!(imp::<MyImputor>, data::AbstractVector)` method.
"""
abstract type Imputor end

# A couple utility methods to avoid messing up var and obs dimensions
obsdim(dims) = dims
vardim(dims) = dims == 1 ? 2 : 1

function obswise(data::AbstractMatrix; dims=1)
    return (selectdim(data, obsdim(dims), i) for i in axes(data, obsdim(dims)))
end

function varwise(data::AbstractMatrix; dims=2)
    return (selectdim(data, vardim(dims), i) for i in axes(data, vardim(dims)))
end

function filterobs(f::Function, data::AbstractMatrix; dims=1)
    mask = [f(x) for x in obswise(data; dims=dims)]
    return dims == 1 ? data[mask, :] : data[:, mask]
end

function filtervars(f::Function, data::AbstractMatrix; dims=2)
    mask = [f(x) for x in varwise(data; dims=dims)]
    return dims == 1 ? data[:, mask] : data[mask, :]
end

"""
    splitkwargs(::Type{T}, kwargs...) where T <: Imputor -> (imp, rem)

Takes an Imputor type with kwargs and returns the constructed imputor and the
unused kwargs which should be passed to the `impute!` call.

NOTE: This is used by utility methods with construct and imputor and call impute in 1 call.
"""
function splitkwargs(::Type{T}, kwargs...) where T <: Imputor
    rem = Dict(kwargs...)
    kwdef = empty(rem)

    for f in fieldnames(T)
        if haskey(rem, f)
            kwdef[f] = rem[f]
            delete!(rem, f)
        end
    end

    return (T(; kwdef...), rem)
end

# Some utility methods for constructing imputors and imputing data in 1 call.
# NOTE: This is only intended for internal use and is not part of the public API.
function _impute(data, t::Type{T}, kwargs...) where T <: Imputor
    imp, rem = splitkwargs(t, _extract_context_kwargs(kwargs...)...)
    return impute(data, imp; rem...)
end

function _impute!(data, t::Type{T}, kwargs...) where T <: Imputor
    imp, rem = splitkwargs(t, _extract_context_kwargs(kwargs...)...)
    return impute!(data, imp; rem...)
end

"""
    impute(data, imp::Imputor; kwargs...)

Returns a new copy of the `data` with the missing data imputed by the imputor `imp`.

# Keywords
* `dims`: The dimension to impute along (e.g., observations dim)
"""
function impute(data, imp::Imputor; kwargs...)
    # Call `deepcopy` because we can trust that it's available for all types.
    return impute!(deepcopy(data), imp; kwargs...)
end

"""
    impute!(data::AbstractMatrix, imp::Imputor; kwargs...)

Impute the data in a matrix by imputing the values one variable at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `data::AbstractMatrix`: the data to impute
* `imp::Imputor`: the Imputor method to use

# Keywords
* `dims`: The dimension to impute along (e.g., observations dim)

# Returns
* `AbstractMatrix`: the input `data` with values imputed

# Example
```jldoctest
julia> using Impute: Interpolate, Context, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Interpolate(; context=Context(; limit=1.0)); dims=2)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  3.0  4.0  5.0
 1.1  2.2  3.3  4.4  5.5
```
"""
function impute!(data::AbstractMatrix, imp::Imputor; dims=1)
    for var in varwise(data; dims=dims)
        impute!(var, imp)
    end
    return data
end

"""
    impute!(table, imp::Imputor)

Imputes the data in a table by imputing the values 1 column at a time;
if this is not the desired behaviour custom imputor methods should overload this method.

# Arguments
* `imp::Imputor`: the Imputor method to use
* `table`: the data to impute

# Returns
* the input `data` with values imputed

# Example
``jldoctest
julia> using DataFrames; using Impute: Interpolate, Context, impute
julia> df = DataFrame(:a => [1.0, 2.0, missing, missing, 5.0], :b => [1.1, 2.2, 3.3, missing, 5.5])
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ missing  │ 3.3      │
│ 4   │ missing  │ missing  │
│ 5   │ 5.0      │ 5.5      │

julia> impute(df, Interpolate(; context=Context(; limit=1.0)))
5×2 DataFrame
│ Row │ a        │ b        │
│     │ Float64⍰ │ Float64⍰ │
├─────┼──────────┼──────────┤
│ 1   │ 1.0      │ 1.1      │
│ 2   │ 2.0      │ 2.2      │
│ 3   │ 3.0      │ 3.3      │
│ 4   │ 4.0      │ 4.4      │
│ 5   │ 5.0      │ 5.5      │
"""
function impute!(table, imp::Imputor)
    istable(table) || throw(MethodError(impute!, (table, imp)))

    # Extract a columns iterator that we should be able to use to mutate the data.
    # NOTE: Mutation is not guaranteed for all table types, but it avoid copying the data
    columntable = Tables.columns(table)

    for cname in propertynames(columntable)
        impute!(getproperty(columntable, cname), imp)
    end

    return table
end


for file in ("drop.jl", "locf.jl", "nocb.jl", "interp.jl", "fill.jl", "chain.jl", "srs.jl")
    include(joinpath("imputors", file))
end
