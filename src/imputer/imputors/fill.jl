"""
    Fill(; value=mean, context=Context())

Fills in the missing data with a specific value.
The current implementation is univariate, so each variable in a table or matrix will
be handled independently.

# Keyword Arguments
* `value::Any`: A scalar or a function that returns a scalar if
  passed the data with missing data removed (e.g, `mean`)
* `context::AbstractContext`: A context which keeps track of missing data
  summary information

# Example
```jldoctest
julia> using Impute: Fill, Context, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, Fill(; context=Context(; limit=1.0)); dims=2)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.66667  2.66667  5.0
 1.1  2.2  3.3      3.025    5.5
```
"""
struct Fill{T} <: Imputor
    value::T
    context::AbstractContext
end

# TODO: Switch to using Base.@kwdef on 1.1
Fill(; value=mean, context=Context()) = Fill(value, context)

function impute!(data::AbstractVector, imp::Fill)
    imp.context() do c
        fill_val = if isa(imp.value, Function)
            # Call `deepcopy` because we can trust that it's available for all types.
            imp.value(Impute.drop(data; context=c))
        else
            imp.value
        end

        for i in eachindex(data)
            if ismissing!(c, data[i])
                data[i] = fill_val
            end
        end

        return data
    end
end
