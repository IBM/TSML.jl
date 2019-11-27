"""
    LOCF(; context=Context())

Last observation carried forward (LOCF) iterates forwards through the `data` and fills
missing data with the last existing observation. The current implementation is univariate,
so each variable in a table or matrix will be handled independently.

See also:
- [NOCB](@ref): Next Observation Carried Backward

WARNING: missing elements at the head of the array may not be imputed if there is no
existing observation to carry forward. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments
* `context::AbstractContext`: A context which keeps track of missing data
  summary information

# Example
```jldoctest
julia> using Impute: LOCF, Context, impute

julia> M = [1.0 2.0 missing missing 5.0; 1.1 2.2 3.3 missing 5.5]
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0   missing  missing  5.0
 1.1  2.2  3.3       missing  5.5

julia> impute(M, LOCF(; context=Context(; limit=1.0)); dims=2)
2×5 Array{Union{Missing, Float64},2}:
 1.0  2.0  2.0  2.0  5.0
 1.1  2.2  3.3  3.3  5.5
```
"""
struct LOCF <: Imputor
    context::AbstractContext
end

# TODO: Switch to using Base.@kwdef on 1.1
LOCF(; context=Context()) = LOCF(context)

function impute!(data::AbstractVector, imp::LOCF)
    imp.context() do c
        start_idx = findfirst(c, data) + 1
        for i in start_idx:lastindex(data)
            if ismissing!(c, data[i])
                data[i] = data[i-1]
            end
        end

        return data
    end
end
