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
