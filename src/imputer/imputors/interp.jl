"""
    Interpolate(; context=Context())

Performs linear interpolation between the nearest values in an vector.
The current implementation is univariate, so each variable in a table or matrix will
be handled independently.

WARNING: Missing values at the head or tail of the array cannot be interpolated if there
are no existing values on both sides. As a result, this method does not guarantee
that all missing values will be imputed.

# Keyword Arguments
* `context::AbstractContext`: A context which keeps track of missing data
  summary information

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
struct Interpolate <: Imputor
    context::AbstractContext
end

# TODO: Switch to using Base.@kwdef on 1.1
Interpolate(; context=Context()) = Interpolate(context)

function impute!(data::AbstractVector{<:Union{T, Missing}}, imp::Interpolate) where T
    imp.context() do c
        i = findfirst(c, data) + 1

        while i < lastindex(data)
            if ismissing!(c, data[i])
                prev_idx = i - 1
                next_idx = findnext(c, data, i + 1)

                if next_idx !== nothing
                    gap_sz = (next_idx - prev_idx) - 1

                    diff = data[next_idx] - data[prev_idx]
                    incr = diff / T(gap_sz + 1)
                    val = data[prev_idx] + incr

                    # Iteratively fill in the values
                    for j in i:(next_idx - 1)
                        data[j] = val
                        val += incr
                    end

                    i = next_idx
                else
                    break
                end
            end
            i += 1
        end

        return data
    end
end
