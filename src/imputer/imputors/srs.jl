struct SRS <: Imputor
    rng::AbstractRNG
    context::AbstractContext
end


"""
    SRS(; rng=Random.GLOBAL_RNG, context=Context())

Simple Random Sampling (SRS) imputation is a method for imputing both continuous and categorical
variables. Furthermore, it completes imputation while preserving the distributional
properties of the variables (e.g., mean, standard deviation).

The basic idea is that for a given variable, `x`, with missing data, we randomly draw
from the observed values of `x` to impute the missing elements. Since the random draws
from `x` for imputation are done in proportion to the frequency distribution of the values
in `x`, the univariate distributional properties are generally not impacted; this is true
for both categorical and continuous data.


# Keyword Arguments
* `rng::AbstractRNG`: A random number generator to use for observation selection
* `context::AbstractContext`: A context which keeps track of missing data
  summary information

# Example
"""

SRS(; rng=Random.GLOBAL_RNG, context=Context()) = SRS(rng, context)

function impute!(data::AbstractVector, imp::SRS)
    imp.context() do c
        obs_values = Impute.dropobs(data)
        if !isempty(obs_values)
            for i in eachindex(data)
                if ismissing!(c, data[i])
                    data[i] = rand(imp.rng, obs_values)
                end
            end
        end

        return data
    end
end
