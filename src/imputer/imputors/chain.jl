"""
    Chain <: Imputor

Runs multiple `Imputor`s on the same data in the order they're provided.

# Fields
* `imputors::Array{Imputor}`
"""
struct Chain <: Imputor
    imputors::Vector{Imputor}
end

"""
    Chain(imputors::Imputor...) -> Chain

Creates a Chain using the `Imputor`s provided (ordering matters).
"""
Chain(imputors::Imputor...) = Chain(collect(imputors))

"""
Compose new `Imputor` chains with the composition operator

# Example

```jldoctest
julia> using Impute: impute, Interpolate, NOCB, LOCF, Context

julia> ctx = Context(; limit=1.0)
Context(0, 0, 1.0, ismissing, Impute.complete)

julia> imp = Interpolate(; context=ctx) ∘ NOCB(; context=ctx) ∘ LOCF(; context=ctx)
Impute.Chain(Impute.Imputor[Interpolate(2, Context(0, 0, 1.0, ismissing, complete)), NOCB(2, Context(0, 0, 1.0, ismissing, complete)), LOCF(2, Context(0, 0, 1.0, ismissing, complete))])
```
"""
Base.:(∘)(a::Imputor, b::Imputor) = Chain([a, b])
function Base.:(∘)(a::Chain, b::Imputor)
    push!(a.imputors, b)
    return a
end

"""
    impute!(data, imp::Chain)

Runs the `Imputor`s on the supplied data.

# Arguments
* `imp::Chain`: the chain to run
* `data`: our data to impute

# Returns
* our imputed data
"""
function impute!(data, imp::Chain)
    for imputor in imp.imputors
        data = impute!(data, imputor)
    end

    return data
end
