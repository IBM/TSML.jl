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
