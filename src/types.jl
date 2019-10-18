@reexport module TSMLTypes

using DataFrames

export 	Transformer,
		TSLearner,
		fit!,
		transform!

abstract type Transformer end
abstract type TSLearner <: Transformer end


"""
    fit!(tr::Transformer, instances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}

Generic `fit!` function to be redefined using multidispatch in  different subtypes of `Transformer`.
"""
function fit!(tr::Transformer, instances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
	error(typeof(tr)," not implemented yet: fit!")
end

"""
    transform!(tr::Transformer, instances::T) where {T<:Union{Vector,Matrix,DataFrame}}

Generic `transform!` function to be redefined using multidispatch in  different subtypes of `Transformer`.
"""
function transform!(tr::Transformer, instances::T) where {T<:Union{Vector,Matrix,DataFrame}}
	error(typeof(tr)," not implemented yet: transform!")
end


end
