module TSMLTypes

using DataFrames

export typerun
export 	Transformer,
		TSLearner,
		fit!,
		transform!

abstract type Transformer end
abstract type TSLearner <: Transformer end

function transform!(tr::Transformer, instances::T) where {T<:Union{Vector,Matrix,DataFrame}}
	error(typeof(tr)," not implemented yet: transform!")
end

function fit!(tr::Transformer, instances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
	error(typeof(tr)," not implemented yet: fit!")
end

end
