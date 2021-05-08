module Outliernicers

using Random
using Dates
using DataFrames
using Statistics
using StatsBase: iqr, quantile, sample

using ..ValDateFilters
using ..AbsTypes
using ..Utils
import ..AbsTypes: fit, fit!, transform, transform!

export fit, fit!, transform, transform!
export Outliernicer


"""
    Outliernicer(Dict(
       :dateinterval => Dates.Hour(1),
       :nnsize => 1,
       :missdirection => :symmetric,
       :scale => 1.25
    ))

Detects outliers below or above (median-scale*iqr,median+scale*iqr)
and calls DateValNNer to replace them with nearest neighbors.

Example:

    fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
    csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
    valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
    valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
    stfier = Statifier(Dict(:processmissing=>true))
    mono = Monotonicer(Dict())
    outliernicer = Outliernicer(Dict(:dateinterval=>Dates.Hour(1)))

    mpipeline = @pipeline csvfilter |> valgator |> mono |> valnner |> outliernicer |> stfier
    results = fit_transform!(mpipeline)


Implements: `fit!`, `transform!`
"""
mutable struct Outliernicer <: Transformer
   name::String
   model::Dict{Symbol,Any}

   function Outliernicer(args=Dict())
      default_args = Dict{Symbol,Any}(
         :name => "outlrncr",
         :dateinterval => Dates.Hour(1),
         :nnsize => 1,
         :missdirection => :symmetric,
         :scale => 1.25,
         :strict => false
      )
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

"""
    fit!(st::Outliernicer, features::T, labels::Vector=[])

Check that `features` are two-colum data.
"""
function fit!(st::Outliernicer, features::DataFrame, labels::Vector=[])::Nothing
   ncol(features) == 2 || throw(ArgumentError("dataframe must have 2 columns: Date, Val"))
   return nothing
end

function fit(st::Outliernicer, features::DataFrame, labels::Vector=[])::Outliernicer
   fit!(st,features,labels)
   return deepcopy(st)
end

"""
    transform!(st::Outliernicer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Locate outliers based on IQR factor and calls DateValNNer to replace them with nearest neighbors.
"""
function transform!(st::Outliernicer, features::DataFrame)::DataFrame
   features != DataFrame() || return DataFrame()
   ncol(features) == 2 || throw(ArgumentError("dataframe must have 2 columns: Date, Val"))
   sum(names(features) .== ("Date","Value"))  == 2 || throw(ArgumentError("wrong column names"))
   mfeatures=deepcopy(features)
   rvals = mfeatures.Value
   # compute the outlier range
   # setup to store both missing and numbers
   mvals = Array{Union{Missing,eltype(rvals)},1}(missing,length(rvals))
   mvals .= rvals
   crvals = skipmissing(rvals) # stat of non-missing
   miqr = iqr(crvals)
   med = median(crvals) # median
   scale = st.model[:scale]
   lower=med - scale*miqr; upper=med + scale*miqr
   missindx = findall(x -> !ismissing(x) && (x > upper || x < lower),rvals) 
   mvals[missindx] .= missing
   mfeatures.Value = mvals
   # use ValNNer to replace missings
   valnner = DateValNNer(st.model)
   fit!(valnner,mfeatures)
   resdf = transform!(valnner,mfeatures)
   resdf.Value = collect(skipmissing(resdf.Value)) 
   resdf
end

function transform(st::Outliernicer, features::DataFrame)::DataFrame
   return transform!(st,features)
end

end
