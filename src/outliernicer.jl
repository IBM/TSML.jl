module Outliernicers

using Dates
using DataFrames
using Statistics
using StatsBase: iqr, quantile, sample
using TSML.ValDateFilters

using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils
import AutoMLPipeline.AbsTypes: fit!, transform!

export fit!,transform!
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
  model
  args

  function Outliernicer(args=Dict())
    default_args = Dict(
        :dateinterval => Dates.Hour(1),
        :nnsize => 1,
        :missdirection => :symmetric,
        :scale => 1.25

    )
    new(nothing, mergedict(default_args, args))
  end
end

"""
    fit!(st::Outliernicer, features::T, labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}

Check that `features` are two-colum data.
"""
function fit!(st::Outliernicer, features::DataFrame, labels::Vector=[]) 
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  st.model = st.args
end

"""
    transform!(st::Outliernicer, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Locate outliers based on IQR factor and calls DateValNNer to replace them with nearest neighbors.
"""
function transform!(st::Outliernicer, features::DataFrame)
  features != DataFrame() || return DataFrame()
  ncol(features) == 2 || error("dataframe must have 2 columns: Date, Val")
  sum(names(features) .== ("Date","Value"))  == 2 || error("wrong column names")
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
  valnner = DateValNNer(st.args)
  fit!(valnner,mfeatures)
  resdf = transform!(valnner,mfeatures)
  resdf.Value = collect(skipmissing(resdf.Value)) 
  resdf
end

end
