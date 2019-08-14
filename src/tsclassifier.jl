# Decision trees as found in DecisionTree Julia package.
@reexport module TSClassifiers
using Reexport

"""
Given a bunch of time-series with specific types. Get the statistical features of each,
use these as inputs to RF classifier with output as the TS type, train and test. Another
option is to use these stat features for clustering and check cluster quality. If
accuracy is poor, add more stat features and repeat same process as outlined for training
and testing. Assume that each time-series is named based on their type which will be
used as target output. For example, temperature time series will be named as temperature?.csv
where ? is an integer.

Loop over each file in a directory, get stat and record in a dictionary/dataframe, train/test.

"""

include("filestats.jl") # common functions for ensembles and other mls
using .FileStats

export TSClassifier, fit!, transform!

using TSML.TSMLTypes
import TSML.TSMLTypes.fit!
import TSML.TSMLTypes.transform!
using TSML.Utils
using TSML.TSMLTransformers

using TSML.DecisionTreeLearners: RandomForest
using TSML.Statifiers
using TSML: CSVDateValReader

using CSV
using DataFrames
using Dates
using Serialization


# Default to using RandomForest for classification of data types
mutable struct TSClassifier <: TSLearner
  model
  args
  function TSClassifier(args=Dict())
    default_args = Dict(
      # training directory
      :trdirectory => "",
      :tstdirectory => "",
      :modeldirectory => "",
      :feature_range => 7:20,
      :juliarfmodelname => "juliarfmodel.serialized",
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_args => Dict(
        # Merge leaves having >= purity_threshold CombineMLd purity.
        :purity_threshold => 1.0,
        # Maximum depth of the decision tree (default: no maximum).
        :max_depth => -1,
        # Minimum number of samples each leaf needs to have.
        :min_samples_leaf => 1,
        # Minimum number of samples in needed for a split.
        :min_samples_split => 2,
        # Minimum purity needed for a split.
        :min_purity_increase => 0.0
      )
    )
    mergedargs=mergedict(default_args, args)
    new(nothing, mergedargs)
  end
end

# get the stats of each file, collect as dataframe, train
function fit!(tsc::TSClassifier, features::T=[], labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  ispathnotempty(tsc.args) || error("empty training/testing/modeling directory")
  ldirname = tsc.args[:trdirectory]
  mdirname = tsc.args[:modeldirectory]
  modelfname=tsc.args[:juliarfmodelname]
  trdata = getstats(ldirname)
  rfmodel = RandomForest(tsc.args)
  xfeatures = tsc.args[:feature_range]
  X=trdata[:,xfeatures]
  Y=trdata[:,:dtype]
  fit!(rfmodel,X,Y)
  mkpath(mdirname)
  serializedmodel = joinpath(mdirname,modelfname)
  open(serializedmodel,"w") do file
    serialize(file,rfmodel)
  end
  trstatfname = joinpath(mdirname,modelfname*".csv")
  trdata |> CSV.write(trstatfname)
  tsc.args[:features] = names(X)
  tsc.model = rfmodel
end

function transform!(tsc::TSClassifier, features::T=[]) where {T<:Union{Vector,Matrix,DataFrame}}
  ldirname = tsc.args[:tstdirectory]
  mdirname = tsc.args[:modeldirectory]
  modelfname=tsc.args[:juliarfmodelname]
  trdata = getstats(ldirname)
  xfeatures = tsc.args[:feature_range]
  X=trdata[:,xfeatures]
  mfeatures=tsc.args[:features]
  (sum(names(X) .== mfeatures ) == length(mfeatures)) || error("features mismatch")
  serializedmodel = joinpath(mdirname,modelfname)
  if isfile(serializedmodel)
    println("loading model from file: "*serializedmodel)
    model=open(serializedmodel,"r") do file
      deserialize(file)
    end
  else
    model= tsc.model
  end
  mpred = transform!(model,X)
  return DataFrame(fname=trdata.fname,predtype=mpred)
end

end # module
