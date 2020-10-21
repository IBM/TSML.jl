module TSClassifiers

using Random
using CSV
using DataFrames
using Dates
using Serialization

using ..Pipelines
using ..DecisionTreeLearners: RandomForest
using ..Statifiers
using ..ValDateFilters
using ..AbsTypes
using ..Utils
import AutoMLPipeline.AbsTypes: fit!, transform!

export fit!, transform!
export TSClassifier, getstats


"""
    TSClassifier(
       Dict(
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
    )

Given a bunch of time-series with specific types. Get the statistical features of each,
use these as inputs to RF classifier with output as the TS type, train and test. Another
option is to use these stat features for clustering and check cluster quality. If
accuracy is poor, add more stat features and repeat same process as outlined for training
and testing. Assume that each time-series is named based on their type which will be
used as target output. For example, temperature time series will be named as temperature?.csv
where ? is an integer. Loop over each file in a directory, get stat and 
record in a dictionary/dataframe, train/test. Default to using RandomForest 
for classification of data types.
"""
mutable struct TSClassifier <: Learner
   name::String
   model::Dict{Symbol,Any}

   function TSClassifier(args=Dict())
      default_args = Dict{Symbol,Any}(
          :name => "tscl",
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
          :impl_args => Dict{Symbol,Any}(
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
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

"""
    fit!(tsc::TSClassifier, features::T=[], labels::Vector=[]) where {T<:Union{Vector,Matrix,DataFrame}}
    
Get the stats of each file, collect as dataframe, and train.
"""
function fit!(tsc::TSClassifier, features::DataFrame=DataFrame(), labels::Vector=[])
   ispathnotempty(tsc.model) || throw(ArgumentError("empty training/testing/modeling directory"))
   ldirname = tsc.model[:trdirectory]
   mdirname = tsc.model[:modeldirectory]
   modelfname=tsc.model[:juliarfmodelname]
   trdata = getstats(ldirname)
   rfmodel = RandomForest(tsc.model)
   xfeatures = tsc.model[:feature_range]
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
   tsc.model[:features] = names(X)
   tsc.model[:fmodel] = rfmodel
end


"""
    transform!(tsc::TSClassifier, features::T=[]) where {T<:Union{Vector,Matrix,DataFrame}}
    
Apply the learned parameters to the new data.
"""
function transform!(tsc::TSClassifier, features::DataFrame=DataFrame())
   ldirname = tsc.model[:tstdirectory]
   mdirname = tsc.model[:modeldirectory]
   modelfname=tsc.model[:juliarfmodelname]
   trdata = getstats(ldirname)
   xfeatures = tsc.model[:feature_range]
   X=trdata[:,xfeatures]
   mfeatures=tsc.model[:features]
   (sum(names(X) .== mfeatures ) == length(mfeatures)) || error("features mismatch")
   serializedmodel = joinpath(mdirname,modelfname)
   if isfile(serializedmodel)
      println("loading model from file: "*serializedmodel)
      model=open(serializedmodel,"r") do file
         deserialize(file)
      end
   else
      model= tsc.model[:fmodel]
   end
   mpred = transform!(model,X)
   return DataFrame(fname=trdata.fname,predtype=mpred)
end


@enum TSType begin
  temperature = 1
  weather = 2
  footfall = 3
  AirOffTemp = 4
  Energy = 5
  Pressure = 6
  RetTemp = 7
end

function ispathnotempty(margs::Dict)
  return margs[:trdirectory] != "" && margs[:tstdirectory] != "" && margs[:modeldirectory] != ""
end

# return stat of a file
function getfilestat(ldirname::AbstractString,lfname::AbstractString)
   myregex = r"(?<dtype>[A-Z _ - a-z]+)(?<number>\d*).(?<ext>\w+)"
   m=match(myregex,lfname)
   ext = m[:ext]; dtype=m[:dtype];num = m[:number]
   (dtype != "" && ext != "")  || error("wrong filename format: dtype[n].csv")
   dtype in string.(instances(TSType)) || error(dtype * ", filename does not indicate known data type.")
   # create a pipeline to get stat
   fname = joinpath(ldirname,lfname)
   csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
   valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
   valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
   stfier = Statifier(Dict(:processmissing=>false))
   mpipeline = @pipeline csvfilter |> valgator |> valnner |> stfier
   df = fit_transform!(mpipeline)
   df.dtype = repeat([dtype],nrow(df))
   df.fname = repeat([lfname],nrow(df))
   return (df)
end

function serialloop(ldirname,mfiles)
   trdata = DataFrame()
   for file in mfiles
      try
         df=getfilestat(ldirname,file)
         trdata = vcat(trdata,df)
         println("getting stats of "*file)
      catch errormsg
         println("skipping "*file*": "*string(errormsg))
         continue
      end
   end
   return trdata
end


function threadloop(ldirname,mfiles)
   @eval using Base.Threads
   trdata = DataFrame()
   mutex = SpinLock()
   Base.Threads.@threads for file in mfiles
      try
         df=getfilestat(ldirname,file)
         lock(mutex)
         trdata = vcat(trdata,df)
         println("getting stats of "*file*" on thread:"*string(Base.Threads.threadid()))
         unlock(mutex)
      catch errormsg
         println("skipping "*file*": "*string(errormsg))
      end
   end
   return trdata
end

# loop over the directory and get stats of each file
# return a dataframe containing stat features and ts type for target
function getstats(ldirname::AbstractString)
   ldirname != "" || error("directory name empty")
   mfiles = readdir(ldirname) |> x->filter(y->match(r".csv",y) != nothing,x)
   mfiles != [] || error("empty csv directory")
   #df = serialloop(ldirname,mfiles)
   # get julia version and run threads if julia 1.3
   jversion = string(Base.VERSION)
   df = DataFrame()
   if match(r"^1.5",jversion) === nothing
      df = serialloop(ldirname,mfiles)
   else
      df = threadloop(ldirname,mfiles)
   end
   return df
end

end
