module TSML

# reexport some needed functions from packages to Main
include("pkgdeps.jl")

export fit!, transform!,fit_transform!

using AutoMLPipeline
using AutoMLPipeline: AbsTypes, Utils, Baselines, Pipelines
using AutoMLPipeline: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AutoMLPipeline: EnsembleMethods

export Machine, Learner, Transformer, Workflow, Computer
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples, 
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris,getprofb,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing
export Baseline, Identity
export Imputer,OneHotEncoder
export PrunedTree,RandomForest,Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline @pipelinex, @pipelinez
export Pipeline, ComboPipeline

import AutoMLPipeline.AbsTypes: fit!, transform!

include("valdatefilters.jl")
using .ValDateFilters
export Matrifier,Dateifier,
       DateValizer,DateValgator,DateValNNer,DateValMultiNNer,
       CSVDateValReader, CSVDateValWriter, DateValLinearImputer
       #BzCSVDateValReader
export impute, impute!, chain, chain!, drop, drop!, 
       interp, interp!, ImputeError, locf, nocb

include("statifier.jl")
using .Statifiers
export Statifier,tsmlfullstat

include("mlbase.jl")
using .MLBaseWrapper
export Standardize,standardize, standardize!, 
       estimate, transform,StandardScaler

include("monotonicer.jl")
using .Monotonicers
export Monotonicer,ismonotonic,dailyflips

include("cliwrapper.jl")
using .CLIWrappers
export tsmlrun

include("tsclassifier.jl")
using .TSClassifiers
export TSClassifier, getstats

include("outliernicer.jl")
using .Outliernicers
export Outliernicer

include("normalizer.jl")
using .Normalizers
export Normalizer

include("svm.jl")
using .SVMModels
export SVMModel

include("timescaledb.jl")
using .TimescaleDBs
export TimescaleDB

include("argparse.jl")
using .ArgumentParsers
export tsmlmain

include("plotter.jl") 
using .Plotters
export Plotter

include("demo.jl")
using .TSMLDemo
export tsml_demo


#include("schema.jl")
#using .Schemalizers
#export Schemalizer, ML, table


end # module
