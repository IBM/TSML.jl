module TSML

# reexport some needed functions from packages to Main
include("pkgdeps.jl")

export fit, fit!, transform, transform!,fit_transform, fit_transform!

using AMLPipelineBase
using AMLPipelineBase: AbsTypes, Utils, BaselineModels, Pipelines
using AMLPipelineBase: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AMLPipelineBase: EnsembleMethods, CrossValidators
using AMLPipelineBase: NARemovers

export Machine, Learner, Transformer, Workflow, Computer
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples, 
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris,getprofb,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing
export Baseline, Identity
export Imputer,OneHotEncoder,Wrapper
export PrunedTree,RandomForest,Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline @pipelinex, @pipelinez
export +, |>, *, |, >>
export Pipeline, ComboPipeline

import AMLPipelineBase.AbsTypes: fit, fit!, transform, transform!

# ----------------------------------------------

include("valdatefilters.jl")
using .ValDateFilters
export Matrifier,Dateifier,
       DateValizer,DateValgator,DateValNNer,DateValMultiNNer,
       CSVDateValReader, CSVDateValWriter, DateValLinearImputer
       #BzCSVDateValReader
export impute, impute!,interp, interp!, locf, nocb

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

#include("svm.jl")
#using .SVMModels
#export SVMModel

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
