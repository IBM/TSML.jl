module TSML

using AutoMLPipeline
using AutoMLPipeline.BaseFilters
import AutoMLPipeline.AbsTypes: fit!, transform!

using AutoMLPipeline.AbsTypes
export Machine, Learner, Transformer, Workflow, Computer

using AutoMLPipeline.Utils
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples, 
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing

export fit!, transform!,fit_transform!

# reexport common functions to Main
include("pkgdeps.jl")
using .PkgDeps

include("imputer/Imputers.jl")
using .Imputers

include("baseline.jl")
using .BaselineAlgos
export Baseline,Identity

using AutoMLPipeline.BaseFilters: Imputer, OneHotEncoder, Wrapper
export Imputer,OneHotEncoder,Wrapper

include("valdatefilters.jl")
using .ValDateFilters
export Matrifier,Dateifier,
       DateValizer,DateValgator,DateValNNer,DateValMultiNNer,
       CSVDateValReader, CSVDateValWriter, DateValLinearImputer,
       BzCSVDateValReader

include("statifier.jl")
using .Statifiers
export Statifier,tsmlfullstat

include("mlbase.jl")
using .MLBaseWrapper
export Standardize,standardize, standardize!, 
       transform, estimate, transform,StandardScaler

using AutoMLPipeline.DecisionTreeLearners
export PrunedTree,RandomForest,Adaboost

include("normalizer.jl")
using .Normalizers
export Normalizer

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

include("plotter.jl")
using .Plotters
export Plotter

include("timescaledb.jl")
using .TimescaleDBs
export TimescaleDB

using AutoMLPipeline.EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

using AutoMLPipeline.FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

using AutoMLPipeline.Pipelines
export @pipeline @pipelinex
export Pipeline, ComboPipeline

include("schema.jl")
using .Schemalizers
export Schemalizer, ML, table

using AutoMLPipeline.CrossValidators
export crossvalidate

include("argparse.jl")
using .ArgumentParsers
export tsmlmain

include("demo.jl")
using .TSMLDemo
export tsml_demo
end # module
