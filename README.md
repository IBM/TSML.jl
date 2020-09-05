<div align="center"> <img
src="https://ibm.github.io/TSML.jl/tsmllogo/tsmllogo13.png"
alt="TSML Logo" width="250"></img> </div>

| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

### TSML is a package for time series data processing, classification, clustering, and prediction written in [Julia](http://julialang.org/).

The design/framework of this package is influenced heavily by Samuel Jenkins' [Orchestra.jl](https://github.com/svs14/Orchestra.jl) and Paulito Palmes [CombineML.jl](https://github.com/ppalmes/CombineML.jl) packages.

Follow these links for demo/tutorial/paper: 
- [Jupyter Notebook TSML Demo](https://github.com/IBM/TSML.jl/blob/master/docs/StaticPlotting.jl.ipynb)
- [JuliaCon 2019 Proceedings Paper](https://doi.org/10.21105/jcon.00051) [![DOI](https://proceedings.juliacon.org/papers/10.21105/jcon.00051/status.svg)](https://doi.org/10.21105/jcon.00051)

- [TSML Binder Notebooks Live Demo](https://mybinder.org/v2/gh/IBM/TSML.jl/binder_support)

## Package Features

- Support for symbolic pipeline composition of transformers and learners
- TS data type clustering/classification for automatic data discovery
- TS aggregation based on date/time interval
- TS imputation based on `symmetric` Nearest Neighbors
- TS statistical metrics for data quality assessment
- TS ML wrapper with more than 100+ libraries from caret, scikitlearn, and julia
- TS date/value matrix conversion of 1-D TS using sliding windows for ML input
- Common API wrappers for ML libs from JuliaML, PyCall, and RCall
- Pipeline API allows high-level description of the processing workflow
- Specific cleaning/normalization workflow based on data type
- Automatic selection of optimised ML model
- Automatic segmentation of time-series data into matrix form for ML training and  prediction
- Easily extensible architecture by using just two main interfaces: fit and transform
- Meta-ensembles for robust prediction
- Support for threads and distributed computation for scalability, and speed

## Installation
TSML is in the Julia Official package registry. The latest release can be installed at the Julia prompt using Julia's package management which is triggered by pressing `]` at the julia prompt:

```julia
julia> ]
(v1.1) pkg> add TSML
```

Or, equivalently, via the `Pkg` API:

```julia
julia> using Pkg
julia> Pkg.add("TSML")
```

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **documentation of the most recently tagged version.**
- [**DEVEL**][docs-dev-url] &mdash; *documentation of the in-development version.*

## Project Status

TSML is tested and actively developed on Julia `1.0` and above for Linux and macOS.

There is no support for Julia versions `0.4`, `0.5`, `0.6` and `0.7`.

## Overview

TSML (Time Series Machine Learning) is a package for Time Series data processing, classification, and prediction. It combines ML libraries from Python's ScikitLearn, R's Caret, and Julia using a common API and allows seamless ensembling and integration of heterogenous ML libraries to create complex models for robust time-series prediction.

## Motivations
Over the past years, the industrial sector has seen
many innovations brought about by automation.
Inherent in this automation is the installation of
sensor networks for status monitoring and data collection.
One of the major challenges in these data-rich
environments is how to extract and exploit
information from these large volume of data to
detect anomalies, discover patterns to reduce
downtimes and manufacturing errors, reduce energy usage, etc.

To address these issues, we developed TSML package.
It leverages AI and ML libraries from ScikitLearn, Caret,
and Julia as building blocks in processing huge amount of
industrial times series data. It has the following characteristics
described below.

## Main Workflow

The package assumes a two-column input composed of Dates and Values. The first part of the workflow aggregates values based on the specified date/time interval which minimizes occurrence of missing values and noise. The aggregated data is then left-joined to the complete sequence of dates in a specified date/time interval. Remaining missing values are replaced by `k` nearest neighbors where `k` is the `symmetric` distance from the location of missing value. This approach can be called several times until there are no more missing values.

TSML uses a pipeline of filters and transformers which iteratively calls the __fit__ and __transform__ families of functions relying on multiple dispatch to select the correct algorithm from the steps outlined above.

TSML supports transforming time series data into matrix form for ML training and prediction. `Dateifier` filter extracts the date features and convert the values into matrix form parameterized by the _size_ and _stride_ of the sliding window representing the dimension of the input for ML training and prediction. Similar workflow is done by the `Matrifier` filter to convert the time series values into matrix form.

The final part combines the dates matrix with the values matrix to become input of the ML with the output representing the values of the time periods to be predicted ahead of time.

Machine learning functions in TSML are wrappers to the corresponding Scikit-learn, Caret, and native Julia ML libraries. There are more than hundred classifiers and regression functions available using a common API. 

Below are examples of the `Pipeline` workflow.

Generally, you will need the different transformers and utils in TSML for time-series processing. To use them, it is standard in TSML code to have the following declared at the topmost part of your application:

- #### Load TSML and setup filters/transformers
```julia
# Setup source data and filters to aggregate and impute hourly
using TSML 

fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
csvreader = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1))) # aggregator
valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))   # imputer
stfier = Statifier(Dict(:processmissing=>true))             # get statistics
mono = Monotonicer(Dict()) # normalize monotonic data
outnicer = Outliernicer(Dict(:dateinterval => Dates.Hour(1))) # normalize outliers
plotter = Plotter() # visualize output
```

- #### Setup pipeline to load csv data, aggregate, and get statistics
```julia
# Setup pipeline without imputation and run
mpipeline1 = @pipeline csvreader |> valgator |> stfier

stats1=fit_transform!(mpipeline1)

# Show statistics including blocks of missing data stats
show(stats1, allcols=true)
```
Note: fit_transform! is equivalent to calling `fit!` and `transform!` functions.

 - #### Load csv data, aggregate, impute, and get statistics
```julia
# Add imputation in the pipeline and rerun
mpipeline2 = @pipeline csvreader |> valgator |> valnner |> stfier

stats2 = fit_transform!(mpipeline2)

# Show statistics including blocks of missing data stats
show(stats2, allcols=true)
```

- #### Load csv data, aggregate, impute, normalize monotonic data, and plot
```julia
# Add imputation in the pipeline, and plot 
mpipeline2 = @pipeline csvreader |> valgator |> valnner |> mono |> plotter

fit_transform!(mpipeline2)
```
Note: It may take some time for the graph to render because just-in-time
compilation kicks-in and plot package takes a bit of time to be pre-compiled.
Suceeding plots will be much faster because Julia uses the pre-compiled image.

- #### Extracting TimeSeries Date,Values into Matrix Form for ML Modeling
```julia
# let's setup date,value dataframe as input
datn = DateTime(2018,1,1):Dates.Day(1):DateTime(2019,1,31) |> collect
valn = rand(1:100,length(datn))
ts = DataFrame(Date=datn,Value=valn)

@show first(ts,5);

args = Dict(:ahead=>24,:size=>24,:stride=>5)
dtfier = Dateifier(args)
mtfier = Matrifier(args)

# setup pipeline concatenating matrified dates with matrified values
ppl = @pipeline dtfier + mtfier

dateval = fit_transform!(ppl,ts)
@show first(dateval,5);
```
We can use the matrified dateval as input features for prediction/classication.
Let's create a dummy response consisting of `yes` or `no` and use Random Forest
to learn the mapping.
```julia
target = rand(["yes","no"],nrow(dateval)) 

rf = RandomForest()

accuracy(x,y) = score(:accuracy,x,y)

crossvalidate(rf,dateval,target,accuracy)
# sample output:
fold: 1, 14.285714285714285
fold: 2, 57.14285714285714
fold: 3, 71.42857142857143
fold: 4, 85.71428571428571
fold: 5, 57.14285714285714
fold: 6, 57.14285714285714
fold: 7, 57.14285714285714
fold: 8, 71.42857142857143
fold: 9, 42.857142857142854
fold: 10, 71.42857142857143
(mean = 58.57142857142857, std = 19.57600456294711, folds = 10)
```

## Feature Requests and Contributions

We welcome contributions, feature requests, and suggestions. Here is the link to open an [issue][issues-url] for any problems you encounter. If you want to contribute, please follow the guidelines in [contributors page][contrib-url].

## Help usage

Usage questions can be posted in:
- [Julia Community](https://julialang.org/community/) 
- [Gitter TSML Community][gitter-url]
- [Julia Discourse forum][discourse-tag-url]


[contrib-url]: https://github.com/IBM/TSML.jl/blob/master/CONTRIBUTORS.md
[issues-url]: https://github.com/IBM/TSML.jl/issues

[discourse-tag-url]: https://discourse.julialang.org/

[gitter-url]: https://gitter.im/TSMLearning/community
[gitter-img]: https://badges.gitter.im/ppalmes/TSML.jl.svg

[slack-img]: https://img.shields.io/badge/chat-on%20slack-yellow.svg
[slack-url]: https://julialang.slack.com


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ibm.github.io/TSML.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://ibm.github.io/TSML.jl/dev/

[travis-img]: https://travis-ci.org/IBM/TSML.jl.svg?branch=master
[travis-url]: https://travis-ci.org/IBM/TSML.jl

[codecov-img]: https://codecov.io/gh/IBM/TSML.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/IBM/TSML.jl

