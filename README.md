<div align="center"> <img
src="https://ibm.github.io/TSML.jl/tsmllogo/tsmllogo13.png"
alt="TSML Logo" width="250"></img> </div>

| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

### TSML is a package for time series data processing, classification, clustering, and prediction written in [Julia](http://julialang.org/).

The design/framework of this package is influenced heavily by Samuel Jenkins' [Orchestra.jl](https://github.com/svs14/Orchestra.jl) and Paulito Palmes [CombineML.jl](https://github.com/ppalmes/CombineML.jl) packages.

Follow this link for a quick [Jupyter Notebook TSML Demo](https://github.com/IBM/TSML.jl/blob/master/docs/StaticPlotting.jl.ipynb).

## Package Features

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
- Support for distributed computation, for scalability, and speed

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

The next part extracts the date features and convert the values into matrix form parameterized by the _size_ and _stride_ of the sliding window representing the dimension of the input for ML training and prediction.

The final part combines the date features and the matrix of values as input to the ML with the output representing the values of the time periods to be predicted ahead of time.

TSML uses a pipeline which iteratively calls the __fit__ and __transform__ families of functions relying on multiple dispatch to select the correct algorithm from the steps outlined above.

Machine learning functions in TSML are wrappers to the corresponding Scikit-learn, Caret, and native Julia ML libraries. There are more than hundred classifiers and regression functions available using a common API. 

Below are examples of the `Pipeline` workflow.

Generally, you will need the different transformers and utils in TSML for time-series processing. To use them, it is standard in TSML code to have the following declared at the topmost part of your application:

- #### Load TSML and supporting submodules
```julia
using TSML 
using TSML.TSMLTransformers
using TSML.TSMLTypes
using TSML.Utils
```

- #### Setup different transformers
```julia
using TSML: DataReader, DateValgator, DateValNNer
using TSML: Statifier, Monotonicer, Outliernicer

# Setup source data and filters to aggregate and impute hourly
fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")

csvreader = DataReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1))) # aggregator
valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))   # imputer
stfier = Statifier(Dict(:processmissing=>true))             # get statistics
mono = Monotonicer(Dict()) # normalize monotonic data
outnicer = Outliernicer(Dict(:dateinterval => Dates.Hour(1))) # normalize outliers
```

- #### Load csv data, aggregate, and get statistics
```julia
# Setup pipeline without imputation and run
mpipeline1 = Pipeline(Dict(
  :transformers => [csvreader,valgator,stfier]
 )
)
fit!(mpipeline1)
respipe1 = transform!(mpipeline1)

# Show statistics including blocks of missing data stats
@show respipe1
```

 - #### Load csv data, aggregate, impute, and get statistics
```julia
# Add imputation in the pipeline and rerun
mpipeline2 = Pipeline(Dict(
  :transformers => [csvreader,valgator,valnner,stfier]
 )
)
fit!(mpipeline2)
respipe2 = transform!(mpipeline2)

# Show statistics including blocks of missing data stats
@show respipe2
```

- #### Load csv data, aggregate, impute, and normalize outliers
```julia
# Add imputation in the pipeline and rerun
mpipeline2 = Pipeline(Dict(
  :transformers => [csvreader,valgator,valnner,outnicer]
 )
)
fit!(mpipeline2)
respipe2 = transform!(mpipeline2)

# Show statistics including blocks of missing data stats
@show respipe2
```

- #### Load csv data, aggregate, impute, and normalize monotonic data
```julia
# Add imputation in the pipeline and rerun
mpipeline2 = Pipeline(Dict(
  :transformers => [csvreader,valgator,valnner,mono]
 )
)
fit!(mpipeline2)
respipe2 = transform!(mpipeline2)

# Show statistics including blocks of missing data stats
@show respipe2
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
[docs-dev-url]: https://ibm.github.io/TSML.jl/latest/

[travis-img]: https://travis-ci.org/ppalmes/TSML.jl.svg?branch=master
[travis-url]: https://travis-ci.org/ppalmes/TSML.jl

[codecov-img]: https://codecov.io/gh/IBM/TSML.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/IBM/TSML.jl
