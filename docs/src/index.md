```@meta
Author = "Paulito P. Palmes"
```

# TSML (Time-Series Machine Learning)

TSML (Time Series Machine Learning) is a package 
for Time Series data processing, classification,
and prediction. It combines ML libraries from Python's 
ScikitLearn, R's Caret, and Julia ML using a common API 
and allows seamless ensembling and integration of 
heterogenous ML libraries to create complex models 
for robust time-series pre-processing and prediction/classification.

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
industrial time series data. It has the following characteristics 
described below.

## Package Features

- TS data type clustering/classification for automatic data discovery
- TS aggregation based on date/time interval
- TS imputation based on Nearest Neighbors
- TS statistical metrics for data quality assessment
- TS ML wrapper more than 100+ libraries from caret, scikitlearn, and julia
- TS date/value matrix conversion of 1-D TS using sliding windows for ML input
- Common API wrappers for ML libs from JuliaML, PyCall, and RCall
- Pipeline API allows high-level description of the processing workflow
- Specific cleaning/normalization workflow based on data type
- Automatic selection of optimised ML model
- Automatic segmentation of time-series data into matrix form for ML training and  prediction
- Easily extensible architecture by using just two main interfaces: fit and transform
- Meta-ensembles for robust prediction
- Support for distributed computation for scalability and speed


## Installation

TSML is in the Julia Official package registry. 
The latest release can be installed at the Julia 
prompt using Julia's package management which is triggered
by pressing `]` at the julia prompt:
```julia
julia> ]
(v1.0) pkg> add TSML
```

or

```julia
julia> using Pkg
julia> pkg"add TSML"
```

or

```julia
julia> using Pkg
julia> Pkg.add("TSML")
```

or 

```julia
julia> pkg"add TSML"
```
Once TSML is installed, you can load the TSML package by:

```julia
julia> using TSML
```

or 

```julia
julia> import TSML
```
Generally, you will need the different transformers and utils in TSML for
time-series processing. To use them, it is standard in TSML code to have the
following declared at the topmost part of your application:

```julia
using TSML 
using TSML.TSMLTransformers
using TSML.TSMLTypes
using TSML.Utils
```

## Tutorial Outline
```@contents
Pages = [
  "tutorial/aggregators.md",
  "tutorial/pipeline.md",
  "tutorial/statistics.md",
  "tutorial/monotonic_plotting.md",
  "tutorial/tsclassifier.md"
]
Depth = 3
```


## Manual Outline
```@contents
Pages = [
  "man/valueproc.md",
  "man/dateproc.md",
  "man/aggregation.md",
  "man/imputation.md",
]
Depth = 3
```

## ML Library
```@contents
Pages = [
  "lib/decisiontree.md",
  "lib/functions.md"
]
```

```@index
```
