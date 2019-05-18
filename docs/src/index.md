```@meta
Author = "Paulito P. Palmes"
```

# TSML (Time-Series Machine Learning)

TSML (Time Series Machine Learning) is package 
for Time Series data processing, classification,
and prediction. It combines ML libraries from Python's 
ScikitLearn, R's Caret, and Julia ML using a common API 
and allows seamless ensembling and integration of 
heterogenous ML libraries to create complex models 
for robust time-series pre-processing and prediction/classification.

## Package Features

- TS aggregation based on time/date interval
- TS imputation based on Nearest Neighbors
- TS statistical metrics of data quality
- TS classification for automatic data discovery
- TS prediction with more than 100+ libraries from caret, scikitlearn, and julia
- TS date/val matrix conversion of 1-d TS using sliding windows for ML input
- Pipeline API allows high-level description of the processing workflow
- Easily extensible architecture by using just two main interfaces: fit and transform
- Support for hundreds of external ML libs from Scikitlearn and Caret by using common API wrappers for PyCall and RCall


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
  "tutorial/monotonic.md",
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
Pages = ["lib/decisiontree.md"]
```

```@index
```
