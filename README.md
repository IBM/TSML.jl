<div align="center">

![Visitor](https://visitor-badge.laobi.icu/badge?page_id=ppalmes.TSML.jl)

<img
src="https://ibm.github.io/TSML.jl/tsmllogo/tsmllogo13.png"
alt="TSML Logo" width="250">
</img>

[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7094/badge)](https://bestpractices.coreinfrastructure.org/projects/7094)

![Overall Stats](https://github-readme-stats.vercel.app/api?username=ppalmes&count_private=true&show_icons=true&hide=contribs)

|                             **Documentation**                             |                    **Build Status**                     |                         **Help**                          |
| :-----------------------------------------------------------------------: | :-----------------------------------------------------: | :-------------------------------------------------------: |
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][gha-img]][gha-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

</div>

#### Stargazers over time

[![Stargazers over time](https://starchart.cc/IBM/TSML.jl.svg)](https://starchart.cc/IBM/TSML.jl)

### TSML (Timeseries Machine Learning) ![Visitor](https://visitor-badge.laobi.icu/badge?page_id=ppalmes.TSML.jl)

---

**TSML** is a package for time series data
processing, classification, clustering,
and prediction. It combines ML libraries
from Python's ScikitLearn (thru its complementary
[AutoMLPipeline](https://github.com/IBM/AutoMLPipeline.jl)
package) and Julia MLs using
a common API and allows seamless ensembling
and integration of heterogenous ML libraries
to create complex models for robust time-series prediction.
The design/framework of this package is influenced heavily
by Samuel Jenkins' [Orchestra.jl](https://github.com/svs14/Orchestra.jl)
and [CombineML.jl](https://github.com/ppalmes/CombineML.jl) packages.
**TSML** is actively developed and tested in `Julia 1.0`
and above for Linux, MacOS, and Windows.

Links to **TSML** demo, tutorial, and published JuliaCon paper:

- [TSML Binder Notebooks Live Demo](https://mybinder.org/v2/gh/IBM/TSML.jl/binder_support)
- [Jupyter Notebook TSML Demo](https://github.com/IBM/TSML.jl/blob/master/docs/notebooks/StaticPlotting.jl.ipynb)
- [JuliaCon 2019 Proceedings Paper](https://doi.org/10.21105/jcon.00051) [![DOI](https://proceedings.juliacon.org/papers/10.21105/jcon.00051/status.svg)](https://doi.org/10.21105/jcon.00051)

#### Package Features

- Support for symbolic pipeline composition of transformers and learners
- TS data type clustering/classification for automatic data discovery
- TS aggregation based on date/time interval
- TS imputation based on `symmetric` Nearest Neighbors
- TS statistical metrics for data quality assessment
- TS ML wrapper with more than 100+ libraries from scikitlearn and julia
- TS date/value matrix conversion of 1-D TS using sliding windows for ML input
- Common API wrappers for ML libs from JuliaML, PyCall, and RCall
- Pipeline API allows high-level description of the processing workflow
- Specific cleaning/normalization workflow based on data type
- Automatic selection of optimised ML model
- Automatic segmentation of time-series data into matrix form for ML training and prediction
- Easily extensible architecture by using just two main interfaces: fit and transform
- Meta-ensembles for robust prediction
- Support for threads and distributed computation for scalability, and speed

#### Installation

**TSML** is in the Julia Official package registry.
The latest release can be installed at the Julia
prompt using Julia's package management
which is triggered by pressing `]` at the Julia prompt:

```julia
julia> ]
(v1.1) pkg> add TSML
```

Or, equivalently, via the `Pkg` API:

```julia
julia> using Pkg
julia> Pkg.add("TSML")
```

#### Motivations

Over the past years, the industrial sector has seen
many innovations brought about by automation.
Inherent in this automation is the installation of
sensor networks for status monitoring and data collection.
One of the major challenges in these data-rich
environments is how to extract and exploit
information from these large volume of data to
detect anomalies, discover patterns to reduce
downtimes and manufacturing errors, reduce energy usage, etc.

To address these issues, we developed **TSML** package.
It leverages AI and ML libraries from ScikitLearn
and Julia as building blocks in processing huge amount of
industrial times series data. It has the following characteristics
described below.

#### Main Workflow

The package assumes a two-column input composed of Dates and Values.
The first part of the workflow aggregates values based on the specified
date/time interval which minimizes occurrence of missing values and noise.
The aggregated data is then left-joined to the complete sequence of dates
in a specified date/time interval. Remaining missing values are replaced
by `k` nearest neighbors where `k` is the `symmetric` distance from the
location of missing value. This approach can be called several times until
there are no more missing values.

**TSML** uses a pipeline of filters and transformers which iteratively calls
the `fit!` and `transform!` families of functions relying on multiple
dispatch to select the correct algorithm from the steps outlined above.

**TSML** supports transforming time series data into matrix form for
ML training and prediction. `Dateifier` filter extracts the date
features and convert the values into matrix form parameterized by
the _size_ and _stride_ of the sliding window representing the
dimension of the input for ML training and prediction. Similar
workflow is done by the `Matrifier` filter to convert the time
series values into matrix form.

The final part combines the dates matrix with the values matrix to
become input of the ML with the output representing the values
of the time periods to be predicted ahead of time.

Machine learning functions in **TSML** are wrappers to the
corresponding Scikit-learn and native Julia ML libraries.
There are more than hundred classifiers and regression
functions available using a common API. In order to access these
Scikit-learn wrappers, one should load the related package
called [AutoMLPipeline](https://github.com/IBM/AutoMLPipeline.jl).

Below are examples of the `Pipeline` workflow.

- ##### Load TSML and setup filters/transformers

```julia
# Setup source data and filters to aggregate and impute hourly
using TSML

fname        = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
csvread      = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
aggregate    = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))   # aggregator
impute       = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))    # imputer
chkstats     = Statifier(Dict(:processmissing=>true))             # get statistics
normtonic    = Monotonicer(Dict()) # normalize monotonic data
chkoutlier   = Outliernicer(Dict(:dateinterval => Dates.Hour(1))) # normalize outliers
```

- ##### Pipeline to load csv data

```julia
pipexpr = csvread
data    = fit_transform!(pipexpr)
first(data,5)

5×2 DataFrame
│ Row │ Date                │ Value   │
│     │ DateTime            │ Float64 │
├─────┼─────────────────────┼─────────┤
│ 1   │ 2014-01-01T00:06:00 │ 10.0    │
│ 2   │ 2014-01-01T00:18:00 │ 10.0    │
│ 3   │ 2014-01-01T00:29:00 │ 10.0    │
│ 4   │ 2014-01-01T00:40:00 │ 9.9     │
│ 5   │ 2014-01-01T00:51:00 │ 9.9     │
```

- ##### Pipeline to aggregate and check statistics

```julia
pipexpr = csvread |> aggregate |> chkstats
stats   = fit_transform!(pipexpr)

1×26 DataFrame. Omitted printing of 19 columns
│ Row │ tstart              │ tend                │ sfreq    │ count │ max     │ min     │ median  │
│     │ DateTime            │ DateTime            │ Float64  │ Int64 │ Float64 │ Float64 │ Float64 │
├─────┼─────────────────────┼─────────────────────┼──────────┼───────┼─────────┼─────────┼─────────┤
│ 1   │ 2014-01-01T00:00:00 │ 2015-01-01T00:00:00 │ 0.999886 │ 3830  │ 18.8    │ 8.5     │ 10.35   │
```

Note: `fit_transform!` is equivalent to calling in sequence `fit!` and `transform!` functions.

- ##### Pipeline to aggregate, impute, and check stats

```julia
pipexpr = csvread |> aggregate |> impute |> chkstats
stats2  = fit_transform!(pipexpr)

1×26 DataFrame. Omitted printing of 19 columns
│ Row │ tstart              │ tend                │ sfreq    │ count │ max     │ min     │ median  │
│     │ DateTime            │ DateTime            │ Float64  │ Int64 │ Float64 │ Float64 │ Float64 │
├─────┼─────────────────────┼─────────────────────┼──────────┼───────┼─────────┼─────────┼─────────┤
│ 1   │ 2014-01-01T00:00:00 │ 2015-01-01T00:00:00 │ 0.999886 │ 8761  │ 18.8    │ 8.5     │ 10.0    │
```

- ##### Pipeline to aggregate, impute, and normalize monotonic data

```julia
pipexpr = csvread |> aggregate |> impute |> normtonic
fit_transform!(pipexpr)

8761×2 DataFrame
│ Row  │ Date                │ Value    │
│      │ DateTime            │ Float64? │
├──────┼─────────────────────┼──────────┤
│ 1    │ 2014-01-01T00:00:00 │ 10.0     │
│ 2    │ 2014-01-01T01:00:00 │ 9.9      │
│ 3    │ 2014-01-01T02:00:00 │ 10.0     │
│ 4    │ 2014-01-01T03:00:00 │ 10.0     │
│ 5    │ 2014-01-01T04:00:00 │ 10.0     │
│ 6    │ 2014-01-01T05:00:00 │ 10.0     │
│ 7    │ 2014-01-01T06:00:00 │ 10.0     │
⋮
```

- ##### Transforming timeseries data into matrix form for ML Modeling

```julia
# create artificial timeseries data
datets  = DateTime(2018,1,1):Dates.Day(1):DateTime(2019,1,31) |> collect
valuets = rand(1:100,length(datets))
ts      = DataFrame(Date=datets,Value=valuets)
@show first(ts,5);

5×2 DataFrame
│ Row │ Date                │ Value │
│     │ DateTime            │ Int64 │
├─────┼─────────────────────┼───────┤
│ 1   │ 2018-01-01T00:00:00 │ 56    │
│ 2   │ 2018-01-02T00:00:00 │ 93    │
│ 3   │ 2018-01-03T00:00:00 │ 40    │
│ 4   │ 2018-01-04T00:00:00 │ 15    │
│ 5   │ 2018-01-05T00:00:00 │ 78    │
```

```julia
# Pipeline to concatinate matrified value and date series
args     = Dict(:ahead => 24,:size => 24,:stride => 5)
datemtr  = Dateifier(args)
valuemtr = Matrifier(args)
ppl      = datemtr + valuemtr
dateval  = fit_transform!(ppl,ts)
first(dateval,5)

5×33 DataFrame. Omitted printing of 21 columns
│ Row │ year  │ month │ day   │ hour  │ week  │ dow   │ doq   │ qoy   │ x1    │ x2    │ x3    │ x4    │
│     │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │ Int64 │
├─────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ 1   │ 2019  │ 1     │ 7     │ 0     │ 2     │ 1     │ 7     │ 1     │ 94    │ 97    │ 18    │ 76    │
│ 2   │ 2019  │ 1     │ 2     │ 0     │ 1     │ 3     │ 2     │ 1     │ 99    │ 93    │ 65    │ 68    │
│ 3   │ 2018  │ 12    │ 28    │ 0     │ 52    │ 5     │ 89    │ 4     │ 88    │ 8     │ 59    │ 1     │
│ 4   │ 2018  │ 12    │ 23    │ 0     │ 51    │ 7     │ 84    │ 4     │ 76    │ 5     │ 6     │ 92    │
│ 5   │ 2018  │ 12    │ 18    │ 0     │ 51    │ 2     │ 79    │ 4     │ 6     │ 54    │ 66    │ 72    │
```

- ##### ML Modeling and Prediction
  We can use the matrified dateval as input features for prediction/classication.
  Let's create a dummy response consisting of `yes` or `no` and use Random Forest
  to learn the mapping. More examples of ML modeling can be found in TSML's
  complementary packages: [AutoMLPipeline](https://github.com/IBM/AutoMLPipeline.jl) and
  [AMLPipelineBase](https://github.com/IBM/AMLPipelineBase.jl).

```julia
target        = rand(["yes","no"],nrow(dateval))
rf            = RandomForest()
accuracy(x,y) = score(:accuracy,x,y)
crossvalidate(rf,dateval,target,accuracy)

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

## Extending TSML

If you want to add your own filter or transformer or learner,
take note that `filters` and `transformers` process the
input features but ignores the output argument. On the other hand,
`learners` process both their input and output arguments during `fit!`
while `transform!` expects one input argument in all cases.

The first step is to import the abstract types and define your own mutable structure
as subtype of either Learner or Transformer. Next is to import the `fit!` and
`transform!` functions so that you can overload them. Also, you must
load the DataFrames package because it is the main format for data processing.
Finally, implement your own `fit` and `transform` and export them.

```julia
  using DataFrames
  using TSML.AbsTypes

  # import functions for overloading
  import TSML.AbsTypes: fit!, transform!

  # export the new definitions for dynamic dispatch
  export fit!, transform!, MyFilter

  # define your filter structure
  mutable struct MyFilter <: Transformer
    name::String
    model::Dict
    args::Dict
    function MyFilter(args::Dict())
        ....
    end
  end

# define your fit! function.
  function fit!(fl::MyFilter, inputfeatures::DataFrame, target::Vector=Vector())
       ....
  end

  #define your transform! function
  function transform!(fl::MyFilter, inputfeatures::DataFrame)::DataFrame
       ....
  end
```

Remember that the main format to exchange data is dataframe which requires `transform!`
output to return a dataframe. The features as input for fit! and transform! shall
be in dataframe format too. This is necessary so that
the pipeline passes the dataframe format consistently to
its corresponding filters or transformers or learners. Once you have
create this transformer, you can use plug is as part of the pipeline element
together with the other learners and transformers.

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
[travis-img]: https://github.com/IBM/TSML.jl/actions/workflows/ci.yml/badge.svg
[travis-url]: https://github.com/IBM/TSML.jl/actions/workflows/ci.yml
[codecov-img]: https://codecov.io/gh/IBM/TSML.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/IBM/TSML.jl
