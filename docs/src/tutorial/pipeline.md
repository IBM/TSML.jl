```@meta
Author = "Paulito P. Palmes"
```

# Pipeline

Instead of calling `fit!` and `transform!` for each transformer to process time series data, we can
use the `Pipeline` transformer which does this automatically by iterating through the transformers
and calling `fit!` and `transform!` repeatedly for each transformer in its argument.

Let's start again by using a function to generate a time series dataframe with some missing data.

```@setup pipeline
using Random, Dates, DataFrames
function generateDataWithMissing()
   Random.seed!(123)
   gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
   gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
   gmissing = 50000
   gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
   df = DataFrame(Date=gdate,Value=gval)
   df[:Value][gndxmissing] .= missing
   return df
end
```

```@repl pipeline
X = generateDataWithMissing();
first(X,15)
```

## Workflow of Pipeline

Let's use the pipeline transformer to aggregate and impute:

```@example pipeline
using Dates
using TSML
using TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML: Pipeline
using TSML: DateValgator
using TSML: DateValNNer

dtvalgator = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
dtvalnner = DateValNNer(Dict(:dateinterval => Dates.Hour(1)))

mypipeline = Pipeline(
  Dict( :transformers => [
            dtvalgator,
            dtvalnner
         ]
  )
)

fit!(mypipeline,X)
results = transform!(mypipeline,X)
nothing #hide
```

```@repl pipeline
first(results,10)
```

Using the `Pipeline` transformer, it becomes straightforward to process the
time series data. It also becomes trivial to extend TSML functionality by
adding more transformers and making sure each support the `fit!` and `transform!`
interfaces. Any new transformer can then be easily added to the `Pipeline` workflow 
without invasively changing the existing codes.

## Extending TSML

To illustrate how simple it is to add a new transformer, below extends
TSML by adding `CSVReader` transformer and added in the pipeline to process CSV data:

```@example pipeline
using TSML.TSMLTypes
using TSML.Utils
import TSML.TSMLTypes.fit!
import TSML.TSMLTypes.transform!

using CSV

mutable struct CSVReader <: Transformer
    model
    args
    function CSVReader(args=Dict())
        default_args = Dict(
            :filename => "",
            :dateformat => ""
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(csvrdr::CSVReader,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}
    fname = csvrdr.args[:filename]
    fmt = csvrdr.args[:dateformat]
    (fname != "" && fmt != "") || error("missing filename or date format")
    model = csvrdr.args
end

function transform!(csvrdr::CSVReader,x::T=[]) where {T<:Union{DataFrame,Vector,Matrix}}
    fname = csvrdr.args[:filename]
    fmt = csvrdr.args[:dateformat]
    df = CSV.read(fname)
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    df[:Date] = DateTime.(df[:Date],fmt)
    df
end
nothing #hide
```

Instead of passing table X that contains the time series, we will add 
an instance of the`CSVReader` at the start of the array of transformers in the pipeline 
to read the csv data. CSVReader `transform!` function converts the csv time series table
into a dataframe, which will be consumed by the next transformer in the pipeline 
for processing.

```@example pipeline
fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
csvreader = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"d/m/y H:M"))
fit!(csvreader)
csvdata = transform!(csvreader)
nothing #hide
```

```@repl pipeline
first(csvdata,10)
```

Let us now include the newly created `CSVReader` in the pipeline to read the csv data
and process it by aggregation and imputation.


```@example pipeline
mypipeline = Pipeline(
  Dict( :transformers => [
            csvreader,
            dtvalgator,
            dtvalnner
         ]
  )
)

fit!(mypipeline)
results = transform!(mypipeline)
nothing #hide
```

```@repl pipeline
first(results,10)
```

Notice that there is no more the need to pass X in the arguments of `fit!` and `transform`
because the data is now transmitted by the `CSVReader` instance to the other transformers
in the pipeline.
