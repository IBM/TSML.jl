```@meta
Author = "Paulito P. Palmes"
```

# Pipeline

Instead of calling `fit!` and `transform!` to process time-series data, we can
use the `Pipeline` transformer which does this automatically by iterating the transformers
passed throught its argument and calling `fit!` and `transform` repeatedly for each transformer.

Let's have a function to generate dataframe with missing data.

```@example pipeline
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

X = generateDataWithMissing()
first(X,20)
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
first(results,10)
```

Using the `Pipeline` transformer, it becomes straightforward to process the
time-series data. It also becomes trivial to extend TSML functionality by
adding more transformers and making sure each support the `fit!` and `transform!`
interfaces. Any new transformer can then be easily added to the `Pipeline` workflow 
without invasively changing existing codes.

## Extending TSML

To illustrate how simple it is to add a new transformer, below illustrates 
extending TSML to support CSV reading which will then be added in the pipeline:

```@example pipeline
using TSML.TSMLTypes
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
```


Instead of passing input X, we will add an instance of the 
`CSVReader` at the start of the array of transformers in the pipeline 
to read the data and pass its output to the other transformers for processing.

```@example pipeline
fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
csvreader = CSVReader(Dict(:filename=>fname,:dateformat=>"d/m/y H:M"))
fit!(csvreader)
csvdata = transform!(csvreader)
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
first(results,10)
```

Notice that there is no more the need to pass X in the arguments of `fit!` and `transform`
because the data is now transmitted by the `CSVReader` instance to the other transformers
in the pipeline.
