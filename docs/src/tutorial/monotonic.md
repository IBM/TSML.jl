```@meta
Author = "Paulito P. Palmes"
```

# Monotonic Detection

One important preprocessing step for time series data processing is the detection 
of monotonic data and transform it to non-monotonic type by using the finite difference
operator.

Let's create an artificial monotonic data and apply our monotonic transformer to normalize it:

```@example mono
using Dates, DataFrames, Random

Random.seed!(123)
mdates = DateTime(2017,12,31,1):Dates.Hour(1):DateTime(2017,12,31,10) |> collect
mvals = rand(length(mdates)) |> cumsum
df =  DataFrame(Date=mdates ,Value = mvals)
```

Now that we have a monotonic data, let's use the `Monotonicer` to normalize it:

```@example mono
using TSML, TSML.Utils, TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML: Monotonicer

mono = Monotonicer(Dict())
fit!(mono,df)
res=transform!(mono,df)
```

## Real Data Example

We will now apply the entire pipeline 
starting from reading csv data, aggregate, impute, and normalize
if it's monotonic. We will consider three 
different data types: a regular time series data, a  
monotonic data, and a daily monotonic data. The difference between  
monotonic and daily monotonic is that the values in daily monotonic resets to 
zero or some baseline and cumulatively increases in a day until the 
next day where it resets to zero or some baseline value. `Monotonicer`
automatically detects these three different types and apply the corresponding
normalization accordingly.

```@example mono
using TSML: DataReader
using TSML: DateValgator, DateValNNer, Statifier, Monotonicer
regularfile = joinpath(dirname(pathof(TSML)),"../data/typedetection/regular.csv")
monofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/monotonic.csv")
dailymonofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/dailymonotonic.csv")

regularfilecsv = DataReader(Dict(:filename=>regularfile,:dateformat=>"dd/mm/yyyy HH:MM"))
monofilecsv = DataReader(Dict(:filename=>monofile,:dateformat=>"dd/mm/yyyy HH:MM"))
dailymonofilecsv = DataReader(Dict(:filename=>dailymonofile,:dateformat=>"dd/mm/yyyy HH:MM"))

valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
stfier = Statifier(Dict(:processmissing=>true))
mono = Monotonicer(Dict())
nothing #hide
```

## Regular TS Processing
Let's test by feeding the regular time series type to the pipeline. We expect that for this type,
`Monotonicer` will not perform further processing:


- Pipeline with `Monotonicer`: regular time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [regularfilecsv,valgator,valnner,mono]
   )
)
fit!(pipeline)
regulardf=transform!(pipeline)
first(regulardf,5)
```

- Pipeline without `Monotonicer`: regular time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [regularfilecsv,valgator,valnner]
   )
)
fit!(pipeline)
regulardf=transform!(pipeline)
first(regulardf,5)
```

Notice that the outputs are the same with or without the `Monotonicer` instance.

## Monotonic TS Processing
Let's now feed the same pipeline with a monotonic csv data.

- Pipeline with `Monotonicer`: monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [monofilecsv,valgator,valnner,mono]
   )
)
fit!(pipeline)
monodf=transform!(pipeline)
first(monodf,10)
```

- Pipeline without `Monotonicer`: monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [monofilecsv,valgator,valnner]
   )
)
fit!(pipeline)
monodf=transform!(pipeline)
first(monodf,10)
```

Notice that without the `Monotonicer` instance, the data becomes monotonic while with
the `Monotonicer` instance in the pipeline, it becomes a regular time series data.

## Daily Monotonic TS Processing
Lastly, let's feed the daily monotonic data using similar pipeline and examine its output.

- Pipeline with `Monotonicer`: daily monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [dailymonofilecsv,valgator,valnner,mono]
   )
)
fit!(pipeline)
dailymonodf=transform!(pipeline)
first(dailymonodf,50)
```

- Pipeline without `Monotonicer`: daily monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [dailymonofilecsv,valgator,valnner]
   )
)
fit!(pipeline)
dailymonodf=transform!(pipeline)
first(dailymonodf,50)
```

Notice that the first 27 rows behave like a regular time series with no monotonic signature.
Only after row 27 the data behaves in a monotonic fashion. 
Notice further that the series reset to baseline value
in row 38 at 1:00 am. This daily monotonic pattern can be seen when the data is plotted.
In the pipeline with `Monotonicer`, the normalization replaces
the baseline values to their immediate neighbor after applying the finite difference operation.
