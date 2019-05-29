```@meta
Author = "Paulito P. Palmes"
```

# Monotonic Detection and Plotting

One important preprocessing step for time series data processing is the detection 
of monotonic data and transform it to non-monotonic type by using the finite difference
operator.

## Artificial Data Example

Let's create an artificial monotonic data and apply our monotonic transformer to normalize it.
We can use the `Plotter` filter to visualize the generated data.

```@example mono
using Dates, DataFrames, Random
using TSML, TSML.Utils, TSML.TSMLTypes
using TSML: Plotter

Random.seed!(123)
pltr = Plotter(Dict(:interactive => false))
mdates = DateTime(2017,12,1,1):Dates.Hour(1):DateTime(2017,12,31,10) |> collect
mvals = rand(length(mdates)) |> cumsum
df =  DataFrame(Date=mdates ,Value = mvals)
fit!(pltr,df)
transform!(pltr,df)
```

Now that we have a monotonic data, let's use the `Monotonicer` to normalize and plot the result:

```@example mono
using TSML, TSML.Utils, TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML: Monotonicer

mono = Monotonicer(Dict())

pipeline = Pipeline(Dict(
   :transformers => [mono,pltr]
   )
)

fit!(pipeline,df)
res=transform!(pipeline,df)

```

## Real Data Example

We will now apply the entire pipeline 
starting from reading csv data, aggregate, impute, normalize
if it's monotonic, and plot. We will consider three 
different data types: a regular time series data, a  
monotonic data, and a daily monotonic data. The difference between  
monotonic and daily monotonic is that the values in daily monotonic resets to 
zero or some baseline and cumulatively increases in a day until the 
next day where it resets to zero or some baseline value. `Monotonicer`
automatically detects these three different types and apply the corresponding
normalization accordingly.

```@example mono
using TSML: DateValgator, DateValNNer, Statifier, Monotonicer
regularfile = joinpath(dirname(pathof(TSML)),"../data/typedetection/regular.csv")
monofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/monotonic.csv")
dailymonofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/dailymonotonic.csv")

regularfilecsv = CSVDateValReader(Dict(:filename=>regularfile,:dateformat=>"dd/mm/yyyy HH:MM"))
monofilecsv = CSVDateValReader(Dict(:filename=>monofile,:dateformat=>"dd/mm/yyyy HH:MM"))
dailymonofilecsv = CSVDateValReader(Dict(:filename=>dailymonofile,:dateformat=>"dd/mm/yyyy HH:MM"))

valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
stfier = Statifier(Dict(:processmissing=>true))
mono = Monotonicer(Dict())
pltr = Plotter(Dict(:interactive => false))
nothing #hide
```

## Regular TS Processing
Let's test by feeding the regular time series type to the pipeline. We expect that for this type,
`Monotonicer` will not perform further processing:


- Pipeline with `Monotonicer`: regular time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [regularfilecsv,valgator,valnner,mono,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

- Pipeline without `Monotonicer`: regular time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [regularfilecsv,valgator,valnner,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

Notice that the plots are the same with or without the `Monotonicer` instance.

## Monotonic TS Processing
Let's now feed the same pipeline with a monotonic csv data.

- Pipeline without `Monotonicer`: monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [monofilecsv,valgator,valnner,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

- Pipeline with `Monotonicer`: monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [monofilecsv,valgator,valnner,mono,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

Notice that without the `Monotonicer` instance, the data is monotonic. Applying
the `Monotonicer` instance in the pipeline converts the data into
a regular time series but with outliers.

We can use the `Outliernicer` filter to remove outliers. Let's apply this filter after the
`Monotonicer` and plot the result.

- Pipeline with `Monotonicer` and `Outliernicer`: monotonic time series
```@example mono
using TSML: Outliernicer
outliernicer = Outliernicer(Dict(:dateinterval=>Dates.Hour(1)));

pipeline = Pipeline(Dict(
    :transformers => [monofilecsv,valgator,valnner,mono, outliernicer,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

## Daily Monotonic TS Processing
Lastly, let's feed the daily monotonic data using similar pipeline and examine its plot.

- Pipeline without `Monotonicer`: daily monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [dailymonofilecsv,valgator,valnner,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

This plot is characterized by monotonically increasing trend but resets to certain baseline value 
at the end of the day and repeat similar trend daily. The challenge for the monotonic normalizer
is to differentiate between daily monotonic from the typical monotonic function to apply
the correct normalization.

- Pipeline with `Monotonicer`: daily monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [dailymonofilecsv,valgator,valnner,mono,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

While the `Monotonicer` filter is able to transform the data into a regular time series,
there are significant outliers due to noise and the nature of this kind of data or sensor.

Let's remove the outliers by applying the `Outliernicer` filter and examine the result.

- Pipeline with `Monotonicer` and `Outliernicer`: daily monotonic time series
```@example mono
pipeline = Pipeline(Dict(
    :transformers => [dailymonofilecsv,valgator,valnner,mono,outliernicer,pltr]
   )
)
fit!(pipeline)
transform!(pipeline)
```

The `Outliernicer` filter effectively removed the outliers as shown in the plot.
