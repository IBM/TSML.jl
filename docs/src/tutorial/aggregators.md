```@meta
Author = "Paulito P. Palmes"
```

# [Aggregators and Imputers](@id aggregators_imputers)

The package assumes a two-column table composed of `Dates` and `Values`. 
The first part of the workflow aggregates values based on the specified 
date-time interval which minimizes occurence of missing values and noise. 
The aggregated data is then left-joined to the complete sequence of  `DateTime` 
in a specified date-time interval. Remaining missing values are replaced 
by `k` nearest neighbors where `k` is the symmetric distance from the location 
of missing value. This replacement algo is called several times until there 
are no more missing values.

Let us create a Date, Value table with some missing values and output the first
15 rows. We will then apply some TSML functions to normalize/clean the data.
Below is the code of the `generateDataWithMissing()` function:

```@example 1
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
nothing #hide
```
```@repl 1
X = generateDataWithMissing();
first(X,15)
```
## DateValgator
You'll notice several blocks of missing in the table above with reading frequency of every 15 minutes. 
To minimize noise and lessen the occurrence of missing values,
let's aggregate our dataset by taking the hourly median using the `DateValgator` transformer.

```@example 1
using TSML
using TSML.TSMLTypes
using TSML.Utils
using TSML.TSMLTransformers
using TSML: DateValgator

dtvlgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
fit!(dtvlgator,X)
results = transform!(dtvlgator,X)
nothing #hide
```

```@repl 1
first(results,10)
```

The occurrence of missing values is now reduced because of the hourly aggregation. While
the default is hourly aggregation, you can easily change it by using a different interval
in the argument during instance creation. Below indicates every 30 minutes interval.

```@repl 1
dtvlgator = DateValgator(Dict(:dateinterval=>Dates.Minute(30)))
```

`DateValgator` is one of the several TSML transformers to preprocess and clean the 
time series data. In order to create additional transformers to extend TSML, 
each transformer must overload the two `Transformer` functions:`fit!` and `transform!`. 
`DateValgator` `fit!` performs initial setups of necessary parameters
and validation of arguments while its `transform!` function contains the algorithm 
for aggregation. 

For machine learning prediction and classification transformer, 
`fit!` function is equivalent to ML training or parameter optimization, 
while the `transform!` function is for doing the actual prediction.
The later part of the tutorial will provide an example how to add a `Transformer` to
extend the functionality of TSML.

## DateValNNer

Let's perform further processing to replace the remaining missing values with their nearest neighbors. 
We will use `DateValNNer` which is a TSML transformer to process the output of `DateValgator`.
`DateValNNer` can also process non-aggregated data by first running similar workflow
of `DateValgator` before performing its imputation routine.

```@example 1
using TSML: DateValNNer

datevalnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
fit!(datevalnner, X)
results = transform!(datevalnner,X)
nothing #hide
```

```@repl 1
first(results,10)
```

After running the `DateValNNer`, it's guaranteed that there will be no more
missing data unless the input are all missing data.

## DateValizer

One more imputer to replace missing data is `DateValizer`. It computes the hourly
median over 24 hours and use the `hour => median` hashmap learned
to replace missing data using `hour` as the key. In this implementation, `fit!`
function is doing the training of parameters by computing the medians and save it
for the `transform!` function to use for imputation. It is possible that the
hashmap can contain missing values in cases where the pooled hourly median in
a particular hour have all missing data.
Below is a sample workflow to replace missing data in X with the hourly medians.

```@example 1
using TSML: DateValizer

datevalizer = DateValizer(Dict(:dateinterval=>Dates.Hour(1)))
fit!(datevalizer, X)
results = transform!(datevalizer,X)
nothing #hide
```

```@repl 1
first(results,10)
```
