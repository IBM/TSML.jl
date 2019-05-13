```@meta
Author = "Paulito P. Palmes"
```

# Aggregators and Imputers

The package assumes a two-column input composed of Dates and Values. 
The first part of the workflow aggregates values based on the specified 
date/time interval which minimizes occurence of missing values and noise. 
The aggregated data is then left-joined to the complete sequence of dates 
in a specified date/time interval. Remaining missing values are replaced 
by k nearest neighbors where k is the symmetric distance from the location 
of missing value. This approach can be called several times until there 
are no more missing values.

Let us create Date, Value input with some missing values and apply TSML functions
to normalize/clean the data:

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
```

Let's output the first 20 rows:

```@example 1
X = generateDataWithMissing()
first(X,20)
```
## DateValgator
You'll notice several blocks of missing with reading frequency every 15 minutes. 
Let's aggregate our dataset by taking the hourly median using the `DateValgator` transformer.

```@example 1
using TSML
using TSML.TSMLTypes
using TSML.Utils
using TSML.TSMLTransformers
using TSML: DateValgator

dtvlgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
fit!(dtvlgator,X)
results = transform!(dtvlgator,X)
first(results,20)
```

Missing values are now reduced because of the aggregation applied using
`DateValgator` transformer. TSML transformers support the two main functions:
`fit!` and `transform!`. `DateValgator fit!` performs initial setups of necessary parameters
and validation of arguments while its `transform!` contains the algorithm for aggregation.

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
first(results,20)
```

After running the `DateValNNer`, it's guaranteed that there will be no more
missing data. 

## DateValizer

One more imputer to replace missing data is `DateValizer`. It computes the hourly
median over 24 hours and use the hour => median mapping 
to replace missing data with the hour as the key. Below is a sample
workflow to replace missing data in X with the hourly medians.

```@example 1
using TSML: DateValizer

datevalizer = DateValizer(Dict(:dateinterval=>Dates.Hour(1)))
fit!(datevalizer, X)
results = transform!(datevalizer,X)
first(results,20)
```


