```@meta
Author = "Paulito P. Palmes"
```

# Statistical Metrics

Let us again start generating an artificial data with missing values which we 
will use in our demo tutorial.

```@example stat
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

## Statifier

TSML includes `Statifier` transformer that computes scalar statistics to
characterize the time-series data. By default, it also computes statistics of 
missing blocks of data. To disable this feature, one can pass 
`:processmissing => false` to the argument during its instance creation. Below
illustrates this workflow.

```@example stat
using Dates
using TSML
using TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML: Pipeline
using TSML: DateValgator
using TSML: DateValNNer
using TSML: Statifier

dtvalgator = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
dtvalnner = DateValNNer(Dict(:dateinterval => Dates.Hour(1)))
dtvalizer = DateValizer(Dict(:dateinterval => Dates.Hour(1)))
stfier = Statifier()

mypipeline = Pipeline(
  Dict( :transformers => [
            dtvalgator,
            stfier
         ]
  )
)

fit!(mypipeline,X)
results = transform!(mypipeline,X)
```

If you are not intested with the statistics of the missing blocks, you can indicate
`:processmissing => false` in the instance argument.

```@example stat
stfier = Statifier(Dict(:processmissing=>false))
mypipeline = Pipeline(
  Dict( :transformers => [
            dtvalgator,
            stfier
         ]
  )
)
fit!(mypipeline,X)
results = transform!(mypipeline,X)
```

Let us check the statistics after the imputation. We expect that if the imputation is successful,
the stats for missing blocks will all be NaN because stats of empty set is an NaN.

```@example stat
stfier = Statifier(Dict(:processmissing=>true))
mypipeline = Pipeline(
  Dict( :transformers => [
            dtvalgator,
            dtvalnner,
            stfier
         ]
  )
)
fit!(mypipeline,X)
results = transform!(mypipeline,X)
```

As we expected, the imputation is successful and there are no more missing values in the
processed time-series dataset.

Let's try with the other imputation using `DateValizer` and validate that there are no more
missing values based on the stats.

```@example stat
stfier = Statifier(Dict(:processmissing=>true))
mypipeline = Pipeline(
  Dict( :transformers => [
            dtvalgator,
            dtvalizer,
            stfier
         ]
  )
)
fit!(mypipeline,X)
results = transform!(mypipeline,X)
```

Indeed, the imputation is a success.
