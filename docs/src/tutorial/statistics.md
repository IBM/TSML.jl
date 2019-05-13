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
:processmissing => false to the argument during its instance creation. Below
illustrates this workflow.


