```@meta
Author = "Paulito P. Palmes"
```

# Imputation
There are two ways to impute the `date,value` TS data. One uses `DateValNNer` which uses
nearest neighbor and `DateValizer` which uses the dictionary of medians mapped to 
certain date-time interval grouping.

## DateValNNer
`DateValNNer` expects the following arguments with their default values during instantation: 
- `:dateinterval => Dates.Hour(1)`  
    - grouping interval
- `:nnsize => 1` 
    - size of neighborhood
- `:missdirection => :symmetric` 
    -  `:forward` vs `:backward` vs `:symmetric`
- `:strict => true` 
    - whether or not to repeatedly iterate until no more missing data

The `:missdirection` indicates the imputation direction and the extent of neighborhood. Symmetric
implies getting info from both sides of the missing data. `:forward` direction starts
imputing from the top while the `:reverse` starts from the bottom. Please refer to 
[Aggregators and Imputers](@ref aggregators_imputers) for other examples.

```@setup impute
using Random
using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers

using Dates, DataFrames

function generateXY()
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X[:Value][gndxmissing] .= missing
    Y = rand(length(gdate))
    (X,Y)
end
X,Y = generateXY()
```
Let's use the same dataset we have used in the tutorial and print the first few rows.

```@repl impute
first(X,10)
```

Let's try the following setup grouping daily with `forward` imputation and 10 neighbors:
```@example impute
dnnr = DateValNNer(Dict(:dateinterval=>Dates.Hour(2),
             :nnsize=>10,:missdirection => :forward,
             :strict=>false))
fit!(dnnr,X)
forwardres=transform!(dnnr,X)
nothing #hide
```

```@repl impute
first(forwardres,5)
```

Same parameters as above but uses `reverse` instead of `forward` direction:
```@example impute
dnnr = DateValNNer(Dict(:dateinterval=>Dates.Hour(2),
             :nnsize=>10,:missdirection => :reverse,
             :strict=>false))
fit!(dnnr,X)
reverseres=transform!(dnnr,X)
nothing #hide
```

```@repl impute
first(reverseres,5)
```

Using `symmetric` imputation:

```@example impute
dnnr = DateValNNer(Dict(:dateinterval=>Dates.Hour(2),
             :nnsize=>10,:missdirection => :symmetric,
             :strict=>false))
fit!(dnnr,X)
symmetricres=transform!(dnnr,X)
nothing #hide
```

```@repl impute
first(symmetricres,5)
```

Unlike `symmetric` imputation that guarantees 100% imputation of missing
data as long as the input has non-missing elements, `forward` and `reverse`
cannot guarantee that the imputation replaces all missing data because
of the boundary issues. If the top or bottom of the input is missing,
the assymetric imputation will not be able to replace the endpoints that
are missing. It is advised that to have successful imputation, `symmetric`
imputation shall be used.

In the example above, the number of remaining missing data not imputed for
`forward`, `reverse`, and `symmetric` is:

```@repl impute
sum(ismissing.(forwardres[:Value]))
sum(ismissing.(reverseres[:Value]))
sum(ismissing.(symmetricres[:Value]))
```

## DateValizer
`DateValizer` operates on the principle that there is a reqularity of patterns
in a specific time period such that replacing values is just a matter of 
extracting which time period it belongs and used the pooled median in that time
period to replace the missing data. The default time period for `DateValizer`
is hourly. In a more advanced implementation, we can add daily, hourly, and weekly 
periods but it will require much larger hash table. Additional grouping criteria 
can result into smaller subgroups which may contain 100% missing in some
of these subgroups resulting to imputation failure. `DateValizer` only depends
on the `:dateinterval => Dates.Hour(1)`  argument with default value of hourly.
Please refer to [Aggregators and Imputers](@ref aggregators_imputers) for more examples.

Let's try hourly, daily, and monthly median as the basis of imputation:

```@repl impute
hourlyzer = DateValizer(Dict(:dateinterval => Dates.Hour(1)))
monthlyzer = DateValizer(Dict(:dateinterval => Dates.Month(1)))
dailyzer = DateValizer(Dict(:dateinterval => Dates.Day(1)))

fit!(hourlyzer,X)
hourlyres = transform!(hourlyzer,X)

fit!(dailyzer,X)
dailyres = transform!(dailyzer,X)

fit!(monthlyzer,X)
monthlyres = transform!(monthlyzer,X)
```

