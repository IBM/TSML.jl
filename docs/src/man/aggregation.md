```@meta
Author = "Paulito P. Palmes"
```

# Aggregation 
`DateValgator` is a data type that supports operation 
for aggregation to minimize noise and 
lessen the occurrence of missing data. It expects to receive one
argument which is the date-time interval for grouping values by taking 
their median. For example, hourly median as the basis of aggregation
can be carried out by passing this argument: `:dateinterval => Dates.Hour(1)`

To illustrate `DateValgator` usage, let's start by 
generating an artificial data with sample
frequencey every 5 minutes and print the first 10 rows.

```@example datevalgator
using Dates, DataFrames
gdate = DateTime(2014,1,1):Dates.Minute(5):DateTime(2014,5,1)
gval = rand(length(gdate))

df = DataFrame(Date=gdate,Value=gval)
nothing #hide
```

```@repl datevalgator
first(df,10)
```


## DateValgator

Let's apply the aggregator and try diffent groupings: hourly vs half hourly
vs daily aggregates of the data.

```@example datevalgator
using TSML, TSML.TSMLTransformers, TSML.Utils, TSML.TSMLTypes

hourlyagg = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
halfhourlyagg = DateValgator(Dict(:dateinterval => Dates.Minute(30)))
dailyagg = DateValgator(Dict(:dateinterval => Dates.Day(1)))

fit!(halfhourlyagg,df)
halfhourlyres = transform!(halfhourlyagg,df)

fit!(hourlyagg,df)
hourlyres = transform!(hourlyagg,df)

fit!(dailyagg,df)
dailyres = transform!(dailyagg,df)
nothing #hide
```

The first 5 rows of half-hourly, hourly, and daily aggregates:
```@repl datevalgator
first(halfhourlyres,5)
first(hourlyres,5)
first(dailyres,5)
```
