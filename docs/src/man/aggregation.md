```@meta
Author = "Paulito P. Palmes"
```

# Aggregation 
`DateValgator` is the type that performs aggregation to minimize
noise and lessen the occurrence of missing data. It expects one
argument which is the date-time interval to group values by taking 
their median. For example, grouping hourly can be carried out
by passing this argument: `:dateinterval => Dates.Hour(1)`

Let's start by generating an artificial data with sample
frequencey every 5 minutes and print the first 10 rows.

```@example datevalgator
using Dates, DataFrames
gdate = DateTime(2014,1,1):Dates.Minute(5):DateTime(2014,5,1)
gval = rand(length(gdate))

df = DataFrame(Date=gdate,Value=gval)
first(df,10)
```
Let's apply the aggregator and try hourly, half hourly,
and daily aggregates of the data.

```@example datevalgator
using TSML, TSML.TSMLTransformers, TSML.Utils, TSML.TSMLTypes

hourlyagg = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
halfhourlyagg = DateValgator(Dict(:dateinterval => Dates.Minute(30)))
dailyagg = DateValgator(Dict(:dateinterval => Dates.Day(1)))
nothing #hide
```

Here's the first 5 rows of hourly aggregate:
```@example datevalgator
fit!(hourlyagg,df)
hourlyres = transform!(hourlyagg,df)
first(hourlyres, 5)
```

The first 5 rows of half hourly aggregate:
```@example datevalgator
fit!(halfhourlyagg,df)
halfhourlyres = transform!(halfhourlyagg,df)
first(halfhourlyres, 5)
```

The first 5 rows of daily aggregate:
```@example datevalgator
fit!(dailyagg,df)
dailyres = transform!(dailyagg,df)
first(dailyres, 5)
```
