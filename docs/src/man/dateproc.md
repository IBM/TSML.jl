```@meta
Author = "Paulito P. Palmes"
```

# Date Preprocessing
Extracting the Date features in a Date, Value table follows
similar workflow with the Value preprocessing in the previous
section. The main difference is we are only interested on the
date corresponding to the last column of the values generated
by the `Matrifier`. This last column contains the values before 
the prediction happens and the dates corresponding to these
values carry significant information based on recency compared
to the other dates.

Let us start by creating a Date,Value dataframe similar to the previous section.

```@example dateifier
using Dates
using TSML, TSML.Utils, TSML.TSMLTypes
using TSML.TSMLTransformers
using DataFrames

lower = DateTime(2017,1,1)
upper = DateTime(2018,1,31)
dat=lower:Dates.Day(1):upper |> collect
vals = rand(length(dat))
x = DataFrame(Date=dat,Value=vals)
first(x,5)
```

Let us create an instance of `Dateifier` passing the size of row,
stride, and steps ahead to predict:

```@example dateifier
mtr = Dateifier(Dict(:ahead=>24,:size=>24,:stride=>5))
fit!(mtr,x)
res = transform!(mtr,x)
first(res,5)
```

The output extract automatically several date features
such as year, month, day, hour, week, day of the week, 
day of quarter, quarter of year.
