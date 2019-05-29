```@meta
Author = "Paulito P. Palmes"
```

# Date Preprocessing
Extracting the Date features in a `Date,Value` table follows
similar workflow with the [value preprocessing](@ref valueprep) 
of the previous section. The main difference 
is we are only interested on the
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
nothing #hide
```

```@repl dateifier
first(x,5)
```

## Dateifier
Let us create an instance of `Dateifier` passing the size of row,
stride, and steps ahead to predict:

```@example dateifier
mtr = Dateifier(Dict(:ahead=>24,:size=>24,:stride=>5))
fit!(mtr,x)
res = transform!(mtr,x)
nothing #hide
```

```@repl dateifier
first(res,5)
```

The model `transform!` output extracts automatically several date features
such as year, month, day, hour, week, day of the week, 
day of quarter, quarter of year.

## ML Features: Matrifier and Datefier 

You can then combine the outputs in both the `Matrifier` and `Datefier` 
as input features to a machine learning model. Below is an example of the
workflow where the code extracts the Date and Value features combining them
to form a matrix of features as input to a machine learning model.

```@example dateifier
commonargs = Dict(:ahead=>3,:size=>5,:stride=>2)
dtr = Dateifier(commonargs)
mtr = Matrifier(commonargs)

lower = DateTime(2017,1,1)
upper = DateTime(2018,1,31)
dat=lower:Dates.Day(1):upper |> collect
vals = rand(length(dat))
X = DataFrame(Date=dat,Value=vals)

fit!(mtr,X)
valuematrix = transform!(mtr,X)
fit!(dtr,X)
datematrix = transform!(dtr,X)
mlfeatures = hcat(datematrix,valuematrix)
nothing #hide
```

```@repl dateifier
first(mlfeatures,5)
```
