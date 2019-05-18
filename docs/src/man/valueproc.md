```@meta
Author = "Paulito P. Palmes"
```

# Value Preprocessing
In order to process 1-D TS as input for ML model, it has to
be converted into Matrix form where each row represents a 
slice of 1-D TS representing daily/hourly/weekly pattern
depending on the size of the chunk, stride, and 
number of steps ahead for prediction. Below illustrates
the processing workflow to `Matrify` a 1-D TS.

For illustration purposes, the code below generates a 
Date,Value dataframe where the values are just a sequece
of integer from 1 to the length of the date sequence.
We use this simple sequence to have a better understanding how the
slicing of rows, steps ahead, and the stride to create the `Matrified` output
is generated.


```@example matrify
using Dates
using TSML, TSML.Utils, TSML.TSMLTypes
using TSML.TSMLTransformers
using DataFrames

lower = DateTime(2017,1,1)
upper = DateTime(2017,1,5)
dat=lower:Dates.Hour(1):upper |> collect
vals = 1:length(dat)
x = DataFrame(Date=dat,Value=vals)
first(x,5)
```

Let us create an instance of Matrifier passing the size of row,
stride, and steps ahead to predict:

```@example matrify
mtr = Matrifier(Dict(:ahead=>24,:size=>24,:stride=>5))
fit!(mtr,x)
res = transform!(mtr,x)
```

In this example, we have hourly values. We indicated in the 
`Matrifier` to generate a matrix where the size of each row
is 24 hours, steps ahead for prediction is 24 hours and the
stride of 5 hours.

Let us try to make a matrix with the size of 12 hours, ahead of
1 hour, and stride of 6 hours:

```@example matrify
mtr = Matrifier(Dict(:ahead=>1,:size=>12,:stride=>6))
fit!(mtr,x)
res = transform!(mtr,x)
```


