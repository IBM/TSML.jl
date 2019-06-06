module TSMLDemo

using Dates
using DataFrames
using Random
using Statistics
using Plots

using TSML: Statifier
using TSML.TSMLTransformers: Pipeline, DateValgator, DateValNNer
using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils
using TSML: Plotter

export tsml_demo

function pauseme(msg)
  println(msg)
  println(">>> enter key to continue <<<")
  read(stdin,Char);
  nothing
end

function generateX()
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2014,1,3)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 0.30 * length(gdate) |> floor |> Integer
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X[:Value][gndxmissing] .= missing
    return X
end


function tsml_demo()
  println("\n")
  println(">>>>>>>>>><<<<<<<<<<")
  println("Welcome to TSML demo")
  println(">>>>>>>>>><<<<<<<<<<")
  println("\n")
  println("Let's start by generating random Time Series")
  pauseme("data and output the first 20 rows")
  data=generateX()
  println(first(data,20))
  println("...")
  println("...")
  pauseme("Notice the presence of missing values.")

  println("Let's create a pipeline for plotting")
  println(" ")
  pauseme("
  julia code:
      pltr = Plotter()
      mpipeline = Pipeline(Dict(
            :transformers => [pltr]
           )
      )
      fit!(mpipeline,data)
      transform!(mpipeline,data)
  ")
  println("Please wait until the plot is shown before continuing.")

  pltr = Plotter()
  xpipeline = Pipeline(Dict(
          :transformers => [pltr]
       )
  )
  fit!(xpipeline,data)
  res=transform!(xpipeline,data) 
  display(res)
  println("Notice the breaks in the plot due to missing data.\n")
  println("Let's create a pipeline to aggregate and")
  println("measure data quality including missing values")
  println(" ")
  pauseme("
  julia code:
      valgator = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
      stfier = Statifier(Dict(:processmissing => true))
      mpipeline = Pipeline(Dict(
            :transformers => [valgator,stfier]
           )
      )
      fit!(mpipeline,data)
      res = transform!(mpipeline,data)
  ")
 
  valgator = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
  stfier = Statifier(Dict(:processmissing => true))
  mpipeline = Pipeline(Dict(
        :transformers => [valgator,stfier]
       )
  )
  fit!(mpipeline,data)
  res = transform!(mpipeline,data)
  show(res,allcols=true)
  println("\n")
  println("Notice the stats of blocks of missing values denoted by fieldnames starting with b.")
  println(" ")
  println("Let's create a pipeline for aggregation with imputation and get data quality")
  pauseme("
  julia code:
      valgator = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
      valnner = DateValNNer(Dict(:dateinterval => Dates.Hour(1)))
      stfier = Statifier(Dict(:processmissing => true))
      mpipeline = Pipeline(Dict(
            :transformers => [valgator,valnner,stfier]
           )
      )
      fit!(mpipeline,data)
      res = transform!(mpipeline,data)
  ")
  valgator = DateValgator(Dict(:dateinterval => Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval => Dates.Hour(1)))
  stfier = Statifier(Dict(:processmissing => true))
  mpipeline = Pipeline(Dict(
          :transformers => [valgator,valnner,stfier]
       )
  )
  fit!(mpipeline,data)
  res = transform!(mpipeline,data)
  show(res,allcols=true)
  println("\n")
  println("Notice that the imputation causes the statistics of missing values return NaN.")
  println("")
  pauseme("Let's create a pipeline to plot the cleaned and imputed data")
  pauseme("
  julia code:
      mpipeline = Pipeline(Dict(
              :transformers => [valgator,valnner,pltr]
           )
      )
      fit!(mpipeline,data)
      res=transform!(mpipeline,data)
  ")
  mpipeline = Pipeline(Dict(
          :transformers => [valgator,valnner,pltr]
       )
  )
  fit!(mpipeline,data)
  res=transform!(mpipeline,data)
  display(res)
  nothing
end


end
