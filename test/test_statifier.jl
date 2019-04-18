module TestStatifier

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers

using TSML.Statifiers
using TSML.DataReaders


using DataFrames
using Dates
using Random
using Test

function test_statifier()
  Random.seed!(123)
  dt=[missing;rand(1:10,3);missing;missing;missing;rand(1:5,3)]
  dat = DataFrame(Date= DateTime(2017,12,31,1):Dates.Hour(1):DateTime(2017,12,31,10) |> collect,
		  Value = dt)
  statfier = Statifier(Dict(:processmissing=>false))
  fit!(statfier,dat)
  res=transform!(statfier,dat)
  @test ncol(res) == 10
  @test res|>Matrix|>sum |> x->round(x,digits=5) == -8.59573

  statfier = Statifier(Dict(:processmissing=>true))
  fit!(statfier,dat)
  res=transform!(statfier,dat)
  @test res|>Matrix|>sum |> x->round(x,digits=5) == 3.40427

  fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  csvfilter = DataReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))

  stfier = Statifier(Dict(:processmissing=>true))
  mpipeline1 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,stfier]
     )
  )
  fit!(mpipeline1)
  respipe1 = transform!(mpipeline1)
  @test respipe1 |> Matrix |> sum |> x->round(x,digits=5) == -106687.08093

  mpipeline2 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,valnner,stfier]
     )
  )
  fit!(mpipeline2)
  respipe2 = transform!(mpipeline2) |>Matrix
  @test respipe2[(!isnan).(respipe2)] |> sum |> x->round(x,digits=5) == -236661.05262

  stfier = Statifier(Dict(:processmissing=>false))
  mpipeline1 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,stfier]
     )
  )
  fit!(mpipeline1)
  respipe1 = transform!(mpipeline1)
  @test respipe1 |> Matrix |> sum |> x->round(x,digits=5) == -109092.63981

  mpipeline2 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,valnner,stfier]
     )
  )
  fit!(mpipeline2)
  respipe2 = transform!(mpipeline2) |>Matrix
  @test respipe2[(!isnan).(respipe2)] |> sum |> x->round(x,digits=5) == -236661.05262
end
@testset "Statifier: readcsv |> valgator |> valnner |> stfier" begin
  test_statifier()
end

end
