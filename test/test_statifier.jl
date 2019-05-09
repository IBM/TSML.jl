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
  @test ncol(res) == 20
  @test res[1,3:end]|>Vector|>sum |> x->round(x,digits=5) == 16.29114

  statfier = Statifier(Dict(:processmissing=>true))
  fit!(statfier,dat)
  res=transform!(statfier,dat)
  @test ncol(res) == 26
  @test res[1,3:end]|>Vector|>sum |> x->round(x,digits=5) == 28.29114

  fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  csvfilter = DataReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))

  stfier = Statifier(Dict(:processmissing=>true))
  mpipeline0 = Pipeline(Dict(
	  :transformers => [csvfilter,stfier]
     )
  )
  fit!(mpipeline0)
  respipe0 = transform!(mpipeline0)
  @test round(respipe0[1,:SFreq],digits=2) ==  0.18
  #@test respipe1[1,3:end] |> Vector |> sum |> x->round(x,digits=5) == -106687.08093
  vals = respipe0[1,3:end]
  @test (vals[(!isnan).(vals)] |> sum |> x->round(x,sigdigits=2)) == -1.3e6

  stfier = Statifier(Dict(:processmissing=>true))
  mpipeline1 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,stfier]
      #:transformers => [csvfilter,stfier]
     )
  )
  fit!(mpipeline1)
  respipe1 = transform!(mpipeline1)
  @test respipe1[1,3:end] |> Vector |> sum |> x->round(x,digits=5) == -102779.88226

  mpipeline2 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,valnner,stfier]
     )
  )
  fit!(mpipeline2)
  respipe2 = transform!(mpipeline2) 
  res2 = respipe2[1,3:end] |> Vector
  @test (res2[(!isnan).(res2)] |> sum |> x->round(x,sigdigits=2)) == -230000.00

  stfier = Statifier(Dict(:processmissing=>false))
  mpipeline1 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,stfier]
     )
  )
  fit!(mpipeline1)
  respipe1 = transform!(mpipeline1)
  res1 = respipe1[1,3:end] |> Vector
  @test res1 |> sum |> x->round(x,digits=5) == -105185.44114

  stfier = Statifier(Dict(:processmissing=>true))
  mpipeline2 = Pipeline(Dict(
      :transformers => [csvfilter,valgator,valnner,stfier]
     )
  )
  fit!(mpipeline2)
  respipe2 = transform!(mpipeline2) 
  res2 = respipe2[1,3:end] |> Vector
  @test res2[(!isnan).(res2)] |> sum |> x->round(x,digits=5) == -227824.85354

end
@testset "Statifier: readcsv |> valgator |> valnner |> stfier" begin
  test_statifier()
end

end
