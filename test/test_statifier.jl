module TestStatifier

using Test
using TSML

function test_statifier()

  Random.seed!(123)
  dt=[missing;rand(1:10,3);missing;missing;missing;rand(1:5,3)]
  dat = DataFrame(Date= DateTime(2017,12,31,1):Dates.Hour(1):DateTime(2017,12,31,10) |> collect,
		  Value = dt)
  statfier = Statifier(Dict(:processmissing=>false))
  fit!(statfier,dat)
  res=transform!(statfier,dat)
  @test ncol(res) == 20
  @test res[1,3:end]|>Vector|>sum |> x->round(x,digits=5) > 0.0
  m=fit(statfier,dat)
  res=transform(m,dat)
  @test ncol(res) == 20
  @test res[1,3:end]|>Vector|>sum |> x->round(x,digits=5) > 0.0

  Random.seed!(123)
  statfier = Statifier(Dict(:processmissing=>true))
  fit!(statfier,dat)
  res=transform!(statfier,dat)
  @test ncol(res) == 26
  @test res[1,3:end]|>Vector|>sum |> x->round(x,digits=5) > 0.0
  m=fit(statfier,dat)
  res=transform!(m,dat)
  @test ncol(res) == 26
  @test res[1,3:end]|>Vector|>sum |> x->round(x,digits=5) > 0.0

  fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1),:strict=>false))

  stfier = Statifier(Dict(:processmissing=>true))
  mpipeline0 = csvfilter |> stfier
  respipe0 = fit_transform!(mpipeline0)
  @test round(respipe0[1,:sfreq],digits=2) ==  0.18
  vals = respipe0[1,3:end] |> Vector
  @test (vals[(!isnan).(vals)] |> sum |> x->round(x,sigdigits=2)) < 0.0
  respipe0 = fit_transform(mpipeline0)
  @test round(respipe0[1,:sfreq],digits=2) ==  0.18
  vals = respipe0[1,3:end] |> Vector
  @test (vals[(!isnan).(vals)] |> sum |> x->round(x,sigdigits=2)) < 0.0

  stfier = Statifier(Dict(:processmissing=>true))
  mpipeline1 = csvfilter |> valgator |> stfier
  fit!(mpipeline1)
  respipe1 = transform!(mpipeline1)
  @test respipe1[1,3:end] |> Vector |> sum |> x->round(x,digits=5) < 0.0
  m=fit(mpipeline1)
  respipe1 = transform(m)
  @test respipe1[1,3:end] |> Vector |> sum |> x->round(x,digits=5) < 0.0

  mpipeline2 = csvfilter |> valgator |> valnner |> stfier
  fit!(mpipeline2)
  respipe2 = transform!(mpipeline2) 
  res2 = respipe2[1,3:end] |> Vector
  @test (res2[(!isnan).(res2)] |> sum |> x->round(x,sigdigits=2)) < 0.0
  m=fit(mpipeline2)
  respipe2 = transform(m) 
  res2 = respipe2[1,3:end] |> Vector
  @test (res2[(!isnan).(res2)] |> sum |> x->round(x,sigdigits=2)) < 0.0

  stfier = Statifier(Dict(:processmissing=>false))
  mpipeline1 = csvfilter |> valgator |> stfier
  respipe1 = fit_transform!(mpipeline1)
  res1 = respipe1[1,3:end] |> Vector
  @test res1 |> sum |> x->round(x,digits=5) < 0.0
  respipe1 = fit_transform(mpipeline1)
  res1 = respipe1[1,3:end] |> Vector
  @test res1 |> sum |> x->round(x,digits=5) < 0.0

  stfier = Statifier(Dict(:processmissing=>true))
  mpipeline2 = csvfilter |> valgator |> valnner |> stfier
  respipe2 = fit_transform!(mpipeline2) 
  res2 = respipe2[1,3:end] |> Vector
  @test res2[(!isnan).(res2)] |> sum |> x->round(x,digits=5) < 0.0
  respipe2 = fit_transform(mpipeline2) 
  res2 = respipe2[1,3:end] |> Vector
  @test res2[(!isnan).(res2)] |> sum |> x->round(x,digits=5) < 0.0

end
@testset "Statifier: readcsv |> valgator |> valnner |> stfier" begin
  test_statifier()
end

end
