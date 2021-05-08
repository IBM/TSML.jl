module TestMonotonicer

using Test
using TSML

function test_basicmonotonicer()
  Random.seed!(123)
  dat = rand(1:10,5)
  # check monotonic increasing
  @test ismonotonic(dat) == false 
  @test ismonotonic(cumsum(dat)) == true

  mdates = DateTime(2017,12,31,1):Dates.Hour(1):DateTime(2017,12,31,10) |> collect
  mvals = rand(length(mdates)) |> cumsum
  df =  DataFrame(Date=mdates ,Value = mvals)

  mono = Monotonicer(Dict())
  fit!(mono,df)
  res=transform!(mono,df)
  @test ismonotonic(res.Value) == false 
  m=fit(mono,df)
  res=transform(m,df)
  @test ismonotonic(res.Value) == false 

  fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1),:strict=>false))
  stfier = Statifier(Dict(:processmissing=>true))
  mono = Monotonicer(Dict())

  mpipeline1 = csvfilter |> valgator |> mono |> stfier
  respipe1 = fit_transform!(mpipeline1)

  mpipeline2 = csvfilter |> valgator |> stfier
  respipe2 = fit_transform!(mpipeline2)
  @test (respipe1[1:1,:] .== respipe2[1:1,:]) |> Matrix |> sum  == ncol(respipe1)

  respipe1 = fit_transform(mpipeline1)
  respipe2 = fit_transform(mpipeline2)
  @test (respipe1[1:1,:] .== respipe2[1:1,:]) |> Matrix |> sum  == ncol(respipe1)

  mpipeline3 = csvfilter |> valgator |> valnner |> mono |> stfier
  respipe3 = fit_transform!(mpipeline3)
  mpipeline5 = csvfilter |> valgator |> valnner |> stfier
  fit!(mpipeline5)
  respipe5 = transform!(mpipeline5)
  val1 = respipe3[1,3:end] |> Vector
  val2 = respipe5[1,3:end] |> Vector
  ok1 = (!isnan).(val1)
  ok2 = (!isnan).(val2)
  @test sum( val1[ok1].== val2[ok2]) == length(val1[ok1])

end
@testset "Monotonicer: readcsv |> valgator |> valnner |> mono |> stfier" begin
  test_basicmonotonicer()
end

function test_typesmonotonicer()
  regularfile = joinpath(dirname(pathof(TSML)),"../data/typedetection/regular.csv")
  monofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/monotonic.csv")
  dailymonofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/dailymonotonic.csv")
  regularfilecsv = CSVDateValReader(Dict(:filename=>regularfile,:dateformat=>"dd/mm/yyyy HH:MM"))
  monofilecsv = CSVDateValReader(Dict(:filename=>monofile,:dateformat=>"dd/mm/yyyy HH:MM"))
  dailymonofilecsv = CSVDateValReader(Dict(:filename=>dailymonofile,:dateformat=>"dd/mm/yyyy HH:MM"))

  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1),:strict=>false))
  stfier = Statifier(Dict(:processmissing=>true))
  mono = Monotonicer(Dict())

  regpipeline = regularfilecsv |> valgator |> valnner |> mono
  regulardf=fit_transform!(regpipeline)

  monopipeline = monofilecsv |> valgator |> valnner |> mono
  monodf=fit_transform!(monopipeline)

  dailymonopipeline = dailymonofilecsv |> valgator |> valnner |> mono
  dailymonodf=fit_transform!(dailymonopipeline)

  @test round(dailyflips(regulardf),digits=2) == 11.98
  @test round(dailyflips(monodf),digits=2) == 10.32
  @test round(dailyflips(dailymonodf),digits=2) == 8.49
end
@testset "Monotonicer: check type format detection" begin
  test_basicmonotonicer()
end

end
