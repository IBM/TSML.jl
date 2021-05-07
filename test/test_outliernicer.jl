module TestOutliernicer

using StatsBase: sample, mean
using Test
using TSML

function test_artificialdata()
  Random.seed!(123)
  mdates = DateTime(2017,1,1):Dates.Hour(1):DateTime(2017,6,1)
  mvals = rand(1:1000,length(mdates))
  # create some outliers
  soutliers = rand([500:10000;-10000:500],div(length(mdates),10))
  soutndx = sample(1:length(mdates),length(soutliers))
  mvals[soutndx] = soutliers
  df = DataFrame(Date=mdates,Value=mvals)
  outnicer = Outliernicer(Dict(:dateinterval => Dates.Hour(1)))
  fit!(outnicer,df)
  resdf = transform!(outnicer,df)
  @test round(mean(resdf.Value),digits=2) > 0.0
  m=fit(outnicer,df)
  resdf = transform(m,df)
  @test round(mean(resdf.Value),digits=2) > 0.0
end
@testset "Outliernicer: using artificial data" begin
  test_artificialdata()
end


function test_basicoutlier()
  Random.seed!(123)
  fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
  stfier = Statifier(Dict(:processmissing=>true))
  mono = Monotonicer(Dict())
  outliernicer = Outliernicer(Dict(:dateinterval=>Dates.Hour(1)))

  mpipeline1 = csvfilter |> valgator |> mono |> valnner |> outliernicer |> stfier
  respipe1 = fit_transform!(mpipeline1)
  @test round(sum(respipe1[1,3:20])) == -206651.0
  respipe1 = fit_transform(mpipeline1)
  @test round(sum(respipe1[1,3:20])) == -206651.0

  mpipeline2 = csvfilter |> valgator |> mono |> outliernicer |> stfier
  respipe2 = fit_transform!(mpipeline2)
  @test round(sum(respipe2[1,3:20])) == -213860.0
  respipe2 = fit_transform(mpipeline2)
  @test round(sum(respipe2[1,3:20])) == -213860.0

  mpipeline3 = csvfilter |> valgator |> valnner |> mono |> outliernicer |> stfier
  respipe3 = fit_transform!(mpipeline3)
  @test round(sum(respipe3[1,3:20])) == -206651.0  
  respipe3 = fit_transform(mpipeline3)
  @test round(sum(respipe3[1,3:20])) == -206651.0  
end
@testset "Outliernicer: readcsv |> valgator |> valnner |> mono |> outliernicer |> stfier" begin
  test_basicoutlier()
end

function test_typesoutliernicer()
  regularfile = joinpath(dirname(pathof(TSML)),"../data/typedetection/regular.csv")
  monofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/monotonic.csv")
  dailymonofile = joinpath(dirname(pathof(TSML)),"../data/typedetection/dailymonotonic.csv")
  regularfilecsv = CSVDateValReader(Dict(:filename=>regularfile,:dateformat=>"dd/mm/yyyy HH:MM"))
  monofilecsv = CSVDateValReader(Dict(:filename=>monofile,:dateformat=>"dd/mm/yyyy HH:MM"))
  dailymonofilecsv = CSVDateValReader(Dict(:filename=>dailymonofile,:dateformat=>"dd/mm/yyyy HH:MM"))

  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
  stfier = Statifier(Dict(:processmissing=>true))
  mono = Monotonicer(Dict())
  outliernicer = Outliernicer(Dict(:dateinterval=>Dates.Hour(1)))

  regpipeline = regularfilecsv |> valgator |> valnner |> mono |> outliernicer |> stfier
  fit!(regpipeline)
  regulardf=transform!(regpipeline)
  @test round(sum(regulardf[1,3:20])) == -60438.0 
  m=fit(regpipeline)
  regulardf=transform(m)
  @test round(sum(regulardf[1,3:20])) == -60438.0 

  monopipeline = monofilecsv |> valgator |> valnner |> mono |> outliernicer |> stfier
  fit!(monopipeline)
  monodf=transform!(monopipeline)
  @test round(sum(monodf[1,3:20])) == -884621.0
  m=fit(monopipeline)
  monodf=transform(m)
  @test round(sum(monodf[1,3:20])) == -884621.0

  dailymonopipeline = dailymonofilecsv |> valgator |> valnner |> mono |> outliernicer |> stfier
  fit!(dailymonopipeline)
  dailymonodf=transform!(dailymonopipeline)
  @test round(sum(dailymonodf[1,3:20])) == -294446.0 
  m=fit(dailymonopipeline)
  dailymonodf=transform(m)
  @test round(sum(dailymonodf[1,3:20])) == -294446.0 

end
@testset "Outliernicer: monotonic type and outlier detections" begin
  test_typesoutliernicer()
end

end
