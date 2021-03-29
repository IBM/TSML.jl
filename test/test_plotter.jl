module TestPlotter

using Test
using Plots
using StatsBase: sample 
using TSML

default(show=false, reuse=true)

function generatedf()
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2014,1,5)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = floor(0.30*length(gdate)) |> Integer
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    df = DataFrame(Date=gdate,Value=gval)
    df.Value[gndxmissing] .= missing
    return df
end;

function test_artificialdataplotter()
  Random.seed!(123)
  mdates = DateTime(2017,1,1):Dates.Hour(1):DateTime(2017,6,1)
  mvals = rand(1:1000,length(mdates))
  # create some outliers
  soutliers = rand([500:10000;-10000:500],div(length(mdates),10))
  soutndx = sample(1:length(mdates),length(soutliers))
  mvals[soutndx] = soutliers
  df = DataFrame(Date=mdates,Value=mvals)
  pltr = TSML.Plotter(Dict(:interactive => false))
  fit!(pltr,df)
  myplot=transform!(pltr,df);
  @test isa(myplot,Plots.Plot) == true;
  df = generatedf()
  fit!(pltr,df)
  myplot1=transform!(pltr,df);
  @test isa(myplot1,Plots.Plot) == true;
end
@testset "Plotter: using artificial data" begin
  test_artificialdataplotter()
end

function test_realdataplotter()
  Random.seed!(123)
  fname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
  pltr = TSML.Plotter(Dict(:interactive => false))
  mpipeline1 = csvfilter |> pltr
  myplot = fit_transform!(mpipeline1);
  @test isa(myplot,Plots.Plot) == true;
end
@testset "Plotter: readcsv |> plotter" begin
  test_realdataplotter()
end

end
