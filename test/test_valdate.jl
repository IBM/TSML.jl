module TestDateVal

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers

using Random
using Statistics
using DataFrames
using Dates
using MLDataUtils
using Test

function generateXY()
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X[:Value][gndxmissing] .= missing
    Y = rand(length(gdate))
    (X,Y)
end

const (XX,YY)=generateXY()
const (X1,Y1)=generateXY()

function test_datevalizer()
    dvzr1 = DateValizer(Dict(:dateinterval=>Dates.Hour(1)))
    dvzr2 = DateValizer(dvzr1.args)
    @test dvzr1.args == dvzr2.args
    fit!(dvzr2,XX,YY)
    @test size(dvzr2.args[:medians]) == (24,2)
    res = transform!(dvzr2,XX)
    @test sum(ismissing.(res[:Value])) == 0
    @test sum(X1[:Value] .!== XX[:Value]) == 0
    @test sum(Y1 .!== YY) == 0
    @test round(sum(res[:Value]),digits=2) == 8796.64
    @test nrow(dvzr2.args[:medians]) == 24
end
@testset "DateValizer: Fill missings with medians" begin
    test_datevalizer()
end

function test_datevalgator()
    dtvl = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
    fit!(dtvl,XX,YY)
    res = transform!(dtvl,XX)
    @test sum(ismissing.(res[:Value])) == 4466
    @test round(sum(skipmissing(res[:Value])),digits=2) == 6557.97
    @test sum(X1[:Value] .!== XX[:Value]) == 0
    @test sum(Y1 .!== YY) == 0
end
@testset "DateValgator: aggregate by timeperiod without filling missings" begin
    test_datevalgator()
end

function test_datevalnner()
    dnnr = DateValNNer(Dict(:dateinterval=>Dates.Hour(25),:nnsize=>10,:missdirection => :forward))
    fit!(dnnr,XX,YY)
    res=transform!(dnnr,XX)
    @test sum(size(res) .== (701,2)) == 2
    @test round(sum(res[:Value]),digits=2) == 352.1
    dnnr.args[:missdirection] = :reverse
    res=transform!(dnnr,XX)
    @test sum(size(res) .== (701,2)) == 2
    @test round(sum(res[:Value]),digits=2) == 352.19
    dnnr.args[:dateinterval]=Dates.Hour(1)
    @test_throws ErrorException res=transform!(dnnr,XX) 
    dnnr.args[:missdirection] = :forward
    @test_throws ErrorException res=transform!(dnnr,XX) 
    # testing boundaries
    Random.seed!(123)
    dlnr = DateValNNer(Dict(:dateinterval=>Dates.Hour(1),:nnsize=>2,:missdirection => :forward))
    v1=DateTime(2014,1,1,1,0):Dates.Hour(1):DateTime(2014,1,3,1,0)
    val=Array{Union{Missing,Float64}}(rand(length(v1)))
    x=DataFrame(Date=v1,Value=val)
    x[45:end-1,:Value] = missing
    fit!(dlnr,x,[])
    @test_throws ErrorException transform!(dlnr,x)
    dlnr.args[:missdirection] = :reverse
    res = transform!(dlnr,x)
    @test sum((size(res)) .== (49,2)) == 2
    @test round(sum(res[:Value]),digits=2) == 27.85
    dlnr.args[:missdirection] = :forward
    dlnr.args[:strict] = false
    @test sum(ismissing.(transform!(dlnr,x)[:Value])) == 2
    defdnr=DateValNNer(Dict(:strict=>false))
    fit!(defdnr,XX,YY)
    @test sum((size(transform!(defdnr,XX))) .== (17521,2)) == 2
end
@testset "DateValNNer: replace missings with nearest neighbors" begin
    test_datevalnner()
end

function test_dateifier()
  dtr = Dateifier(Dict())
  lower = DateTime(2017,1,1)
  upper = DateTime(2018,1,31)
  x=lower:Dates.Day(1):upper |> collect
  y = x
  fit!(dtr,x,y)
  res = transform!(dtr,x)
  @test sum(size(res) .== (389,8)) == 2
  dtr.args[:stride]=2
  res = transform!(dtr,x)
  @test sum(size(res) .== (194,8)) == 2
end
@testset "Dateifier: extract sliding windows date features" begin
    test_dateifier()
end

function test_matrifier()
  mtr = Matrifier(Dict(:ahead=>24,:size=>24,:stride=>5))
  sz = mtr.args[:size]
  x=collect(1:100)
  y=collect(1:100)
  fit!(mtr,x,y)
  res = transform!(mtr,x)
  @test sum(size(res) .== (11,25)) == 2
  mtr.args = Dict(:ahead=>24,:size=>24,:stride=>1)
  res = transform!(mtr,x)
  @test sum(size(res) .== (53,25)) == 2
  mtr.args = Dict(:ahead=>1,:size=>24,:stride=>12)
  res = transform!(mtr,x)
  @test sum(size(res) .== (6,25)) == 2
  dtr = Matrifier()
  fit!(dtr,x,y)
  res = transform!(dtr,x)
  @test sum(size(res) .== (93,8)) == 2
  dtr.args = Dict(:ahead=>-1,:size=>24,:stride=>12)
  @test_throws AssertionError transform!(dtr,x)
end
@testset "Dateifier: extract sliding windows date features" begin
    test_matrifier()
end

end
