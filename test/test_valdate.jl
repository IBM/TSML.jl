module TestValDate

using Test
using TSML
using DataFrames: DataFrame, nrow

function generateXY()
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X.Value[gndxmissing] .= missing
    Y = rand(length(gdate))
    (X,Y)
end

const (XX,YY)=generateXY()
const (X1,Y1)=generateXY()

function test_datevalizer()
    dvzr1 = DateValizer(Dict(:dateinterval=>Dates.Hour(1)))
    dvzr2 = DateValizer(dvzr1.model)
    @test dvzr1.model[:dateinterval] == dvzr2.model[:dateinterval]
    res = fit_transform!(dvzr2,XX)
    @test sum(ismissing.(res.Value)) == 0
    @test sum(X1.Value .!== XX.Value) == 0
    @test sum(Y1 .!== YY) == 0
    @test round(sum(res.Value),digits=2) == 8798.2
    @test nrow(dvzr2.model[:medians]) == 24
    res = fit_transform(dvzr2,XX)
    @test sum(ismissing.(res.Value)) == 0
    @test sum(X1.Value .!== XX.Value) == 0
    @test sum(Y1 .!== YY) == 0
    @test round(sum(res.Value),digits=2) == 8798.2
    @test nrow(dvzr2.model[:medians]) == 24
end
@testset "DateValizer: Fill missings with medians" begin
    test_datevalizer()
end

function test_datevalgator()
    dtvl = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
    res = fit_transform!(dtvl,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 6556.17
    @test sum(X1.Value .!== XX.Value) == 0
    @test sum(Y1 .!== YY) == 0
    res = fit_transform(dtvl,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 6556.17
    @test sum(X1.Value .!== XX.Value) == 0
    @test sum(Y1 .!== YY) == 0

    dtvlmean = DateValgator(Dict(
	  :dateinterval=>Dates.Hour(1),
	  :aggregator => :mean))
    res = fit_transform!(dtvlmean,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 6557.97
    res = fit_transform(dtvlmean,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 6557.97

    dtvlmax = DateValgator(Dict(
	  :dateinterval=>Dates.Hour(1),
	  :aggregator => :maximum))
    res = fit_transform!(dtvlmax,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 7599.95
    res = fit_transform(dtvlmax,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 7599.95

    dtvlmin = DateValgator(Dict(
	  :dateinterval=>Dates.Hour(1),
	  :aggregator => :minimum))
    res = fit_transform!(dtvlmin,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 5516.92
    res = fit_transform(dtvlmin,XX)
    @test sum(ismissing.(res.Value)) == 4466
    @test round(sum(skipmissing(res.Value)),digits=2) == 5516.92
end
@testset "DateValgator: aggregate by timeperiod without filling missings" begin
    test_datevalgator()
end

function test_datevalnner()
    dnnr = DateValNNer(Dict(
	  :dateinterval=>Dates.Hour(1),
	  :nnsize=>10,
	  :missdirection => :symmetric,
	  :strict=>true,
	  :aggregator => :mean))
    res=fit_transform!(dnnr,XX)
    @test sum(size(res) .== (17521,2)) == 2
    @test round(sum(res.Value),digits=2) == 8807.28
    res=fit_transform(dnnr,XX)
    @test sum(size(res) .== (17521,2)) == 2
    @test round(sum(res.Value),digits=2) == 8807.28

    dnnr = DateValNNer(Dict(
	  :dateinterval=>Dates.Hour(1),
	  :nnsize=>10,
	  :missdirection => :symmetric,
	  :strict=>true,
	  :aggregator => :maximum))
    res=fit_transform!(dnnr,XX)
    @test sum(size(res) .== (17521,2)) == 2
    @test round(sum(res.Value),digits=2) == 10339.47
    res=fit_transform(dnnr,XX)
    @test sum(size(res) .== (17521,2)) == 2
    @test round(sum(res.Value),digits=2) == 10339.47
    dnnr = DateValNNer(Dict(
	  :dateinterval=>Dates.Hour(1),
	  :nnsize=>10,
	  :missdirection => :symmetric,
	  :strict=>true,
	  :aggregator => :minimum))
    res=fit_transform!(dnnr,XX)
    @test sum(size(res) .== (17521,2)) == 2
    @test round(sum(res.Value),digits=2) == 7290.01
    res=fit_transform(dnnr,XX)
    @test sum(size(res) .== (17521,2)) == 2
    @test round(sum(res.Value),digits=2) == 7290.01

    dnnr = DateValNNer(Dict(
	  :dateinterval=>Dates.Hour(25),
	  :nnsize=>10,
	  :missdirection => :forward,
	  :strict=>true))
    res=fit_transform!(dnnr,XX)
    @test sum(size(res) .== (701,2)) == 2
    @test round(sum(res.Value),digits=2) == 350.57
    res=fit_transform(dnnr,XX)
    @test sum(size(res) .== (701,2)) == 2
    @test round(sum(res.Value),digits=2) == 350.57

    dnnr.model[:missdirection] = :reverse
    res=TSML.transform!(dnnr,XX)
    @test sum(size(res) .== (701,2)) == 2
    @test round(sum(res.Value),digits=2) == 350.17
    dnnr.model[:dateinterval]=Dates.Hour(1)
    @test_throws ErrorException res=TSML.transform!(dnnr,XX) 
    dnnr.model[:missdirection] = :reverse
    @test_throws ErrorException res=TSML.transform!(dnnr,XX) 
    dnnr.model[:missdirection] = :symmetric
    @test sum(size(TSML.transform!(dnnr,XX)) .== (17521,2)) == 2

    # testing boundaries
    Random.seed!(123)
    dlnr = DateValNNer(Dict(
	    :dateinterval=>Dates.Hour(1),
	    :nnsize=>2,
	    :missdirection => :forward,
	    :strict=>true))
    v1=DateTime(2014,1,1,1,0):Dates.Hour(1):DateTime(2014,1,3,1,0)
    val=Array{Union{Missing,Float64}}(rand(length(v1)))
    x=DataFrame(Date=v1,Value=val)
    x[45:end-1,:Value] .= missing
    fit!(dlnr,x,[])
    @test_throws ErrorException TSML.transform!(dlnr,x)
    @test_throws ErrorException fit_transform(dlnr,x)
    dlnr.model[:missdirection] = :reverse
    res = TSML.transform!(dlnr,x)
    @test sum((size(res)) .== (49,2)) == 2
    @test round(sum(res.Value),digits=2) == 25.3
    dlnr.model[:missdirection] = :forward
    dlnr.model[:strict] = false
    @test sum(ismissing.(TSML.transform!(dlnr,x)[!,:Value])) == 2
    dlnr.model[:missdirection] = :symmetric
    @test sum(ismissing.(TSML.transform!(dlnr,x)[!,:Value])) == 0
    defdnr=DateValNNer(Dict(:strict=>false))
    fit!(defdnr,XX,YY)
    @test sum((size(TSML.transform!(defdnr,XX))) .== (17521,2)) == 2
    m=fit(defdnr,XX,YY)
    @test sum((size(TSML.transform(m,XX))) .== (17521,2)) == 2

    # test with controlled locations of missings
    dlnr = DateValNNer(Dict(:dateinterval=>Dates.Hour(1),
			    :nnsize=>10,:strict=>false,
			    :missdirection => :forward))
    v1=DateTime(2014,1,1,1,0):Dates.Hour(1):DateTime(2014,1,3,1,0)
    val=Array{Union{Missing,Float64}}(collect(1:(length(v1))))
    x=DataFrame(Date=v1,Value=val)
    x[45:end,:Value] .= missing
    x[1:10,:Value] .= missing
    x[20:30,:Value] .= missing
    fit!(dlnr,x,[])
    res = TSML.transform!(dlnr,x)
    @test sum(ismissing.(TSML.transform!(dlnr,x)[!,:Value])) == 6
    dlnr.model[:missdirection] = :reverse
    @test sum(ismissing.(TSML.transform!(dlnr,x)[!,:Value])) == 5 
    dlnr.model[:missdirection] = :symmetric
    @test sum(ismissing.(TSML.transform!(dlnr,x)[!,:Value])) == 0
end
@testset "DateValNNer: replace missings with nearest neighbors" begin
    test_datevalnner()
end

function test_dateifier()
  dtr = Dateifier(Dict())
  lower = DateTime(2017,1,1)
  upper = DateTime(2018,1,31)
  dat=lower:Dates.Day(1):upper |> collect 
  vals = rand(length(dat))
  x=DataFrame(Date=dat,Value=vals)
  fit!(dtr,x)
  res = TSML.transform!(dtr,x)
  @test sum(size(res) .== (389,8)) == 2
  m=fit(dtr,x)
  res = TSML.transform(m,x)
  @test sum(size(res) .== (389,8)) == 2
  dtr.model[:stride]=2
  res = TSML.transform!(dtr,x)
  @test sum(size(res) .== (194,8)) == 2
end
@testset "Dateifier: extract sliding windows date features" begin
    test_dateifier()
end

function test_matrifier()
  mtr = Matrifier(Dict(:ahead=>24,:size=>24,:stride=>5))
  sz = mtr.model[:size]
  lower = DateTime(2017,1,1)
  upper = DateTime(2017,1,5)
  dat=lower:Dates.Hour(1):upper |> collect 
  vals = 1:length(dat)
  x = DataFrame(Date=dat,Value=vals)
  res = fit_transform!(mtr,x)
  @test sum(size(res) .== (10,25)) == 2
  res = fit_transform(mtr,x)
  @test sum(size(res) .== (10,25)) == 2
  mtr.model = Dict(:ahead=>24,:size=>24,:stride=>1)
  res = TSML.transform!(mtr,x)
  @test sum(size(res) .== (50,25)) == 2
  mtr.model = Dict(:ahead=>1,:size=>24,:stride=>12)
  res = TSML.transform!(mtr,x)
  @test sum(size(res) .== (6,25)) == 2
  dtr = Matrifier()
  fit!(dtr,x)
  res = TSML.transform!(dtr,x)
  @test sum(size(res) .== (90,8)) == 2
  m=fit(dtr,x)
  res = TSML.transform(m,x)
  @test sum(size(res) .== (90,8)) == 2
  dtr.model = Dict(:ahead=>-1,:size=>24,:stride=>12)
  @test_throws AssertionError TSML.transform!(dtr,x)
end
@testset "Dateifier: extract sliding windows date features" begin
    test_matrifier()
end

function test_csvreaderwriter()
  inputfile =joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  outputfile = joinpath(dirname(pathof(TSML)),"../data/testdata_output.csv")
  csvreader = CSVDateValReader(Dict(:filename=>inputfile,:dateformat=>"d/m/y H:M"))
  csvwtr = CSVDateValWriter(Dict(:filename=>outputfile,:dateformat=>"d/m/y H:M"))
  filter1 = DateValgator()
  filter2 = DateValNNer(Dict(:nnsize=>1,:strict=>:false))
  mypipeline = csvreader |> filter1 |> filter2
  res=fit_transform!(mypipeline)
  @test nrow(res) == 8761
  @test ncol(res) == 2
  @test sum(ismissing.(res.Value)) == 0
  @test floor(sum(res.Value)) == 93080.0
  res=fit_transform(mypipeline)
  @test nrow(res) == 8761
  @test ncol(res) == 2
  @test sum(ismissing.(res.Value)) == 0
  @test floor(sum(res.Value)) == 93080.0
  dat = fit_transform!(csvreader)
  res1=fit_transform!(filter1,dat)
  res2=fit_transform!(filter2,res1)
  @test nrow(res2) == 8761
  @test ncol(res2) == 2
  @test sum(ismissing.(res2.Value)) == 0
  @test floor(sum(res2.Value)) == 93080.0
  dat = fit_transform(csvreader)
  res1=fit_transform(filter1,dat)
  res2=fit_transform(filter2,res1)
  @test nrow(res2) == 8761
  @test ncol(res2) == 2
  @test sum(ismissing.(res2.Value)) == 0
  @test floor(sum(res2.Value)) == 93080.0
  mypipeline = csvreader |> filter1 |> filter2 |> csvwtr
  res=fit_transform!(mypipeline)
  @test nrow(res2) == 8761
  @test ncol(res2) == 2
  @test sum(ismissing.(res2.Value)) == 0
  @test floor(sum(res2.Value)) == 93080.0
  @test filesize(csvwtr.model[:filename]) > 209220
  rm(outputfile,force=true)
end
@testset "CSVDateValReaderWriter: reading csv with Date,Value columns" begin
  test_csvreaderwriter()
end

#function test_bzcsvreaderwriter()
#  inputfile =joinpath(dirname(pathof(TSML)),"../data/testdata.csv.bz2")
#  csvreader = BzCSVDateValReader(Dict(:filename=>inputfile,:dateformat=>"d/m/y H:M"))
#  filter1 = DateValgator()
#  filter2 = DateValNNer(Dict(:nnsize=>1))
#  mypipeline = csvreader |> filter1 |> filter2
#  res=fit_transform!(mypipeline)
#  @test nrow(res) == 8761
#  @test ncol(res) == 2
#end
#@testset "BzCSVDateValReader: reading bzipped csv with Date,Value columns" begin
#  test_bzcsvreaderwriter()
#end

function test_statoutputwriter()
  inputfile =joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
  statoutputfile = joinpath(dirname(pathof(TSML)),"../data/testdata_stat.csv")
  csvreader = CSVDateValReader(Dict(:filename=>inputfile,:dateformat=>"d/m/y H:M"))
  csvstatwtr = CSVDateValWriter(Dict(:filename=>statoutputfile))
  statfier =  Statifier(Dict(:processmissing=>true))
  filter1 = DateValgator(Dict(:nnsize=>1))
  filter2 = DateValNNer(Dict(:nnsize=>1,:strict=>false))

  mypipeline = csvreader |> filter1  |> filter2 |> statfier  |> csvstatwtr
  res=fit_transform!(mypipeline)
  @test nrow(res) == 1
  @test ncol(res) == 26
  @test filesize(csvstatwtr.model[:filename]) > 400
  rm(statoutputfile,force=true)

end
@testset "CSVDateValReaderWriter: writing stat output" begin
  test_statoutputwriter()
end

function test_datevalmultinner()
  Random.seed!(123)
  gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
  gval1 = Array{Union{Missing,Float64}}(rand(length(gdate)))
  gval2 = Array{Union{Missing,Float64}}(rand(length(gdate)))
  gval3 = Array{Union{Missing,Float64}}(rand(length(gdate)))
  gmissing = 50000
  gndxmissing1 = Random.shuffle(1:length(gdate))[1:gmissing]
  gndxmissing2 = Random.shuffle(1:length(gdate))[1:gmissing]
  gndxmissing3 = Random.shuffle(1:length(gdate))[1:gmissing]
  X = DataFrame(Date=gdate,Temperature=gval1,Humidity=gval2,Ozone=gval3)
  X.Temperature[gndxmissing1] .= missing
  X.Humidity[gndxmissing2] .= missing
  X.Ozone[gndxmissing3] .= missing
  dnnr = DateValMultiNNer(Dict(:type=>:linear))
  res = fit_transform!(dnnr,X)
  mysum(x)=sum(skipmissing(x))
  @test mysum.(eachcol(res[:,2:end])) |> sum == 26363.629570617708
  res = fit_transform(dnnr,X)
  @test mysum.(eachcol(res[:,2:end])) |> sum == 26363.629570617708
  dnnr = DateValMultiNNer(Dict(:type=>:knn))
  res = fit_transform!(dnnr,X)
  @test mysum.(eachcol(res[:,2:end]))  |> sum == 26368.053898361875
  res = fit_transform(dnnr,X)
  @test mysum.(eachcol(res[:,2:end]))  |> sum == 26368.053898361875
  dnnr = DateValMultiNNer()
  @test_throws ArgumentError fit!(dnnr,DataFrame(Date=X.Date,Value1=X.Date,Value2=X.Temperature))
  @test_throws ArgumentError fit(dnnr,DataFrame(Date=X.Date,Value1=X.Date,Value2=X.Temperature))
end
@testset "DateValMultiNNer: multicolumn imputation" begin
  test_datevalmultinner()
end

function test_datevallinearimputer()
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval = Array{Union{Missing,Float64}}(rand(length(gdate)))
    gmissing = 50000
    gndxmissing = Random.shuffle(1:length(gdate))[1:gmissing]
    X = DataFrame(Date=gdate,Value=gval)
    X.Value[gndxmissing] .= missing
    dnnr = DateValLinearImputer()
    @test fit_transform!(dnnr,X) |> x-> sum(x.Value) == 8791.719321255328
    @test fit_transform(dnnr,X) |> x-> sum(x.Value) == 8791.719321255328
end
@testset "DateValLinearImputer: linear imputation" begin
  test_datevallinearimputer()
end

end
