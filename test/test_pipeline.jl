module TestPipeline

using Test
using TSML
using DataFrames

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

function test_pipeline()
  dtvalgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  dtvalnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1),:strict=>true,:nnsize=>1,:missdirection=>:symmetric))
  dtr = Dateifier(Dict())
  mtr = Matrifier(Dict())
  fit!(mtr,XX,YY)
  ## try pipeline 
  mydatepipeline = Pipeline(Dict(
    :transformers => [
	dtvalgator,
	dtvalnner,
	dtr
    ]
  ))
  fit!(mydatepipeline,XX,YY)
  date=transform!(mydatepipeline,XX)
  myvalpipeline = Pipeline(Dict(
    :transformers => [
	dtvalgator,
	dtvalnner,
	mtr
    ]
  ))
  fit!(myvalpipeline,XX,YY)
  val=transform!(myvalpipeline,XX)
  @test sum(size(val) .== size(date)) == 2
end
@testset "Pipeline: check " begin
  test_pipeline()
end


function test_pipeline_macro()
  data = getiris()
  X=data[:,1:5]
  Y=data[:,5] |> Vector
  X[!,5]= X[!,5] .|> string
  ohe = OneHotEncoder()
  ohe1 = OneHotEncoder()
  linear1 = Pipeline(Dict(:name=>"lp",:machines => [ohe]))
  linear2 = Pipeline(Dict(:name=>"lp",:machines => [ohe]))
  combo1 = ComboPipeline(Dict(:machines=>[ohe,ohe]))
  combo2 = ComboPipeline(Dict(:machines=>[linear1,linear2]))
  linear1 = Pipeline([ohe])
  linear2 = Pipeline([ohe])
  combo1 = ComboPipeline([ohe,ohe])
  combo2 = ComboPipeline([linear1,linear2])
  fit!(combo1,X)
  res1=transform!(combo1,X)
  res2=fit_transform!(combo1,X)
  @test (res1 .== res2) |> Matrix |> sum == 2100
  fit!(combo2,X)
  res3=transform!(combo2,X)
  res4=fit_transform!(combo2,X)
  @test (res3 .== res4) |> Matrix |> sum == 2100
  pcombo1 = @pipeline ohe1 + ohe1
  pres1 = fit_transform!(pcombo1,X)
  @test (pres1 .== res1) |> Matrix |> sum == 2100
  features = data[:,1:4]
  pca = Normalizer(:pca)
  fa = Normalizer(:fa)
  ica = Normalizer(:ica)
  sq = Normalizer(:sqrt)
  pcombo2 = @pipeline (pca |> fa) + (fa |> pca)
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 155
  pcombo2 = @pipeline sq |> pca |> fa
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 152
  disc = CatNumDiscriminator()
  catf = CatFeatureSelector()
  numf = NumFeatureSelector()
  rf = RandomForest()
  pcombo3 = @pipeline disc |> ((catf |> numf) + (numf |> pca) + (numf |> fa) + (catf |> ohe)) |> rf
  (fit_transform!(pcombo3,X,Y)  .== Y) |> sum == 150
end
@testset "Pipeline Macro" begin
    test_pipeline_macro()
end


end
