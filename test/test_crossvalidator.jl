module TestCrossValidator

using Test
using Random
using TSML

function test_crossvalidator()
  Random.seed!(123)
  acc(X,Y) = score(:accuracy,X,Y)
  data=getiris()
  X=data[:,1:4] |> Matrix
  Y=data[:,5] |> collect
  rf = RandomForest()
  @test CrossValidators.crossvalidate(rf,X,Y,acc).mean ≈ 94.0
  Random.seed!(123)
  ppl1 = Pipeline(Dict(:transformers=>[RandomForest()]))
  @test CrossValidators.crossvalidate(ppl1,X,Y,acc).mean ≈ 94.0
  Random.seed!(123)
  ohe = OneHotEncoder()
  stdsc= StandardScaler()
  ppl2 = Pipeline(Dict(:transformers=>[ohe,stdsc,RandomForest()]))
  @test CrossValidators.crossvalidate(ppl2,X,Y,acc).mean ≈ 94.66666667
  Random.seed!(123)
  mpca = Normalizer(Dict(:method=>:pca))
  mppca = Normalizer(Dict(:method=>:ppca))
  mfa = Normalizer(Dict(:method=>:fa))
  mlog = Normalizer(Dict(:method=>:log))
  msqrt = Normalizer(Dict(:method=>:sqrt))
  ppl3 = Pipeline(Dict(:transformers=>[msqrt,mlog,mpca,mppca,RandomForest()]))
  @test_throws BoundsError CrossValidators.crossvalidate(ppl3,X,Y,acc)
  Random.seed!(123)
  fit!(ppl3,X,Y)
  @test size(transform!(ppl3,X))[1] == length(Y)
  Random.seed!(123)
  ppl5 = Pipeline(Dict(:transformers=>[msqrt,mlog,mppca,RandomForest()]))
  @test CrossValidators.crossvalidate(ppl5,X,Y,acc).mean ≈ 77.333333
end
@testset "CrossValidator" begin
  test_crossvalidator()
end


end
