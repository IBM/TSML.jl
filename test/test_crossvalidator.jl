module TestCrossValidator

using Test
using Random
using TSML

function test_crossvalidator()
  Random.seed!(123)
  acc(X,Y) = score(:accuracy,X,Y)
  data=getiris()
  ppl1 = Pipeline(Dict(:transformers=>[RandomForest()]))
  @test crossvalidate(ppl1,data,[1:4],[],5,acc).mean ≈ 94.0
  ohe = OneHotEncoder()
  stdsc= StandardScaler()
  ppl2 = Pipeline(Dict(:transformers=>[ohe,stdsc,RandomForest()]))
  @test crossvalidate(ppl2,data,[1:4],[],5,acc).mean ≈ 94.0
  mpca = Normalizer(Dict(:method=>:pca))
  mppca = Normalizer(Dict(:method=>:ppca))
  mfa = Normalizer(Dict(:method=>:fa))
  mlog = Normalizer(Dict(:method=>:log))
  msqrt = Normalizer(Dict(:method=>:sqrt))
  ppl3 = Pipeline(Dict(:transformers=>[msqrt,mlog,mpca,mppca,RandomForest()]))
  fit!(ppl3,data[:,1:4],collect(data[:,5]))
  res=transform!(ppl3,data[:,1:4])
  @test score(:accuracy,res,collect(data[:,5])) ≈ 99.333333
  ppl3 = Pipeline(Dict(:transformers=>[msqrt,mlog,mppca,RandomForest()]))
  @test crossvalidate(ppl3,data,[1:4],[],5,acc,5).mean ≈ 86.666666
end
@testset "CrossValidator" begin
  test_crossvalidator()
end

end
