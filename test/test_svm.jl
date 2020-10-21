module TestDecisionTree

using Test
using TSML
using Random
using DataFrames: nrow, DataFrame
using Statistics

function generateXY()
    Random.seed!(123)
    iris = getiris()
    indx = Random.shuffle(1:nrow(iris))
    features=iris[indx,1:4] 
    sp = iris[indx,5] |> Vector
    z = rand(nrow(iris))
    (features,sp,z)
end

rmse(x,y)= sqrt(mean((x .- y).^2))
acc(x,y)=score(:accuracy,x,y)

function test_libsvm()
   X,Y,Z = generateXY()

   @test_throws ArgumentError SVMModel("SVCP"; gamma = :auto)
   linear=SVMModel("LinearSVC"; gamma = :auto)
   @test_throws MethodError fit!(linear,X,Y)

   model = SVMModel("OneClassSVM")
   @test fit_transform!(model,X,Y) |> sum == 133

   learner1 = SVMModel("SVC"; gamma = :auto)
   learner2 = SVMModel("SVC",Dict(:gamma => :auto))

   @test learner1.model[:impl_args] == learner2.model[:impl_args]
   for learner in ["NuSVC","SVC","LinearSVC"]
      model = SVMModel(learner)
      @test crossvalidate(model,X,Y,acc,10,false).mean > 90.0
   end

   for learner in ["NuSVR","EpsilonSVR"]
      model = SVMModel(learner, tolerance = 0.01)
      @test crossvalidate(model,X,Z,rmse,10,false).mean < 0.3
   end
end
@testset "LIBSVM" begin
   Random.seed!(123)
   test_libsvm()
end

end
