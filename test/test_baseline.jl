module TestBaseline

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML.BaselineAlgos
using Random
using Test
using DataFrames

function test_baseline()
    Random.seed!(123)
    iris=getiris()
    instances=iris[:,1:4] |> Matrix
    labels=iris[:,5] |> Vector
    bl = Baseline()
    fit!(bl,instances,labels)
    @test bl.model == "setosa"
    @test sum(transform!(bl,instances) .== repeat(["setosa"],nrow(iris))) == nrow(iris)
    idy = Identity()
    fit!(idy,instances,labels)
    @test idy.model == nothing
    @test sum(transform!(idy,instances) .== instances) == (*(size(instances)...))
    @test idy.args == Dict()
end
@testset "Baseline Tests" begin
  test_baseline()
end



end
