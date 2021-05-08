module TestBaseline

using Test
using TSML

function test_baseline()
    Random.seed!(123)
    iris=getiris()
    instances=iris[:,1:4] 
    labels=iris[:,5] |> collect
    bl = Baseline()
    fit!(bl,instances,labels)
    @test bl.model == "setosa"
    @test sum(transform!(bl,instances) .== repeat(["setosa"],nrow(iris))) == nrow(iris)
    idy = Identity()
    fit!(idy,instances,labels)
    @test idy.model == nothing
    @test (transform!(idy,instances) .== instances) |> Matrix |> sum == 150*4
    @test idy.args == Dict()
    m = fit(idy,instances,labels)
    @test (transform(m,instances) .== instances) |> Matrix |> sum == 150*4
end
@testset "Baseline Tests" begin
  test_baseline()
end



end
