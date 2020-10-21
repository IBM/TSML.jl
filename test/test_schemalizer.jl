module TestSchemalizer

using Test
using TSML
using TSML.Schemalizers

function generatedf()
  Random.seed!(123)
  gdate = DateTime(2015,1,1):Dates.Minute(30):DateTime(2015,1,2) |> collect
  len = length(gdate)
  sex = rand(['m','f'],len)
  x1=rand(1:100,len)
  x2=rand(1:100,len)
  x3=rand(1:1000,len)
  DataFrame(date=gdate,sex=sex,f1=x1,f2=x2,f3=x3)
end

function test_schemalizer()
  df = generatedf()
  m = Schemalizer(Dict(:schema=>
      Dict(:sex => ML.Categorical)))
  fit!(m,df)
  res=transform!(m,df)
  @test isapprox(sum(Matrix(res.^2)),193,atol=1e-2)
  m = Schemalizer(Dict())
  fit!(m,df)
  res = transform!(m,df)
  @test isapprox(sum(Matrix(res.^2)),144,atol=1e-2)
end
@testset "Schemalizer" begin
  test_schemalizer()
end

end
