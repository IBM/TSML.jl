module TestNormalizer

using Test
using Random
using Dates
using DataFrames
using Statistics
using AMLPBase

function generatedf()
    Random.seed!(123)
    gdate = DateTime(2014,1,1):Dates.Minute(15):DateTime(2016,1,1)
    gval1 = rand(length(gdate))
    gval2 = rand(length(gdate))
    gval3 = rand(length(gdate))
    X = DataFrame(Date=gdate,Value1=gval1,Value2=gval2,Value3=gval3)
    X
end

function test_normalizer()
  Random.seed!(123)
  X = generatedf()
  norm = Normalizer(Dict(:method => :zscore))
  res=fit_transform!(norm,X)
  @test isapprox(mean(res[:,1]),0.0,atol=1e-8)
  @test isapprox(mean(res[:,2]),0.0,atol=1e-8)
  @test isapprox(std(res[:,1]),1.0,atol=1e-8)
  @test isapprox(std(res[:,2]),1.0,atol=1e-8)
  norm = Normalizer(Dict(:method => :unitrange))
  res=fit_transform!(norm,X)
  @test isapprox(minimum(res[:,1]),0.0,atol=1e-8)
  @test isapprox(minimum(res[:,2]),0.0,atol=1e-8)
  @test isapprox(maximum(res[:,1]),1.0,atol=1e-8)
  @test isapprox(maximum(res[:,2]),1.0,atol=1e-8)
  norm = Normalizer(Dict(:method => :pca))
  res=fit_transform!(norm,X)
  @test isapprox(std(res[:,1]),0.28996,atol=1e-2)
  norm = Normalizer(Dict(:method => :fa))
  res = fit_transform!(norm,X)
  @test isapprox(std(res[:,1]),0.81670,atol=1e-2)
  norm = Normalizer(Dict(:method => :ppca))
  res = fit_transform!(norm,X)
  @test isapprox(std(res[:,1]),0.00408,atol=1e-2)
  norm = Normalizer(Dict(:method => :log))
  res = fit_transform!(norm,X)
  @test isapprox(std(res[:,1]),0.99941,atol=1e-2)
  norm = Normalizer(Dict(:method => :sqrt))
  res = fit_transform!(norm,X)
  @test isapprox(std(res[:,1]),0.2355,atol=1e-2)
end
@testset "Normalizer: zscore, unitrange, pca, ppca, fa" begin
  test_normalizer()
end

end
