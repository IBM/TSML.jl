module TestTimescaleDB

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers

using TSML.TimescaleDBs

using DataFrames
using Dates
using Test

function test_timescaledb()
  tsdb = TimescaleDB()
  mpipeline = Pipeline(Dict(
         :transformers => [tsdb]
     )
  )
  fit!(mpipeline)
  respipe = transform!(mpipeline)
  @test sum(respipe[:Value]) ≈ 560997.0
  @test nrow(respipe) == 48349
  @test ncol(respipe) == 2
end
@testset "TimescaleDB" begin
  test_timescaledb()
end

end
