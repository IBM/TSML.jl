module TestTimescaleDB

using Test
using TSML

function test_timescaledb()
  tsdb = TimescaleDB()
  mpipeline = tsdb
  try 
    respipe = fit_transform!(mpipeline)
    @test sum(respipe.Value) ≈ 560997.0
    @test nrow(respipe) == 48349
    @test ncol(respipe) == 2
  catch e
  end
end
@testset "TimescaleDB" begin
  test_timescaledb()
end

end
