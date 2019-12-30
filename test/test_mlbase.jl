module TestMLBaseWrapper

using Test
using TSML
using DataFrames

@testset "MLBase transformers" begin

  @testset "StandardScaler transforms features" begin
    features = [
      5 10;
      -5 0;
      0 5;
    ] |> DataFrame
    labels = [
      "x";
      "y";
      "z";
    ]
    expected_transformed = [
      1.0 1.0;
      -1.0 -1.0;
      0.0 0.0;
    ] |> DataFrame
    standard_scaler = StandardScaler()
    fit!(standard_scaler, features, labels)
    transformed = transform!(standard_scaler, features)
    @test (transformed .== expected_transformed) |> Matrix |> sum == 6
  end

end

end # module
