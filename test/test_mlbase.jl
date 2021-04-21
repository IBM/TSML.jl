module TestMLBaseWrapper

using Test
using TSML
using DataFrames: DataFrame

mtodf(x) = DataFrame(x,:auto)

@testset "MLBase transformers" begin

  @testset "StandardScaler transforms features" begin
    features = [
      5 10;
      -5 0;
      0 5;
    ] |> mtodf
    labels = [
      "x";
      "y";
      "z";
    ]
    expected_transformed = [
      1.0 1.0;
      -1.0 -1.0;
      0.0 0.0;
    ] |> mtodf
    standard_scaler = StandardScaler()
    fit!(standard_scaler, features, labels)
    transformed = TSML.transform!(standard_scaler, features)
    @test (transformed .== expected_transformed) |> Matrix |> sum == 6
  end

end

end # module
