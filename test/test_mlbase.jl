module TestMLBaseWrapper

using Test
using TSML

@testset "MLBase transformers" begin

  @testset "StandardScaler transforms features" begin
    features = [
      5 10;
      -5 0;
      0 5;
    ]
    labels = [
      "x";
      "y";
      "z";
    ]
    expected_transformed = [
      1.0 1.0;
      -1.0 -1.0;
      0.0 0.0;
    ]
    standard_scaler = StandardScaler()
    fit!(standard_scaler, features, labels)
    transformed = transform!(standard_scaler, features)
    @test transformed == expected_transformed
  end

end

end # module
