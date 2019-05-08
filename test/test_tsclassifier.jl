module TestTSClassifier

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers

using TSML.TSClassifiers
using TSML.TSClassifiers: TSClassifier

using Test

function test_tsclassifier()
  @test_throws ErrorException  TSClassifier(Dict()) 
  trdirname = joinpath(dirname(pathof(TSML)),"../data/tsclassification/training")
  tstdirname = joinpath(dirname(pathof(TSML)),"../data/tsclassification/testing")
  modeldirname = joinpath(dirname(pathof(TSML)),"../data/tsclassification/model")
  tscl = TSClassifier(Dict(:trdirectory=>trdirname,
			   :tstdirectory=>tstdirname,
			   :modeldirectory=>modeldirname,
			   :num_trees=>30))
  modelfname = joinpath(tscl.args[:modeldirectory],tscl.args[:juliarfmodelname])
  fit!(tscl)
  @test isfile(modelfname)
  @test length(transform!(tscl)) > 0
  # cleanup model directory
  if isdir(modeldirname)
    rm(modelfname,force=true)
  end
end

@testset "TSClassifier" begin
  test_tsclassifier()
end

end
