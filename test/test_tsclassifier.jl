module TestTSClassifier

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers

using TSML.TSClassifiers
using TSML.TSClassifiers: TSClassifier

using DataFrames
using Test

function test_tsclassifier()
  tscl=TSClassifier(Dict()) 
  @test_throws ErrorException fit!(tscl)
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

function test_realdatatsclassifier()
  tscl=TSClassifier(Dict()) 
  @test_throws ErrorException fit!(tscl)
  trdirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/training")
  tstdirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/testing")
  modeldirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/model")

  tscl = TSClassifier(Dict(:trdirectory=>trdirname,
			   :tstdirectory=>tstdirname,
			   :modeldirectory=>modeldirname,
			   :num_trees=>30))
  modelfname = joinpath(tscl.args[:modeldirectory],tscl.args[:juliarfmodelname])
  fit!(tscl)
  
  dfresults = transform!(tscl)

  apredict = dfresults[:predtype]
  fnames = dfresults[:fname]

  myregex = r"(?<dtype>[A-Z _ - a-z]+)(?<number>\d*).(?<ext>\w+)"
  mtypes=map(fnames) do fname
    mymatch=match(myregex,fname)
    mymatch[:dtype]
  end

  # misclassified one data
  sum(mtypes .== apredict) == length(mtypes) - 1

end
@testset "TSClassifier" begin
  test_realdatatsclassifier()
end



end
