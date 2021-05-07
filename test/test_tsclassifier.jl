module TestTSClassifier

using Test
using TSML

function test_realdatatsclassifier()
  tscl=TSClassifier(Dict()) 
  @test_throws ArgumentError fit!(tscl)
  trdirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/training")
  tstdirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/testing")
  modeldirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/model")

  tscl = TSClassifier(Dict(:trdirectory=>trdirname,
			   :tstdirectory=>tstdirname,
			   :modeldirectory=>modeldirname,
			   :feature_range => 5:20,
			   :num_trees=>100))
  fit!(tscl)
  dfresults = transform!(tscl)
  m=fit(tscl)
  dfresults1 = transform(m)
  @test (dfresults .== dfresults1) |> Matrix |> sum > 0

  apredict = dfresults.predtype
  fnames = dfresults.fname
  myregex = r"(?<dtype>[A-Z _ - a-z]+)(?<number>\d*).(?<ext>\w+)"
  mtypes=map(fnames) do fname
    mymatch=match(myregex,fname)
    mymatch[:dtype]
  end
  # misclassified one data
  @test sum(mtypes .== apredict) > 0
end
@testset "TSClassifier" begin
  test_realdatatsclassifier()
end

end
