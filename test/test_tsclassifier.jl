module TestTSClassifier

using Test
using TSML

function test_realdatatsclassifier()
  tscl=TSClassifier(Dict()) 
  @test_throws ErrorException fit!(tscl)
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

  apredict = dfresults.predtype
  fnames = dfresults.fname
  myregex = r"(?<dtype>[A-Z _ - a-z]+)(?<number>\d*).(?<ext>\w+)"
  mtypes=map(fnames) do fname
    mymatch=match(myregex,fname)
    mymatch[:dtype]
  end
  # misclassified one data
  sum(mtypes .== apredict) == length(mtypes) - 2 
end
@testset "TSClassifier" begin
  test_realdatatsclassifier()
end

end
