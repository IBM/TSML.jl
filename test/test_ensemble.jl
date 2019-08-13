module TestEnsembleMethods

using Test
using Random
using TSML

function getprediction(model::TSLearner,data::Dict)
  Random.seed!(126)
  trfeatures = data[:trfeatures]
  tstfeatures = data[:tstfeatures]
  troutput = data[:troutput]
  tstoutput = data[:tstoutput]
  fit!(model,trfeatures,troutput)
  trresults = transform!(model,tstfeatures)
  sum(trresults .== tstoutput)/length(tstoutput)
end

function test_ensembles()
  modeldirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/model")
  trdirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/training")
  tstdirname = joinpath(dirname(pathof(TSML)),"../data/realdatatsclassification/testing")

  frange = 5:20
  trdata = getstats(trdirname)
  trfeatures = trdata[:,frange]
  troutput = trdata[:,:dtype]

  tstdata = getstats(tstdirname)
  tstfeatures = tstdata[:,frange]
  tstoutput = tstdata[:,:dtype]

  models = [VoteEnsemble(),StackEnsemble(),BestLearner()]

  data = Dict(:trfeatures => trfeatures, 
              :tstfeatures => tstfeatures,
              :troutput => troutput,
              :tstoutput => tstoutput
             )
  #acc = [4//6,3//6,5//6]
  ndx=1
  for model in models
    @test getprediction(model,data) > 0.10
    ndx += 1
  end
end
@testset "Ensemble learners" begin
  test_ensembles()
end
  
end # module
