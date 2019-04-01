module TestDecisionTree

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML.DecisionTreeLearners

using DataFrames
using Random
using Statistics
using Test
using Dates
using RDatasets


function generateXY()
    Random.seed!(123)
    iris = dataset("datasets","iris")
    indx = Random.shuffle(1:nrow(iris))
    features=iris[indx,1:4] 
    sp = iris[indx,5] |> Vector
    (features,sp)
end

const X,Y = generateXY()

function test_decisiontree()
    Random.seed!(123)
    learners = Dict(:rf=>RandomForest(),:ada=>Adaboost(),:ptree=>PrunedTree())
    results = Dict(:rf=>98.0,:ada=>95.0,:ptree=>100.0)
    for (name,obj) in learners
	fit!(obj,X,Y)
	res = transform!(obj,X)
	@testset "$name: Full dataset" begin
	    @test sum(res .== Y)/length(Y)*100 |> floor == results[name]
	end
    end
    trndx = 1:80
    tstndx = 81:nrow(X)
    results = Dict(:rf=>95.0,:ada=>92.0,:ptree=>95.0)
    for (name,obj) in learners
        fit!(obj,X[trndx,:],Y[trndx])
        res = transform!(obj,X[tstndx,:])
        @testset "$name: partial dataset" begin
            @test sum(res .== Y[tstndx])/length(Y[tstndx])*100 |> floor == results[name]
        end
    end
end
@testset "DecisionTrees" begin
    test_decisiontree()
end

end
