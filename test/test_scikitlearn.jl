module TestSKL

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.SKLearners

using Random
using Test
using PyCall
using RDatasets

const IRIS = dataset("datasets","iris")
const X = IRIS[:,1:4] |> Matrix
const Y = IRIS[:,5] |> Vector

const classifiers = [
	"LinearSVC","QDA","MLPClassifier","BernoulliNB",
	"RandomForestClassifier","LDA",
  	"NearestCentroid","SVC","LinearSVC","NuSVC","MLPClassifier",
  	"RidgeClassifierCV","SGDClassifier","KNeighborsClassifier",
  	"GaussianProcessClassifier","DecisionTreeClassifier",
    "PassiveAggressiveClassifier","RidgeClassifier",
	"ExtraTreesClassifier","GradientBoostingClassifier",
	"BaggingClassifier","AdaBoostClassifier"
 ]

function fit_test(learner::String)
	_learner=SKLearner(Dict(:learner=>learner))
	fit!(_learner,X,Y)
	println(learner)
	@test _learner.model != nothing
end

function loopover()
	for cls in classifiers
		fit_test(cls)
	end
end

@testset "scikit leaners" begin
	loopover()
end

end
