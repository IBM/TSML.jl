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

const XX = IRIS[:,1:1] |> Matrix
const YY = IRIS[:,4] |> Vector

const classifiers = [
    "LinearSVC","QDA","MLPClassifier","BernoulliNB",
    "RandomForestClassifier","LDA",
    "NearestCentroid","SVC","LinearSVC","NuSVC","MLPClassifier",
    "RidgeClassifierCV","SGDClassifier","KNeighborsClassifier",
    "GaussianProcessClassifier","DecisionTreeClassifier",
    "PassiveAggressiveClassifier","RidgeClassifier",
    "ExtraTreesClassifier","GradientBoostingClassifier",
    "BaggingClassifier","AdaBoostClassifier","GaussianNB","MultinomialNB",
    "ComplementNB","BernoulliNB"
 ]

const regressors = [
    "SVR",
    "Ridge",
    "RidgeCV",
    "Lasso",
    "ElasticNet",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "BayesianRidge",
    "ARDRegression",
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "KernelRidge",
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",
    "GaussianProcessRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "MLPRegressor",
    "AdaBoostRegressor"
#    "IsotonicRegression"
]
    	

function fit_test(learner::String,in::T,out::Vector) where {T<:Union{Matrix,Vector}}
	_learner=SKLearner(Dict(:learner=>learner))
	fit!(_learner,in,out)
	println(learner)
	@test _learner.model != nothing
end

@testset "scikit classifiers" begin
    for cls in classifiers
	fit_test(cls,X,Y)
    end
end

@testset "scikit regressors" begin
    for rg in regressors
	fit_test(rg,XX,YY)
    end
end



end
