# Decision trees as found in DecisionTree Julia package.
module DecisionTreeLearners

using DataFrames
using TSML.TSMLTypes
import TSML.TSMLTypes.fit!
import TSML.TSMLTypes.transform!
using TSML.Utils

using Random
import DecisionTree
DT = DecisionTree

export PrunedTree, 
       RandomForest,
       Adaboost,
       fit!, 
       transform!

# Pruned CART decision tree.
mutable struct PrunedTree <: TSLearner
  model
  args
  function PrunedTree(args=Dict())
    default_args = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_args => Dict(
        # Merge leaves having >= purity_threshold CombineMLd purity.
        :purity_threshold => 1.0,
        # Maximum depth of the decision tree (default: no maximum).
        :max_depth => -1,
        # Minimum number of samples each leaf needs to have.
        :min_samples_leaf => 1,
        # Minimum number of samples in needed for a split.
        :min_samples_split => 2,
        # Minimum purity needed for a split.
        :min_purity_increase => 0.0
      )
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(tree::PrunedTree, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  instances = features
  if typeof(features) <: DataFrame
    instances=convert(Matrix,features)
  end
  impl_args = tree.args[:impl_args]
  tree.model = DT.build_tree(
    labels,
    instances,
    0, # num_subfeatures (keep all)
    impl_args[:max_depth],
    impl_args[:min_samples_leaf],
    impl_args[:min_samples_split],
    impl_args[:min_purity_increase])
  tree.model = DT.prune_tree(tree.model, impl_args[:purity_threshold])
end

function transform!(tree::PrunedTree, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  instances = features
  if typeof(features) <: DataFrame
    instances=convert(Matrix,features)
  end
  return DT.apply_tree(tree.model, instances)
end

function ptreerun()
  Random.seed!(125)
  data = getiris()
  features = data[:,1:4]
  sp = data[:Species] |> Vector
  pt = PrunedTree()
  fit!(pt,features,sp)
  res=transform!(pt,features)
  sum(sp .== res)/length(sp)
end
ptreerun()


# Random forest (CART).
mutable struct RandomForest <: TSLearner
  model
  args
  function RandomForest(args=Dict())
    default_args = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_args => Dict(
        # Number of features to train on with trees (default: 0, keep all).
        :num_subfeatures => 0,
        # Number of trees in forest.
        :num_trees => 10,
        # Proportion of trainingset to be used for trees.
        :partial_sampling => 0.7,
        # Maximum depth of each decision tree (default: no maximum).
        :max_depth => -1
      )
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(forest::RandomForest, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  instances = features
  if typeof(features) <: DataFrame
    instances=convert(Matrix,features)
  end
  # Set training-dependent options
  impl_args = forest.args[:impl_args]
  # Build model
  forest.model = DT.build_forest(
    labels, 
    instances,
    impl_args[:num_subfeatures],
    impl_args[:num_trees],
    impl_args[:partial_sampling],
    impl_args[:max_depth]
  )
end

function transform!(forest::RandomForest, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  instances = features
  if typeof(features) <: DataFrame
    instances=convert(Matrix,features)
  end
  return DT.apply_forest(forest.model, instances)
end

function rfrun()
  Random.seed!(123)
  data = getiris()
  features = data[:,1:4]
  sp = data[:Species] |> Vector
  rf = RandomForest()
  fit!(rf,features,sp)
  res=transform!(rf,features)
  sum(sp .== res)/length(sp)
end
rfrun()


# Adaboosted decision stumps.
mutable struct Adaboost <: TSLearner
  model
  args
  function Adaboost(args=Dict())
    default_args = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_args => Dict(
        # Number of boosting iterations.
        :num_iterations => 7
      )
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(adaboost::Adaboost, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  instance = features
  if typeof(features) <: DataFrame
    instances = convert(Matrix,features)
  end
  # NOTE(svs14): Variable 'model' renamed to 'ensemble'.
  #              This differs to DecisionTree
  #              official documentation to avoid confusion in variable
  #              naming within CombineML.
  ensemble, coefficients = DT.build_adaboost_stumps(
    labels, instances, adaboost.args[:impl_args][:num_iterations]
  )
  adaboost.model = Dict(
    :ensemble => ensemble,
    :coefficients => coefficients
  )
end

function transform!(adaboost::Adaboost, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  instance = features
  if typeof(features) <: DataFrame
    instances = convert(Matrix,features)
  end
  return DT.apply_adaboost_stumps(
    adaboost.model[:ensemble], adaboost.model[:coefficients], instances
  )
end

function adarun()
  Random.seed!(123)
  data = getiris()
  features = data[:,1:4]
  sp = data[:Species] |> Vector
  ada = Adaboost()
  fit!(ada,features,sp)
  res=transform!(ada,features)
  sum(sp .== res)/length(sp)
end
adarun()


end # module
