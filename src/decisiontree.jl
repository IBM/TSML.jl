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

"""
    PrunedTree(
      Dict(
        :purity_threshold => 1.0,
        :max_depth => -1,
        :min_samples_leaf => 1,
        :min_samples_split => 2,
        :min_purity_increase => 0.0
      )
    )

Decision tree classifier.  
See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparmeters:
- `:purity_threshold` => 1.0 (merge leaves having >=thresh combined purity)
- `:max_depth` => -1 (maximum depth of the decision tree)
- `:min_samples_leaf` => 1 (the minimum number of samples each leaf needs to have)
- `:min_samples_split` => 2 (the minimum number of samples in needed for a split)
- `:min_purity_increase` => 0.0 (minimum purity needed for a split)

Implements `fit!`, `transform!`
"""
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

"""
    fit!(tree::PrunedTree, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}

Function to optimize the hyperparameters of `PrunedTree` instance.
"""
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


"""
    transform!(tree::PrundTree, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Function to predict using the optimized hyperparameters of the trained `PrunedTree` instance.
"""
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

"""
    RandomForest(
      Dict(
        :output => :class,
        :num_subfeatures => 0,
        :num_trees => 10,
        :partial_sampling => 0.7,
        :max_depth => -1
      )
    )

Random forest classification. 
See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparmeters:
- `:num_subfeatures` => 0  (number of features to consider at random per split)
- `:num_trees` => 10 (number of trees to train)
- `:partial_sampling` => 0.7 (fraction of samples to train each tree on)
- `:max_depth` => -1 (maximum depth of the decision trees)
- `:min_samples_leaf` => 1 (the minimum number of samples each leaf needs to have)
- `:min_samples_split` => 2 (the minimum number of samples in needed for a split)
- `:min_purity_increase` => 0.0 (minimum purity needed for a split)

Implements `fit!`, `transform!`
"""
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


"""
    fit!(forest::RandomForest, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}

Function to optimize the parameters of the `RandomForest` instance.
"""
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


"""
    transform!(forest::RandomForest, features::T) where {T<:Union{Vector,Matrix,DataFrame}}


Function to predict using the optimized hyperparameters of the trained `RandomForest` instance.
"""
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

"""
    Adaboost(
      Dict(
        :output => :class,
        :num_iterations => 7
      )
    )

Adaboosted decision tree stumps. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:
- `:num_iterations` => 7 (number of iterations of AdaBoost)

Implements `fit!`, `transform!`
"""
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


"""
    fit!(adaboost::Adaboost, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}

Function to optimize the hyperparameters of `Adaboost` instance.
"""
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

"""
    transform!(adaboost::Adaboost, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Function to predict using the optimized hyperparameters of the trained `Adaboost` instance.
"""
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
