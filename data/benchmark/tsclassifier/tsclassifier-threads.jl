using Base.Threads
nthreads() |> println

using TSML
using DataFrames
using Random
using Statistics
using StatsBase: iqr
using RDatasets


# Julia ML
jrf = RandomForest(Dict(:impl_args=>Dict(:num_trees=>500)))
jpt = PrunedTree()
jada = Adaboost(Dict(:impl_args=>Dict(:num_iterations=>20)))

## Julia Ensembles
jvote_ens=VoteEnsemble(Dict(:learners=>[jrf,jpt,jada]))
jstack_ens=StackEnsemble(Dict(:learners=>[jrf,jpt,jada]))
jbest_ens=BestLearner(Dict(:learners=>[jrf,jpt,jada]))
jsuper_ens=VoteEnsemble(Dict(:learners=>[jvote_ens,jstack_ens,jbest_ens]))


function predict(learner,data,train_ind,test_ind)
    features = convert(Matrix,data[:, 1:(end-1)])
    labels = convert(Array,data[:, end])
    # Create pipeline
    pipeline = Pipeline(
       Dict(
         :transformers => [
           OneHotEncoder(), # Encodes nominal features into numeric
           Imputer(), # Imputes NA values
           StandardScaler(),
           learner # Predicts labels on instances
         ]
       )
    )
    # Train
    fit!(pipeline, features[train_ind, :], labels[train_ind]);
    # Predict
    predictions = transform!(pipeline, features[test_ind, :]);
    # Assess predictions
    result = score(:accuracy, labels[test_ind], predictions)
    return result
end

function extract_features_from_timeseries(datadir)
  println("*** Extracting features ***")
  mdata = getstats(datadir)
  mdata[!,:dtype] = mdata[!,:dtype] |> Array{String}
  return mdata[!,3:(end-1)]
end

function threadedmodel(learners::Dict,data::DataFrame;trials=5)
    Random.seed!(3)
    models=collect(keys(learners))
    global ctable = DataFrame()
    @threads for i=1:trials
        # Split into training and test sets
        (train_ind, test_ind) = holdout(size(data, 1), 0.20)
        mtx = SpinLock()
        @threads for themodel in models
            res=predict(learners[themodel],data,train_ind,test_ind)
            println(themodel," => ",round(res),", thread=",threadid())
            lock(mtx)
            global ctable=vcat(ctable,DataFrame(model=themodel, acc=res))
            unlock(mtx)
        end
    end
    df = ctable |> DataFrame
    gp=by(df,:model) do x
       DataFrame(mean=mean(x.acc),std=std(x.acc),n=nrow(x))
    end
    sort!(gp,:mean,rev=true)
    return gp
end

datadir = joinpath("tsdata/")
tsdata = extract_features_from_timeseries(datadir)
first(tsdata,5)

learners=Dict(
      :jvote_ens=>jvote_ens,:jstack_ens=>jstack_ens,:jbest_ens=>jbest_ens,
      :jrf => jrf,:jada=>jada,:jsuper_ens=>jsuper_ens);
dfres = threadedmodel(learners,tsdata;trials=10)
