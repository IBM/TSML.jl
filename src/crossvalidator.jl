@reexport module CrossValidators

using TSML: TSMLTypes, Utils, Pipeline,fit!,transform!, kfold
using Statistics: mean, std
using DataFrames

export crossvalidate

function crossvalidate(pl::Pipeline,data::DataFrame,
		       cfndx::Array,
		       ctfndx::Array,
		       classndx::Integer,
		       pfunc::Function,nfolds=10) 
  ## flatten arrays
  ppl = deepcopy(pl)
  contfndx = cfndx|> Iterators.flatten |> collect
  catfndx = ctfndx|> Iterators.flatten |> collect
  featuresndx = merge(contfndx,catfndx)
  input = data[:,featuresndx]
  output = data[:,classndx]
  rowsize = size(data)[1]
  folds = kfold(rowsize,nfolds)
  pacc = Float64[]
  for (fold,trainndx) in enumerate(folds)
    testndx = setdiff(1:rowsize,trainndx)
    trX = input[trainndx,:] |> Matrix
    trY = output[trainndx] |> collect
    tstX = input[testndx,:] |> Matrix
    tstY = output[testndx] |> collect
    res = pipe_accuracy(ppl,pfunc,trX,trY,tstX,tstY)
    push!(pacc,res)
    println("fold: ",fold,", ",res)
  end
  (mean=mean(pacc),std=std(pacc),folds=nfolds)
end

function pipe_accuracy(plearner,perf,trainX,trainY,testX,testY)
  learner = deepcopy(plearner)
  fit!(learner,trainX,trainY)
  pred = transform!(learner,testX)
  perf(pred,testY)
end


end
