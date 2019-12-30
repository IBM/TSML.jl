@reexport module CrossValidators

using TSML.TSMLTypes
using TSML.Utils
using Statistics: mean, std
using DataFrames

export crossvalidate

function crossvalidate(pl::Transformer,X::Union{DataFrame,Matrix},Y::Vector,
		       pfunc::Function,nfolds=10) 
  ## flatten arrays
  @assert size(X)[1] == length(Y)
  ppl = deepcopy(pl)
  input = X |> Matrix
  output = Y
  rowsize = size(input)[1]
  folds = kfold(rowsize,nfolds)
  pacc = Float64[]
  for (fold,trainndx) in enumerate(folds)
    testndx = setdiff(1:rowsize,trainndx)
    trX = input[trainndx,:] 
    trY = output[trainndx] |> collect
    tstX = input[testndx,:]
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
