@reexport module CrossValidators

using TSML: TSMLTypes, Utils, Pipeline,fit!,transform!, kfold
using Statistics: mean, std
using DataFrames

export crossvalidate,crossvaldriver

function crossvalidate(ppl::Pipeline,mdata::DataFrame,
											 cfndx::Array,
											 ctfndx::Array,
											 classndx::Integer,
											 pfunc::Function,nfolds=10) 
	data = mdata |> Matrix
	## flatten arrays
	contfndx = cfndx|> Iterators.flatten |> collect
	catfndx = ctfndx|> Iterators.flatten |> collect
	featuresndx = merge(contfndx,catfndx)
  rowsize = size(data)[1]
  folds = kfold(rowsize,nfolds)
  acc = Float64[]
  for (fold,trainndx) in enumerate(folds)
    testndx = setdiff(1:rowsize,trainndx)
    trX = data[trainndx,featuresndx] |> Matrix
    trY = data[trainndx,classndx] |> collect
		tstX = data[testndx,featuresndx] |> Matrix
    tstY = data[testndx,classndx] |> collect
    res = pipe_accuracy(ppl,pfunc,trX,trY,tstX,tstY)
    push!(acc,res)
    println("fold: ",fold,", ",res)
  end
  (mean=mean(acc),std=std(acc),folds=nfolds)
end

function pipe_accuracy(learner,perf,trainX,trainY,testX,testY)
  fit!(learner,trainX,trainY)
  pred = transform!(learner,testX)
  perf(pred,testY)
end

using TSML: RandomForest, score, getiris
function crossvaldriver()
	acc(X,Y) = score(:accuracy,X,Y)
	data=getiris()
	ppl = Pipeline(Dict(:transformers=>[RandomForest()]))
	crossvalidate(ppl,data,[1:4],[],5,acc)
end


end
