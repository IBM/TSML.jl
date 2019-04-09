module CaretLearners

export CaretLearner,fit!,transform!
export caretrun

using TSML.TSMLTypes
using TSML.Utils
using DataFrames

import TSML.TSMLTypes.fit!   # importing to overload
import TSML.TSMLTypes.transform! # importing to overload

using RCall
R"library(caret)"
R"library(e1071)"
R"library(gam)"
R"library(randomForest)"
R"library(nnet)" 
R"library(kernlab)"
R"library(grid)"
R"library(MASS)"
R"library(pls)"
#R"install.packages('xgboost',repos='https://cloud.r-project.org')"
#R"install.packages('pls',repos='https://cloud.r-project.org')"
#R"install.packages('caret', dependencies = c('Depends', 'Suggests'))"
R"library(xgboost)"

mutable struct CaretLearner <: TSLearner
    model
    args

    function CaretLearner(args=Dict())
        #fitControl=:(R"trainControl(method = 'repeatedcv',number = 5,repeats = 5)")
        fitControl="trainControl(method = 'cv',number = 5,repeats = 5)"
        default_args = Dict(
            :output => :class,
            :learner => "rf",
            :fitControl=>fitControl,
            :impl_args => Dict()
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(crt::CaretLearner,x::T,y::Vector) where  {T<:Union{Vector,Matrix}}
    xx = x |> DataFrame
    yy = y |> Vector
    rres = rcall(:train,xx,yy,method=crt.args[:learner],trControl = reval(crt.args[:fitControl]))
    #crt.model = R"$rres$finalModel"
    crt.model = rres
end

function transform!(crt::CaretLearner,x::T) where  {T<:Union{Vector,Matrix}}
    xx = x |> DataFrame
    res = rcall(:predict,crt.model,xx) #in robj
    return rcopy(res) # return extracted robj
end


function caretrun()
    crt = CaretLearner(Dict(:learner=>"rf",:fitControl=>"trainControl(method='cv')")) 
    iris=getiris()
    x=iris[:,1:4]  |> Matrix
    y=iris[:,5] |> Vector
    fit!(crt,x,y)
    print(crt.model)
    transform!(crt,x)
end

end
