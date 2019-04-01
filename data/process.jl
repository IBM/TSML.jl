using Pkg
Pkg.activate("..")
using TSML
using TSML.TSMLTransformers
using DataFrames
using Dates
using CSV
using Plots

fname ="testdata.csv"
dat = CSV.read(fname)
rename!(dat,names(dat)[1]=>:Date,names(dat)[2]=>:Value)
dat[:Date] = DateTime.(dat[:Date],"d/m/y H:M")
orig = deepcopy(dat)
filter1 = DateValgator()
filter2 = DateValNNer(Dict(:nnsize=>1))

fit!(filter1,dat,[])
res1=transform!(filter1,dat)

fit!(filter2,res1,[])
res2=transform!(filter2,res1)

mypipeline = Pipeline(Dict(
		:transformers => [csvreader,filter1,filter2]
	)
)

fit!(mypipeline)
res = transform!(mypipeline)

Plots.plot(res[:Value][end-3000:end])

rfname = replace(fname,".csv"=>"-result.csv")
res |> CSV.write(rfname)

using TSML.TSMLTypes
import TSML.TSMLTypes.fit!
import TSML.TSMLTypes.transform!

mutable struct CSVDateValReader <: Transformer
	model
	args
	function CSVDateValReader(args=Dict())
		default_args = Dict(
			:filename => "",
			:dateformat => ""
		)
		new(nothing,mergedict(default_args,args))
	end
end

function fit!(csvrdr::CSVDateValReader,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}
	fname = csvrdr.args[:filename]
	fmt = csvrdr.args[:dateformat]
	(fname != "" && fmt != "") || error("missing filename or date format")
	model = csvrdr.args
end

function transform!(csvrdr::CSVDateValReader,x::T=[]) where {T<:Union{DataFrame,Vector,Matrix}}
	fname = csvrdr.args[:filename]
	fmt = csvrdr.args[:dateformat]
	df = CSV.read(fname)
	ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
	rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
	df[:Date] = DateTime.(df[:Date],fmt)
	df
end

csvreader = CSVDateValReader(Dict(:filename=>"testdata.csv",:dateformat=>"d/m/y H:M"))

fit!(csvreader)
transform!(csvreader)
