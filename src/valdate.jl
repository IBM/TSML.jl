# Convert a 1-D timeseries into sliding window matrix for ML training
# using Plots

const gAggDict = Dict(
    :median => Statistics.median,
    :mean =>   Statistics.mean,
    :maximum => Statistics.maximum,
    :minimum => Statistics.minimum
)

mutable struct Matrifier <: Transformer
  model
  args

  function Matrifier(args=Dict())
    default_args = Dict{Symbol,Any}(
        :ahead => 1,
        :size => 7,
        :stride => 1,
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(mtr::Matrifier,xx::T,y::Vector=Vector()) where {T<:Union{Matrix,Vector,DataFrame}}
  typeof(xx) <: DataFrame || error("input is not a dataframe")
  x = deepcopy(xx[:Value])
  x isa Vector || error("data should be a vector")
  mtr.model = mtr.args
end

function transform!(mtr::Matrifier,xx::T) where {T<:Union{Matrix,Vector,DataFrame}}
  typeof(xx) <: DataFrame || error("input is not a dataframe")
  x = deepcopy(xx[:Value])
  x isa Vector || error("data should be a vector")
  mtype = eltype(x)
  res=toMatrix(mtr,x)
  resarray=convert(Array{mtype},res) |> DataFrame
  rename!(resarray,names(resarray)[end] => :output)
end

function toMatrix(mtr::Transformer, x::Vector)
  stride=mtr.args[:stride];sz=mtr.args[:size];ahead=mtr.args[:ahead]
  @assert stride>0 && sz>0 && ahead > 0
  xlength = length(x)
  xlength > sz || error("data too short for the given size of sliding window")
  ndx=collect(xlength:-1:1)
  mtuples = slidingwindow(i->(i-ahead),ndx,sz,stride)
  height=size(mtuples)[1]
  mmatrix = Array{Union{DateTime,<:Real},2}(zeros(height,sz+1))
  ctr=1
  gap = xlength - mtuples[1][2][1]
  for (s,k) in mtuples
    v = [reverse(s);k] .+ gap
    mmatrix[ctr,:].=x[v]
    ctr+=1
  end
  mmatrix
end

### ====

# Convert a 1-D date series into sliding window matrix for ML training
mutable struct Dateifier <: Transformer
  model
  args

  function Dateifier(args=Dict())
    default_args = Dict{Symbol,Any}(
        :ahead => 1,
        :size => 7,
        :stride => 1
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(dtr::Dateifier,xx::T,y::Vector=[]) where {T<:Union{Matrix,Vector,DataFrame}}
  typeof(xx) <: DataFrame || error("input not a dataframe")
  x = deepcopy(xx[:Date])
  (eltype(x) <: DateTime || eltype(x) <: Date) || error("array element types are not dates")
  dtr.args[:lower] = minimum(x)
  dtr.args[:upper] = maximum(x)
  dtr.model = dtr.args
end

# transform to day of the month, day of the week, etc
function transform!(dtr::Dateifier,xx::T) where {T<:Union{Matrix,Vector,DataFrame}}
  typeof(xx) <: DataFrame || error("input not a dataframe")
  x = deepcopy(xx[:Date])
  x isa Vector || error("data should be a vector")
  @assert eltype(x) <: DateTime || eltype(x) <: Date
  res=toMatrix(dtr,x)
  endpoints = convert(Array{DateTime},res)[:,end-1]
  dt = DataFrame()
  dt[:year]=Dates.year.(endpoints)
  dt[:month]=Dates.month.(endpoints)
  dt[:day]=Dates.day.(endpoints)
  dt[:hour]=Dates.hour.(endpoints)
  dt[:week]=Dates.week.(endpoints)
  dt[:dow]=Dates.dayofweek.(endpoints)
  dt[:doq]=Dates.dayofquarter.(endpoints)
  dt[:qoy]=Dates.quarterofyear.(endpoints)
  dtr.args[:header] = names(dt)
  #convert(Matrix{Int64},dt)
  return dt
end


### ====

# Date,Val time series
mutable struct DateValgator <: Transformer
  model
  args
  function DateValgator(args=Dict())
    default_args = Dict{Symbol,Any}(
        :dateinterval => Dates.Hour(1),
        :aggregator => :median
    )
    new(nothing,mergedict(default_args,args))
  end
end

function validdateval!(x::T) where {T<:DataFrame}
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Union{Missing,Real} || error("array element types are not values")
  cnames = names(x)
  rename!(x,Dict(cnames[1]=>:Date,cnames[2]=>:Value))
end


function fit!(dvmr::DateValgator,xx::T,y::Vector=[]) where {T<:Union{Matrix,DataFrame}}
  x = deepcopy(xx)
  validdateval!(x)
  aggr = dvmr.args[:aggregator] 
  aggr in keys(gAggDict) || error("aggregator function passed is not recognized: ",aggr)
  dvmr.model=dvmr.args
end

function transform!(dvmr::DateValgator,xx::T) where {T<:DataFrame}
  x = deepcopy(xx)
  validdateval!(x)
  # make sure aggregator function exists
  aggr = dvmr.args[:aggregator] 
  aggr in keys(gAggDict) || error("aggregator function passed is not recognized: ",aggr)
  # get the Statistics function
  aggfn = gAggDict[aggr]
  # pass the aggregator function to the generic aggregator function
  fn = aggregatorclskipmissing(aggfn)
  grpby = typeof(dvmr.args[:dateinterval])
  sym = Symbol(grpby)
  x[sym] = round.(x[:Date],grpby)
  aggr=by(x,sym,MeanValue = :Value=>fn)
  rename!(aggr,Dict(names(aggr)[1]=>:Date,names(aggr)[2]=>:Value))
  lower = round(minimum(x[:Date]),grpby)
  upper = round(maximum(x[:Date]),grpby)
  #create list of complete dates and join with aggregated data
  cdate = DataFrame(Date = collect(lower:dvmr.args[:dateinterval]:upper))
  joined = join(cdate,aggr,on=:Date,kind=:left)
  joined
end

### ====

# Date,Val time series
# Normalize and clean date,val by replacing missings with medians
mutable struct DateValizer <: Transformer
  model
  args

  function DateValizer(args=Dict())
    default_args = Dict{Symbol,Any}(
        :medians => DataFrame(),
        :dateinterval => Dates.Hour(1)
    )
    new(nothing,mergedict(default_args,args))
  end
end

function getMedian(t::Type{T},xx::DataFrame) where {T<:Union{TimePeriod,DatePeriod}}
  x = deepcopy(xx)
  sgp = Symbol(t)
  fn = Dict(Dates.Second=>Dates.second,
            Dates.Minute=>Dates.minute,
            Dates.Hour=>Dates.hour,
            Dates.Day=>Dates.day,
            Dates.Month=>Dates.month)
  try
    x[sgp]=fn[t].(x[:Date])
  catch
    error("unknown dateinterval")
  end
  gpmeans = by(x,sgp,Value = :Value => skipmedian)
  gpmeans
end

function fullaggregate!(dvzr::DateValizer,xx::T) where {T<:DataFrame}
  x = deepcopy(xx)
  grpby = typeof(dvzr.args[:dateinterval])
  sym = Symbol(grpby)
  x[sym] = round.(x[:Date],grpby)
  aggr = by(x,sym,MeanValue = :Value=>skipmedian)
  rename!(aggr,Dict(names(aggr)[1]=>:Date,names(aggr)[2]=>:Value))
  lower = minimum(x[:Date])
  upper = maximum(x[:Date])
  #create list of complete dates and join with aggregated data
  cdate = DataFrame(Date = collect(lower:dvzr.args[:dateinterval]:upper))
  joined = join(cdate,aggr,on=:Date,kind=:left)
  joined
end

function fit!(dvzr::DateValizer,xx::T,y::Vector=[]) where {T<:DataFrame}
  x = deepcopy(xx)
  validdateval!(x)
  # get complete dates and aggregate
  joined = fullaggregate!(dvzr,x)
  grpby = typeof(dvzr.args[:dateinterval])
  sym = Symbol(grpby)
  medians = getMedian(grpby,joined)
  dvzr.args[:medians] = medians
  dvzr.model=dvzr.args
end

function transform!(dvzr::DateValizer,xx::T) where {T<:DataFrame}
  x = deepcopy(xx)
  validdateval!(x)
  # get complete dates, aggregate, and get medians
  joined = fullaggregate!(dvzr,x)
  # copy medians
  medians = dvzr.args[:medians]
  grpby = typeof(dvzr.args[:dateinterval])
  sym = Symbol(grpby)
  fn = Dict(Dates.Hour=>Dates.hour,
            Dates.Minute=>Dates.minute,
            Dates.Second=>Dates.second,
            Dates.Day => Dates.day,
            Dates.Month=>Dates.month)
  try
    joined[sym]=fn[grpby].(joined[:Date])
  catch
    error("unknown dateinterval")
  end
  # find indices of missing
  missingndx = findall(ismissing.(joined[:Value]))
  jmndx=joined[missingndx,sym] .+ 1 # get time period index of missing, convert 0 index time to 1 index
  missingvals::SubArray = @view joined[missingndx,:Value]
  missingvals .= medians[jmndx,:Value] # replace missing with median value
  sum(ismissing.(joined[:,:Value])) == 0 || error("Aggregation by time period failed to replace missings")
  joined[:,[:Date,:Value]]
end

### ====

# fill-in missings with nearest-neighbors median
mutable struct DateValNNer <: Transformer
  model
  args

  function DateValNNer(args=Dict())
    default_args = Dict{Symbol,Any}(
        :missdirection => :symmetric, #:reverse, # or :forward or :symmetric
        :dateinterval => Dates.Hour(1),
        :nnsize => 1,
        :strict => true,
        :aggregator => :median
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(dnnr::DateValNNer,xx::T,y::Vector=[]) where {T<:DataFrame}
  x = deepcopy(xx)
  validdateval!(x)
  aggr = dnnr.args[:aggregator]
  aggr in keys(gAggDict) || error("aggregator function passed is not recognized: ",aggr)
  dnnr.model=dnnr.args
end

function transform!(dnnr::DateValNNer,xx::T) where {T<:DataFrame}
  x = deepcopy(xx)
  validdateval!(x)
  # make sure aggregator function exists
  aggr = dnnr.args[:aggregator]
  aggr in keys(gAggDict) || error("aggregator function pass is not recognized: ",aggr)
  # get the Statistics function
  aggfn = gAggDict[aggr]
  # pass the aggregator function to the generic aggregator function
  fn = aggregatorclskipmissing(aggfn)
  grpby = typeof(dnnr.args[:dateinterval])
  sym = Symbol(grpby)
  # aggregate by time period
  x[sym] = round.(x[:Date],grpby)
  aggr = by(x,sym,MeanValue = :Value=>fn)
  rename!(aggr,Dict(names(aggr)[1]=>:Date,names(aggr)[2]=>:Value))
  lower = round(minimum(x[:Date]),grpby)
  upper = round(maximum(x[:Date]),grpby)
  #create list of complete dates and join with aggregated data
  cdate = DataFrame(Date = collect(lower:dnnr.args[:dateinterval]:upper))
  joined = join(cdate,aggr,on=:Date,kind=:left)
  missingcount = sum(ismissing.(joined[:Value]))
  dnnr.args[:missingcount] = missingcount
  res = transform_worker!(dnnr,joined)
  count=1
  if dnnr.args[:missdirection] == :symmetric
    while sum(ismissing.(res[:Value])) > 0
      res = transform_worker!(dnnr,res)
      count += 1
    end
  end
  dnnr.args[:loopcount] = count
  res
end

function transform_worker!(dnnr::DateValNNer,joinc::T) where {T<:DataFrame}
  joined = deepcopy(joinc)
  maxrow = size(joined)[1]

  # to fill-in with nearest neighbors
  nnsize::Int64 = dnnr.args[:nnsize]
  themissing = findall(ismissing.(joined[:Value]))
  # ==== symmetric nearest neighbor
  missingndx = DataFrame()
  if dnnr.args[:missdirection] == :symmetric
    missed = themissing |> reverse
    missingndx[:Missed] = missed
    # get lower:upper range
    missingndx[:neighbors] = map(missingndx[:Missed]) do m
      lower = (m-nnsize >= 1) ? (m-nnsize) : 1
      upper = (m+nnsize <= maxrow) ? m+nnsize : maxrow
      lower:upper
    end
  else
    # ===== reverse and forward
    missed = (dnnr.args[:missdirection] == :reverse) ? (themissing |> reverse) : themissing
    missingndx[:Missed] = missed
    # dealing with boundary exceptions, default to range until the maxrow
    missingndx[:neighbors] = (m->((m+1>=maxrow) || (m+nnsize>=maxrow)) ? (m+1:maxrow) : (m+1:m+nnsize)).(missingndx[:Missed]) # NN ranges
  end
  #joined[missingndx[:Missed],:Value] = (r -> skipmedian(joined[r,:Value])).(missingndx[:neighbors]) # iterate to each range
  missingvals::SubArray = @view joined[missingndx[:Missed],:Value] # get view of only missings
  missingvals .=  (r -> skipmedian(joined[r,:Value])).(missingndx[:neighbors]) # replace with nn medians
  dnnr.args[:strict] && (sum(ismissing.(joined[:,:Value])) == 0 || error("Nearest Neigbour algo failed to replace missings"))
  joined
end

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
    df = CSV.read(fname) |> DataFrame
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    df[:Date] = DateTime.(df[:Date],fmt)
    df
end

mutable struct CSVDateValWriter <: Transformer
    model
    args
    function CSVDateValWriter(args=Dict())
        default_args = Dict(
            :filename => "",
            :dateformat => ""
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(csvwtr::CSVDateValWriter,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}
    fname = csvwtr.args[:filename]
    fmt = csvwtr.args[:dateformat]
    (fname != "" && fmt != "") || error("missing filename or date format")
    model = csvwtr.args
end

function transform!(csvwtr::CSVDateValWriter,x::T) where {T<:Union{DataFrame,Vector,Matrix}}
    fname = csvwtr.args[:filename]
    fmt = csvwtr.args[:dateformat]
    df = deepcopy(x) |> DataFrame
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    eltype(df[:Date]) <: DateTime || error("Date format error")
    df |> CSV.write(fname)
end
