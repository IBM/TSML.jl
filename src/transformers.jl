module TSMLTransformers

using MLDataUtils
using Dates
using DataFrames
using Statistics
using Random

export fit!,transform!

export Transformer,TSLearner
export Imputer,Pipeline,SKLLearner,OneHotEncoder,Pipeline,Wrapper

export Matrifier,Dateifier
export DateValizer,DateValgator,DateValNNer

export matrifyrun,dateifierrun,
       datevalgatorrun,datevalizerrun,
       datevalnnerrun

using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils
using DataFrames

# Transforms instances with nominal features into one-hot form
# and coerces the instance matrix to be of element type Float64.
mutable struct OneHotEncoder <: Transformer
  model
  args

  function OneHotEncoder(args=Dict())
    default_args = Dict(
      # Nominal columns
      :nominal_columns => nothing,
      # Nominal column values map. Key is column index, value is list of
      # possible values for that column.
      :nominal_column_values_map => nothing
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(ohe::OneHotEncoder, features::T, labels::Vector) where {T<:Union{Matrix,DataFrame}}
  instances=convert(Matrix,features)
  # Obtain nominal columns
  nominal_columns = ohe.args[:nominal_columns]
  if nominal_columns == nothing
    nominal_columns = find_nominal_columns(instances)
  end

  # Obtain unique values for each nominal column
  nominal_column_values_map = ohe.args[:nominal_column_values_map]
  if nominal_column_values_map == nothing
    nominal_column_values_map = Dict{Int, Any}()
    for column in nominal_columns
      nominal_column_values_map[column] = unique(instances[:, column])
    end
  end

  # Create model
  ohe.model = Dict(
    :nominal_columns => nominal_columns,
    :nominal_column_values_map => nominal_column_values_map
  )
end

function transform!(ohe::OneHotEncoder, features::T) where {T<:Union{Matrix,DataFrame}}
  instances=convert(Matrix,features)
  nominal_columns = ohe.model[:nominal_columns]
  nominal_column_values_map = ohe.model[:nominal_column_values_map]

  # Create new transformed instance matrix of type Float64
  num_rows = size(instances, 1)
  num_columns = (size(instances, 2) - length(nominal_columns))
  if !isempty(nominal_column_values_map)
    num_columns += sum(map(x -> length(x), values(nominal_column_values_map)))
  end
  transformed_instances = zeros(Float64, num_rows, num_columns)

  # Fill transformed instance matrix
  col_start_index = 1
  for column in 1:size(instances, 2)
    if !in(column, nominal_columns)
      transformed_instances[:, col_start_index] = instances[:, column]
      col_start_index += 1
    else
      col_values = nominal_column_values_map[column]
      for row in 1:size(instances, 1)
        entry_value = instances[row, column]
        entry_value_index = findfirst(isequal(entry_value),col_values)
        if entry_value_index == 0
          warn("Unseen value found in OneHotEncoder,
                for entry ($row, $column) = $(entry_value).
                Patching value to $(col_values[1]).")
          entry_value_index = 1
        end
        entry_column = (col_start_index - 1) + entry_value_index
        transformed_instances[row, entry_column] = 1
      end
      col_start_index += length(nominal_column_values_map[column])
    end
  end

  return transformed_instances
end

# Finds all nominal columns.
#
# Nominal columns are those that do not have Real type nor
# do all their elements correspond to Real.
function find_nominal_columns(features::T) where {T<:Union{Matrix,DataFrame}}
  instances=convert(Matrix,features)
  nominal_columns = Int[]
  for column in 1:size(instances, 2)
    col_eltype = infer_eltype(instances[:, column])
    if !<:(col_eltype, Real)
      push!(nominal_columns, column)
    end
  end
  return nominal_columns
end

# Convert a 1-D timeseries into sliding window matrix for ML training
mutable struct Matrifier <: Transformer
  model
  args

  function Matrifier(args=Dict())
    default_args = Dict(
        :ahead => 1,
        :size => 7,
        :stride => 1,
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(mtr::Matrifier,x::T,y::Vector=Vector()) where {T<:Union{Matrix,Vector}}
  mtr.model = mtr.args
end

function transform!(mtr::Matrifier,x::T) where {T<:Union{Matrix,Vector}}
  x isa Vector || error("data should be a vector")
  res=toMatrix(mtr,x)
  convert(Array{Float64},res)
end

function toMatrix(mtr::Transformer, x::Vector)
  stride=mtr.args[:stride];sz=mtr.args[:size];ahead=mtr.args[:ahead]
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
  return mmatrix
end


function matrifyrun()
  mtr = Matrifier(Dict(:ahead=>24,:size=>24,:stride=>1))
  sz = mtr.args[:size]
  x=collect(1:100)
  y=collect(1:100)
  println(fit!(mtr,x,y))
  transform!(mtr,x)
end

# Convert a 1-D date series into sliding window matrix for ML training
mutable struct Dateifier <: Transformer
  model
  args

  function Dateifier(args=Dict())
    default_args = Dict(
        :ahead => 1,
        :size => 7,
        :stride => 1,
        :dateinterval => Dates.Hour(1)
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(dtr::Dateifier,x::T,y::Vector=[]) where {T<:Union{Matrix,Vector}}
  (eltype(x) <: DateTime || eltype(x) <: Date) || error("array element types are not dates")
  dtr.args[:lower] = minimum(x)
  dtr.args[:upper] = maximum(x)
  dtr.model = dtr.args
end

# transform to day of the month, day of the week, etc
function transform!(dtr::Dateifier,x::T) where {T<:Union{Matrix,Vector}}
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
  convert(Matrix{Int64},dt)
end

function dateifierrun()
  dtr = Dateifier(Dict(:stride=>5))
  lower = DateTime(2017,1,1)
  upper = DateTime(2019,1,1)
  x=lower:Dates.Hour(1):upper |> collect
  y=lower:Dates.Hour(1):upper |> collect
  fit!(dtr,x,y)
  transform!(dtr,x)
end


# Date,Val time series
mutable struct DateValgator <: Transformer
  model
  args

  function DateValgator(args=Dict())
    default_args = Dict(
        :ahead => 1,
        :size => 7,
        :stride => 1,
        :dateinterval => Dates.Hour(1)
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(dvmr::DateValgator,x::T,y::Vector=[]) where {T<:DataFrame}
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Real || error("array element types are not dates")
  dvmr.model=dvmr.args
end

function transform!(dvmr::DateValgator,x::T) where {T<:DataFrame}
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Real || error("array element types are not dates")
  cnames = names(x)
  rename!(x,Dict(cnames[1]=>:Date,cnames[2]=>:Value))
  grpby = typeof(dvmr.args[:dateinterval])
  sym = Symbol(grpby)
  x[sym] = round.(x[:Date],grpby)
  res=by(x,sym,MeanValue = :Value=>skipmean)
  rename!(res,Dict(names(res)[1]=>:Date,names(res)[2]=>:Value))
end

function datevalgatorrun()
  dtvl = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  dte=DateTime(2014,1,1):Dates.Minute(1):DateTime(2016,1,1)
  val = rand(length(dte))
  fit!(dtvl,DataFrame(date=dte,values=val),[])
  transform!(dtvl,DataFrame(date=dte,values=val))
end


# Date,Val time series
# Normalize and clean date,val by replacing missings with medians
mutable struct DateValizer <: Transformer
  model
  args

  function DateValizer(args=Dict())
    default_args = Dict(
        :ahead => 1,
        :size => 7,
        :stride => 1,
        :dateinterval => Dates.Hour(1)
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(dvzr::DateValizer,x::T,y::Vector=[]) where {T<:DataFrame}
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Union{Missing,Real} || error("array element types are not values")
  dvzr.model=dvzr.args
end

function transform!(dvzr::DateValizer,x::T) where {T<:DataFrame}
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Union{Real,Missing} || error("array element types are not values")
  cnames = names(x)
  rename!(x,Dict(cnames[1]=>:Date,cnames[2]=>:Value))
  grpby = typeof(dvzr.args[:dateinterval])
  sym = Symbol(grpby)
  x[sym] = round.(x[:Date],grpby)
  aggr = by(x,sym,MeanValue = :Value=>skipmean)
  rename!(aggr,Dict(names(aggr)[1]=>:Date,names(aggr)[2]=>:Value))
  lower = minimum(x[:Date])
  upper = maximum(x[:Date])
  cdate = DataFrame(Date = collect(lower:dvzr.args[:dateinterval]:upper))
  joined = join(cdate,aggr,on=:Date,kind=:left)
  medians = getMedian(grpby,joined)
  missingndx = findall(ismissing.(joined[:Value]))
  jmndx=joined[missingndx,sym] .+ 1 # get time period index of missing, convert 0 index time to 1 index
  joined[missingndx,:Value] = medians[jmndx,:Value] # replace missing with median value
  sum(ismissing.(joined[:,:Value])) == 0 || error("Aggregation by time period failed to replace missings")
  joined[:,[:Date,:Value]]
end

function getMedian(t::Type{T},x::DataFrame) where {T<:Union{TimePeriod,DatePeriod}}
  sgp = Symbol(t)
  fn = Dict(:Hour=>Dates.hour,:Minute=>Dates.minute,
            :Second=>Dates.second,:Month=>Dates.month)
  try
    x[sgp]=fn[sgp].(x[:Date])
  catch
    error("unknown dateinterval")
  end
  gpmeans = by(x,sgp,Value = :Value => skipmedian)
end

function datevalizerrun()
  # test passing args from one structure to another
  Random.seed!(123)
  dvzr1 = DateValizer(Dict(:dateinterval=>Dates.Hour(1)))
  dvzr2 = DateValizer(dvzr1.args)
  dte=DateTime(2014,1,1):Dates.Hour(1):DateTime(2016,1,1)
  val = Array{Union{Missing,Float64}}(rand(length(dte)))
  y = []
  x = DataFrame(MDate=dte,MValue=val)
  nmissing=10
  ndxmissing=Random.shuffle(1:length(dte))[1:nmissing]
  x[:MValue][ndxmissing] .= missing
  fit!(dvzr2,x,y)
  transform!(dvzr2,x)
end


# fill-in missings with nearest-neighbors median
mutable struct DateValNNer <: Transformer
  model
  args

  function DateValNNer(args=Dict())
    default_args = Dict(
        :ahead => 1,
        :size => 7,
        :stride => 1,
        :dateinterval => Dates.Hour(1),
        :nnsize => 5 
    )
    new(nothing,mergedict(default_args,args))
  end
end

function fit!(dnnr::DateValNNer,x::T,y::Vector=[]) where {T<:DataFrame}
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Union{Missing,Real} || error("array element types are not values")
  dnnr.model=dnnr.args
end

function transform!(dnnr::DateValNNer,x::T) where {T<:DataFrame}
  size(x)[2] == 2 || error("Date Val timeseries need two columns")
  (eltype(x[:,1]) <: DateTime || eltype(x[:,1]) <: Date) || error("array element types are not dates")
  eltype(x[:,2]) <: Union{Real,Missing} || error("array element types are not values")
  cnames = names(x)
  rename!(x,Dict(cnames[1]=>:Date,cnames[2]=>:Value))
  grpby = typeof(dnnr.args[:dateinterval])
  sym = Symbol(grpby)
  # to fill-in with nearest neighbors
  nnsize = dnnr.args[:nnsize]
  missingndx = DataFrame(missed = findall(ismissing.(x[:Value])))
  missingndx[:neighbors] = (x->(x-nnsize):(x-1)).(missingndx[:missed]) # NN ranges
  x[missingndx[:missed],:Value] = (r -> skipmedian(x[r,:Value])).(missingndx[:neighbors]) # iterate to each range
  sum(ismissing.(x[:,:Value])) == 0 || error("Nearest Neigbour algo failed to replace missings")
  x
end


function datevalnnerrun()
  # test passing args from one structure to another
  Random.seed!(123)
  dnnr = DateValNNer(Dict(:dateinterval=>Dates.Hour(1),:nnsize=>3))
  dte=DateTime(2014,1,1):Dates.Hour(1):DateTime(2016,1,1)
  val = Array{Union{Missing,Float64}}(rand(length(dte)))
  y = []
  x = DataFrame(MDate=dte,MValue=val)
  nmissing=10
  ndxmissing=Random.shuffle(1:length(dte))[1:nmissing]
  x[:MValue][ndxmissing] .= missing
  fit!(dnnr,x,y)
  transform!(dnnr,x)
end




# Imputes NaN values from Float64 features.
mutable struct Imputer <: Transformer
  model
  args

  function Imputer(args=Dict())
    default_args = Dict(
      # Imputation strategy.
      # Statistic that takes a vector such as mean or median.
      :strategy => mean
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(imp::Imputer, instances::T, labels::Vector) where {T<:Union{Matrix,DataFrame}}
  imp.model = imp.args
end

function transform!(imp::Imputer, features::T)  where {T<:Union{Matrix,DataFrame}}
  instances=convert(Matrix,features)
  new_instances = copy(instances)
  strategy = imp.model[:strategy]

  for column in 1:size(instances, 2)
    column_values = instances[:, column]
    col_eltype = infer_eltype(column_values)

    if <:(col_eltype, Real)
      na_rows = map(x -> isnan(x), column_values)
      if any(na_rows)
        fill_value = strategy(column_values[.!na_rows])
        new_instances[na_rows, column] .= fill_value
      end
    end
  end

  return new_instances
end


# Chains multiple transformers in sequence.
mutable struct Pipeline <: Transformer
  model
  args

  function Pipeline(args=Dict())
    default_args = Dict(
      # Transformers as list to chain in sequence.
      :transformers => [OneHotEncoder(), Imputer()],
      # Transformer args as list applied to same index transformer.
      :transformer_args => nothing
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(pipe::Pipeline, features::T, labels::Vector) where {T<:Union{Matrix,DataFrame}}
  instances=convert(Matrix,features)
  transformers = pipe.args[:transformers]
  transformer_args = pipe.args[:transformer_args]

  current_instances = instances
  new_transformers = Transformer[]
  for t_index in 1:length(transformers)
    transformer = create_transformer(transformers[t_index], transformer_args)
    push!(new_transformers, transformer)
    fit!(transformer, current_instances, labels)
    current_instances = transform!(transformer, current_instances)
  end

  pipe.model = Dict(
      :transformers => new_transformers,
      :transformer_args => transformer_args
  )
end

function transform!(pipe::Pipeline, features::T) where {T<:Union{Matrix,DataFrame}}
  instances = convert(Matrix,features)
  transformers = pipe.model[:transformers]

  current_instances = instances
  for t_index in 1:length(transformers)
    transformer = transformers[t_index]
    current_instances = transform!(transformer, current_instances)
  end

  return current_instances
end


# Wraps around an CombineML transformer.
mutable struct Wrapper <: Transformer
  model
  args

  function Wrapper(args=Dict())
    default_args = Dict(
      # Transformer to call.
      :transformer => OneHotEncoder(),
      # Transformer args.
      :transformer_args => nothing
    )
    new(nothing, mergedict(default_args, args))
  end
end

function fit!(wrapper::Wrapper, features::T, labels::Vector) where {T<:Union{Matrix,DataFrame}}
  instances=convert(Matrix,features)
  transformer_args = wrapper.args[:transformer_args]
  transformer = create_transformer(
    wrapper.args[:transformer],
    transformer_args
  )

  if transformer_args != nothing
    transformer_args = mergedict(transformer.args, transformer_args)
  end
  fit!(transformer, instances, labels)

  wrapper.model = Dict(
    :transformer => transformer,
    :transformer_args => transformer_args
  )
end

function transform!(wrapper::Wrapper, instances::T) where {T<:Union{Matrix,DataFrame}}
  transformer = wrapper.model[:transformer]
  return transform!(transformer, instances)
end

end
