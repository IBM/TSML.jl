@reexport module FileStats

export ispathnotempty, getstats, getfilestat, TSType

using DataFrames
using Dates
using TSML

@enum TSType begin
  temperature = 1
  weather = 2
  footfall = 3
  AirOffTemp = 4
  Energy = 5
  Pressure = 6
  RetTemp = 7
end

function ispathnotempty(margs::Dict)
  return margs[:trdirectory] != "" && margs[:tstdirectory] != "" && margs[:modeldirectory] != ""
end

# return stat of a file
function getfilestat(ldirname::AbstractString,lfname::AbstractString)
  myregex = r"(?<dtype>[A-Z _ - a-z]+)(?<number>\d*).(?<ext>\w+)"
  m=match(myregex,lfname)
  ext = m[:ext]; dtype=m[:dtype];num = m[:number]
  (dtype != "" && ext != "")  || error("wrong filename format: dtype[n].csv")
  dtype in string.(instances(TSType)) || error(dtype * ", filename does not indicate known data type.")
  # create a pipeline to get stat
  fname = joinpath(ldirname,lfname)
  csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
  valgator = DateValgator(Dict(:dateinterval=>Dates.Hour(1)))
  valnner = DateValNNer(Dict(:dateinterval=>Dates.Hour(1)))
  stfier = Statifier(Dict(:processmissing=>false))
  mpipeline = Pipeline(Dict(
      :transformers => [csvfilter,valgator,valnner,stfier]
     )
  )
  fit!(mpipeline)
  df = transform!(mpipeline)
  df[!,:dtype] .= dtype
  df[!,:fname] .= lfname
  return (df)
end

function serialloop(ldirname,mfiles)
  trdata = DataFrame()
  for file in mfiles
    try
      df=getfilestat(ldirname,file)
      trdata = vcat(trdata,df)
      println("getting stats of "*file)
    catch errormsg
      println("skipping "*file*": "*string(errormsg))
      continue
    end
  end
  return trdata
end


function threadloop(ldirname,mfiles)
  @eval using Base.Threads
  trdata = DataFrame()
  mutex = SpinLock()
  Base.Threads.@threads for file in mfiles
    try
      df=getfilestat(ldirname,file)
      lock(mutex)
      trdata = vcat(trdata,df)
      println("getting stats of "*file*" on thread:"*string(Base.Threads.threadid()))
      unlock(mutex)
    catch errormsg
      println("skipping "*file*": "*string(errormsg))
    end
  end
  return trdata
end

# loop over the directory and get stats of each file
# return a dataframe containing stat features and ts type for target
function getstats(ldirname::AbstractString)
  ldirname != "" || error("directory name empty")
  mfiles = readdir(ldirname) |> x->filter(y->match(r".csv",y) != nothing,x)
  mfiles != [] || error("empty csv directory")
  # get julia version and run threads if julia 1.3
  jversion = string(Base.VERSION)
  df = DataFrame()
  if match(r"^1.3",jversion) === nothing
    df = serialloop(ldirname,mfiles)
  else
    df = threadloop(ldirname,mfiles)
  end
  return df
end

end
