#!/julia-1.0.3/bin/julia

using TSML
using DataFrames
using TSML.CLIWrappers


function runme(args)
  res=DataFrames.DataFrame()
  if length(args) == 0
    println("no input/output filename")
    println("syntax: /run.jl input output dateformat")
  elseif length(args) == 1
    res=tsmlrun(args[1])
  elseif length(args) == 2
    res=tsmlrun(args[1],args[2])
  elseif length(args) == 3
    res=tsmlrun(args[1],args[2],args[3])
  elseif length(args) == 4
    res=tsmlrun(args[1],args[2],args[3],args[4])
  end
  res
end

runme(ARGS)
