module CLIWrappers

using Random
using Dates
using DataFrames
using CSV

using ..DecisionTreeLearners
using ..Pipelines
using ..AbsTypes
using ..Utils

using ..ValDateFilters
using ..Statifiers
using ..Monotonicers

export tsmlrun

const COMMONARG = Dict(
                    :dateformat=>"dd/mm/yyyy HH:MM",
                    :dateinterval=>Dates.Hour(1),
                    :processmissing=>true,
                    :strict => false
                  )

function tsmlrun(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM",otype::AbstractString="table")
   if otype == "table"
      imputedoutput(inputname,outputname,datefmt)
   elseif otype == "stat"
      imputedstat(inputname,outputname,datefmt)
   else
      throw(ArgumentError("wrong output type"))
   end
end

function rawstat(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM")
   isfile(inputname) || throw(ArgumentError("input file name does not exist"))
   csvreader = CSVDateValReader(Dict(:filename=>inputname,:dateformat=>datefmt))
   stfier = Statifier()
   lpipe = @pipeline csvreader |> stfier
   res=fit_transform!(lpipe)
   if outputname != ""
      res |> CSV.write(outputname)
   end
   return res
end

function aggregatedstat(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM")
   isfile(inputname) || throw(ArgumentError("input file name does not exist"))
   csvreader = CSVDateValReader(Dict(:filename=>inputname,:dateformat=>datefmt))
   valgator = DateValgator(COMMONARG)
   stfier = Statifier()
   lpipe = @pipeline csvreader |> valgator |> stfier
   res=fit_transform!(lpipe)
   if outputname != ""
      res |> CSV.write(outputname)
   end
   return res
end

function aggregatedoutput(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM")
   isfile(inputname) || throw(ArgumentError("input file name does not exist"))
   csvreader = CSVDateValReader(Dict(:filename=>inputname,:dateformat=>datefmt))
   valgator = DateValgator(COMMONARG)
   lpipe = @pipeline csvreader |> valgator
   res=fit_transform!(lpipe)
   if outputname != ""
      res |> CSV.write(outputname)
   end
   return res
end

function imputedstat(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM")
   isfile(inputname) || throw(ArgumentError("input file name does not exist"))
   csvreader = CSVDateValReader(Dict(:filename=>inputname,:dateformat=>datefmt))
   valgator = DateValgator(COMMONARG)
   valnner = DateValNNer(COMMONARG)
   mononicer = Monotonicer()
   stfier = Statifier()
   lpipe = @pipeline csvreader |> valgator |> valnner |> mononicer |> stfier
   res=fit_transform!(lpipe)
   if outputname != ""
      res |> CSV.write(outputname)
   end
   return res
end

function imputedoutput(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM")
   isfile(inputname) || throw(ArgumentError("input file name does not exist"))
   csvreader = CSVDateValReader(Dict(:filename=>inputname,:dateformat=>datefmt))
   valgator = DateValgator(COMMONARG)
   valnner = DateValNNer(COMMONARG)
   mononicer = Monotonicer()
   stfier = Statifier()
   lpipe = @pipeline csvreader |> valgator |> valnner |> mononicer
   res=fit_transform!(lpipe)
   if outputname != ""
      res |> CSV.write(outputname)
   end
   return res
end

end
