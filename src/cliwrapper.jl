module CLIWrappers

using AutoMLPipeline
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

using TSML.ValDateFilters
using TSML: CSVDateValReader
using TSML: CSVDateValWriter
using TSML.Statifiers
using TSML.Monotonicers

using Dates
using DataFrames
using CSV

export tsmlrun

const COMMONARG = Dict(:dateformat=>"dd/mm/yyyy HH:MM",
                        :dateinterval=>Dates.Hour(1),
                        :processmissing=>true
                       )

function tsmlrun(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM",otype::AbstractString="table")
    if otype == "table"
        imputedoutput(inputname,outputname,datefmt)
    elseif otype == "stat"
        imputedstat(inputname,outputname,datefmt)
    else
        error("wrong output type")
    end
end

function rawstat(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM")
    isfile(inputname) || error("input file name does not exist")
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
    isfile(inputname) || error("input file name does not exist")
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
    isfile(inputname) || error("input file name does not exist")
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
    isfile(inputname) || error("input file name does not exist")
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
    isfile(inputname) || error("input file name does not exist")
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
