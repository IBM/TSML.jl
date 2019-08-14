@reexport module ArgumentParsers

using TSML
using TSML.TSMLTypes
using TSML.TSMLTransformers
using TSML.Utils

using TSML: CSVDateValReader
using TSML: CSVDateValWriter
using TSML.Statifiers
using TSML.Monotonicers
using TSML.Outliernicers
using TSML.Plotters

using Dates
using DataFrames
using CSV
using ArgParse

export tsmlmain

const COMMONARG = Dict(
        :dateformat=>"dd/mm/yyyy HH:MM",
        :dateinterval=>Dates.Hour(1),
        :processmissing=>true
      )

const DATEINTERVAL = Dict(
        "minutely" => Dates.Minute(1),
        "hourly" => Dates.Hour(1),
        "daily"=> Dates.Day(1),
        "weekly" => Dates.Week(1),
        "monthly"=> Dates.Month(1),
        "yearly"=> Dates.Year(1)
       )



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--aggregate"
            help = "aggregate interval such as: minutely, hourly, weekly, monthly, Dates.Minute(30),Dates.Hour(2)"
            arg_type = String
            default = "hourly"
        "--dateformat"
            help = "date and time format"
            arg_type = String
            default = "dd/mm/yyyy HH:MM"
        "--impute"
            help = "impute with NN"
            action = :store_true
        "--mono"
            help = "normalize monotonic"
            action = :store_true
        "--clean"
            help = "remove outliers"
            action = :store_true
        "--stat"
            help = "get statistics"
            action = :store_true
        "--save"
            help = "save results"
            action = :store_true
        "csvfilename"
            help = "csv file to process"
            required = true
    end

    return parse_args(s,as_symbols=true)
end

function tsmlmain()
    parsed_args = parse_commandline()
    createrunpipeline(parsed_args)
end

function createrunpipeline(args::Dict)
    dateformat = args[:dateformat]
    fname = args[:csvfilename]
    csvreader = CSVDateValReader(Dict(:filename=>fname,:dateformat=>dateformat))

    csvwriter = CSVDateValWriter()
    fnameoutput = ""
    if args[:save] == true
        name = split(fname,".csv")[1]
        fnameoutput = name*"_output"*".csv"
        csvwriter.args[:filename] = fnameoutput
    end

    # extract date interval
    dateintervalstring = args[:aggregate]
    dateinterval = Dates.Hour(1)

    if dateintervalstring in keys(DATEINTERVAL)
       dateinterval  = DATEINTERVAL[dateintervalstring]
    else
        # parse the string into expression and evaluate
        # example: "Dates.Hour(1)" or "Dates.Minute(30)"
        dateinterval = eval(Meta.parse(dateintervalstring))
    end
    commonarg = Dict(:dateformat=>dateformat,:dateinterval=>dateinterval)
    valgator = DateValgator(commonarg)
    imputer = DateValNNer(commonarg)
    mono = Monotonicer()
    outliernicer = Outliernicer(commonarg)
    statifier = Statifier(Dict(:processmissing=>true))

    transformers = [csvreader,valgator]
    if args[:impute] == true
        transformers = [transformers;imputer]
    end
    if args[:mono] == true
        transformers = [transformers;mono]
    end
    if args[:clean] == true
        transformers = [transformers;outliernicer]
    end
    if args[:stat] == true
        transformers = [transformers;statifier]
    end
    if args[:save] == true
        transformers = [transformers;csvwriter]
    end

    pipeline = Pipeline()
    pipeline.args[:transformers]=transformers
    fit!(pipeline)
    transform!(pipeline)
end


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
    lpipe = Pipeline(Dict(
        :transformers => [csvreader,stfier]
      )
    )
    fit!(lpipe)
    res=transform!(lpipe)
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
    lpipe = Pipeline(Dict(
        :transformers => [csvreader,valgator,stfier]
      )
    )
    fit!(lpipe)
    res=transform!(lpipe)
    if outputname != ""
        res |> CSV.write(outputname)
    end
    return res
end

function aggregatedoutput(inputname::AbstractString,outputname::AbstractString="",datefmt::AbstractString="dd/mm/yyyy HH:MM")
    isfile(inputname) || error("input file name does not exist")
    csvreader = CSVDateValReader(Dict(:filename=>inputname,:dateformat=>datefmt))
    valgator = DateValgator(COMMONARG)
    lpipe = Pipeline(Dict(
        :transformers => [csvreader,valgator]
      )
    )
    fit!(lpipe)
    res=transform!(lpipe)
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
    lpipe = Pipeline(Dict(
        :transformers => [csvreader,valgator,valnner,mononicer,stfier]
      )
    )
    fit!(lpipe)
    res=transform!(lpipe)
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
    lpipe = Pipeline(Dict(
        :transformers => [csvreader,valgator,valnner,mononicer]
      )
    )
    fit!(lpipe)
    res=transform!(lpipe)
    if outputname != ""
        res |> CSV.write(outputname)
    end
    return res
end

end
