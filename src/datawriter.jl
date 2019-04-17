module DataWriters
using FileIO
using CSVFiles
using FeatherFiles
using DataFrames
using HDF5
using JLD
using Feather
using Parquet
using Dates

export DataWriter, fit!, transform!
export Transformer
export datawriterrun

using TSML
using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils

using TSML.DataReaders
using TSML.DataReaders: FILEFMT, DATEFMT



mutable struct DataWriter <: Transformer
    model
    args

    function DataWriter(args=Dict())
        default_args=Dict(
            :filename => "",
            :dateformat => DATEFMT,
            :impl_args => Dict()
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(dtr::DataWriter,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}
    fname = dtr.args[:filename]
    fmt = dtr.args[:dateformat]
    (fname != "" && fmt != "") || error("missing filename or date format")
    dtr.model = dtr.args
end

function transform!(dtr::DataWriter,x::T=[]) where {T<:Union{DataFrame,Vector,Matrix}}
    fullname = dtr.args[:filename]
    fmt = dtr.args[:dateformat]
    (fullname != "" && fmt != "") || error("missing filename or date format")
    fname = basename(fullname)
    fname != "" || error("filename is empty")
    fn,ext=split(fname,".")
    ext != "" || error("no filename extension format")
    ext in keys(FILEFMT)  || error("extension not recognized "*ext)
    # dispatch based on extension
    writefmt(FILEFMT[ext],fullname,x,fmt)
end

function writefmt(::T,fname::String,data::S,datefmt::String) where {T<:Union{Val{:csv},Val{:feather},Val{:parquet}},S<:Union{Matrix,DataFrame}}
    df = data |> DataFrame
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    eltype(df[:Date]) <: DateTime || error("Date format error")
    df[:Date] = Dates.format.(df[:Date],datefmt)
    df |> save(fname)
end

function writefmt(atype::Union{Val{:hdf5},Val{:jld}},fname::String, data::T,datefmt::String) where {T<:Union{Matrix,DataFrame}}
    df = data |> DataFrame
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    eltype(df[:Date]) <: DateTime || error("Date format error")
    df[:Date] = Dates.format.(df[:Date],datefmt)
    if atype == Val{:hdf5}
        fileopen = h5open
    else
        fileopen = jldopen
    end
    ldate = df[:Date]
    lvalue = df[:Value]
    fileopen(fname,"w") do file
        write(file,"dateval/date",ldate)
        write(file,"dateval/value",lvalue)
    end
end

function datawriterrun()
    fname = joinpath(dirname(pathof(TSML)),"../data/testdateval.csv")
    lcsv=DataReader(Dict(:filename=>fname))
    fit!(lcsv)
    dateval=transform!(lcsv)
    @show "csv"
    @show first(dateval,2)
    csvname = replace(fname,"test"=>"out")
    wcsv = DataWriter(Dict(:filename=>csvname))
    fit!(wcsv)
    transform!(wcsv,dateval)
    # check hdf5
    hdf5name = replace(fname,"csv"=>"h5")
    lhdf5 = DataWriter(Dict(:filename=>hdf5name))
    fit!(lhdf5)
    transform!(lhdf5,dateval)
    whdf5 = DataReader(Dict(:filename=>hdf5name))
    fit!(whdf5)
    res = transform!(whdf5)
    @show "hdf5"
    @show first(res,2)
    # check feather
    feathername = replace(fname,"csv"=>"feather")
    lfeather = DataWriter(Dict(:filename=>feathername))
    fit!(lfeather)
    transform!(lfeather,dateval)
    wfeather = DataReader(Dict(:filename=>feathername))
    fit!(wfeather)
    res = transform!(wfeather)
    @show "feather"
    @show first(res,2)
    # check jld
    jldname = replace(fname,"csv"=>"jld")
    ljld = DataWriter(Dict(:filename=>jldname))
    fit!(ljld)
    transform!(ljld,dateval)
    wjld = DataReader(Dict(:filename=>jldname))
    fit!(wjld)
    res = transform!(wjld)
    @show "jld"
    @show first(res,2)
    return nothing
end

end
