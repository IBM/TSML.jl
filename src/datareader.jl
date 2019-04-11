module DataReaders
using Queryverse
using DataFrames
using HDF5
using JLD
using BSON
using Feather
using Dates

export DataReader, fit!, transform!
export Transformer
export datareaderrun

using TSML
using TSML.TSMLTypes
import TSML.TSMLTypes.fit! # to overload
import TSML.TSMLTypes.transform! # to overload
using TSML.Utils

mutable struct DataReader <: Transformer
    model
    args

    function DataReader(args=Dict())
        default_args=Dict(
            :filename => "",
            :impl_args => Dict()
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(dtr::DataReader,x::T=[],y::Vector=[]) where {T<:Union{DataFrame,Vector,Matrix}}
    dtr.model = dtr.args
end

function transform!(dtr::DataReader,x::T=[]) where {T<:Union{DataFrame,Vector,Matrix}}
    fullname = dtr.args[:filename]
    fname = basename(fullname)
    fname != "" || error("filename is empty")
    fn,ext=split(fname,".")
    ext != "" || error("no filename extension format")
    fmt = Dict("csv"=>Val(:csv),"feather"=>Val(:feather),"hdf5"=>Val(:hdf5),
          "h5"=>Val(:hdf5),"parquet"=>Val(:parquet))
    ext in keys(fmt)  || error("extension not recognized "*ext)
    readfmt(fmt[ext],fullname)
end

function readfmt(::T,fname::String) where {T<:Union{Val{:csv},Val{:feather},Val{:parquet}}}
    load(fname) |> DataFrame
end

function datareaderrun()
    lcsv=DataReader(Dict(:filename=>joinpath(dirname(pathof(TSML)),"../data/testdateval.csv")))
    fit!(lcsv)
    first(transform!(lcsv),5)
    lfeather=DataReader(Dict(:filename=>joinpath(dirname(pathof(TSML)),"../data/testdateval.feather")))
    fit!(lfeather)
    first(transform!(lfeather),5)
end

end
