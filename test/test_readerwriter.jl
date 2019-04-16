module TestReaderWriter

using TSML
using TSML.Utils
using TSML.TSMLTypes
using TSML.TSMLTransformers

using TSML.DataReaders
using TSML.DataWriters

using DataFrames
using Dates
using Test

function test_readerwriter()
    gcdims::Tuple = (8761,2)
    ssum = 97564.44999999998
    resdf::DataFrame=DataFrame()
    fname = joinpath(dirname(pathof(TSML)),"../data/testdateval.csv")
    lcsv=DataReader(Dict(:filename=>fname))
    fit!(lcsv)
    dateval=transform!(lcsv)
    @test sum(size(dateval) .== gcdims ) == 2
    @test sum(dateval[:Value]) == ssum
    csvname = replace(fname,"test"=>"out")
    wcsv = DataWriter(Dict(:filename=>csvname))
    fit!(wcsv)
    transform!(wcsv,dateval)
    pcsv = DataReader(Dict(:filename=>csvname))
    fit!(pcsv)
    resdf=transform!(pcsv)
    @test sum(size(resdf) .== gcdims) == 2
    @test sum(resdf[:Value]) == ssum
    # check hdf5
    hdf5name = replace(fname,"csv"=>"h5")
    lhdf5 = DataWriter(Dict(:filename=>hdf5name))
    fit!(lhdf5)
    transform!(lhdf5,dateval)
    whdf5 = DataReader(Dict(:filename=>hdf5name))
    fit!(whdf5)
    resdf = transform!(whdf5)
    @test sum(size(resdf) .== gcdims) == 2
    @test sum(resdf[:Value]) == ssum
    # check feather
    feathername = replace(fname,"csv"=>"feather")
    lfeather = DataWriter(Dict(:filename=>feathername))
    fit!(lfeather)
    transform!(lfeather,dateval)
    wfeather = DataReader(Dict(:filename=>feathername))
    fit!(wfeather)
    resdf = transform!(wfeather)
    @test sum(size(resdf) .== gcdims) == 2
    @test sum(resdf[:Value]) == ssum
    # check jld
    jldname = replace(fname,"csv"=>"jld")
    ljld = DataWriter(Dict(:filename=>jldname))
    fit!(ljld)
    transform!(ljld,dateval)
    wjld = DataReader(Dict(:filename=>jldname))
    fit!(wjld)
    resdf = transform!(wjld)
    @test sum(size(resdf) .== gcdims) == 2
    @test sum(resdf[:Value]) == ssum
end
@testset "Data Readers/Writers: csv,hdf5,feather,jld" begin
    test_readerwriter()
end


end
