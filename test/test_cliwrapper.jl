module TestCLIWrapper

using Test
using TSML

using TSML.CLIWrappers:tsmlrun,rawstat,aggregatedstat,aggregatedoutput,imputedstat,imputedoutput

const inputname = joinpath(dirname(pathof(TSML)),"../data/testdata.csv")
const outputname = joinpath(dirname(pathof(TSML)),"../data/testdata_output.csv")

function test_cli()
  op1=tsmlrun(inputname,"")
  op2=imputedoutput(inputname,"")
  @test mean(op1.Value .== op2.Value) == 1.0
  op5=tsmlrun(inputname,"","dd/mm/yyyy HH:MM","stat")
  op6=imputedstat(inputname,"")
  @test (op5[1:1,:] .== op6[1:1,:]) |> Matrix |> sum == 20
end
@testset "CLIWrapper: check cliwrapper" begin
  test_cli()
end


end
