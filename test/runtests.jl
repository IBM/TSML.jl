module TestTSML
using Test

include("test_valdate.jl")
include("test_decisiontree.jl")
include("test_statifier.jl")
include("test_monotonicer.jl")
include("test_cliwrapper.jl")
include("test_outliernicer.jl")

end
