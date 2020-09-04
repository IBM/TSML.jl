module TestTSML
using Test

include("test_plotter.jl")
include("test_mlbase.jl")
include("test_valdate.jl")
include("test_statifier.jl")
include("test_monotonicer.jl")
include("test_outliernicer.jl")
include("test_normalizer.jl")
include("test_ensemble.jl")
include("test_cliwrapper.jl")
include("test_tsclassifier.jl")

#include("test_timescaledb.jl")
#include("test_schemalizer.jl")

end
