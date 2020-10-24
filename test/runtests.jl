module TestTSML
using Test

include("test_tsclassifier.jl")
include("test_mlbase.jl")
include("test_valdate.jl")
include("test_statifier.jl")
include("test_monotonicer.jl")
include("test_outliernicer.jl")
include("test_ensemble.jl")
include("test_cliwrapper.jl")
include("test_normalizer.jl")
include("test_svm.jl")

# test if running windows
if !Base.Sys.iswindows()
  include("test_plotter.jl")
end

#include("test_timescaledb.jl")
#include("test_schemalizer.jl")

end
