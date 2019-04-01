module TestTSML
using Test
using TSML.System

include("test_valdate.jl")
include("test_decisiontree.jl")

if LIB_SKL_AVAILABLE
    include("test_scikitlearn.jl")
else
    @info "Skipping scikit-learn tests."
end
if LIB_CRT_AVAILABLE
    include("test_caret.jl")
else
    @info "Skipping CARET tests."
end

end
