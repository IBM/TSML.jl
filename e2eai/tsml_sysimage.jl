using PackageCompiler
create_sysimage(["AMLPipelineBase","DataFrames","StatsBase",
                  "ArgParse","TSML","CSV","Dates","Distributed",
                  "Random","ArgParse","Test","Distributed",
                  "Statistics","Serialization","Test"]; 
                  sysimage_path="tsml.so",
                  include_transitive_dependencies=false,
                  precompile_execution_file="tsml_precompile.jl")
