module Plotters

#using Interact
using Random
using RecipesBase
using DataFrames
using Dates

using ..AbsTypes
using ..Utils
using ..ValDateFilters
import ..AbsTypes: fit, fit!, transform, transform!

export fit, fit!, transform, transform!
export Plotter

## setup plotting for publication
#function setupplot(pdfoutput::Bool)
#  Plots.gr();
#  fntsm = Plots.font("sans-serif", 8);
#  fntlg = Plots.font("sans-serif", 8);
#  Plots.default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm);
#  if pdfoutput == true 
#    Plots.default(size=(390,200)); #Plot canvas size
#  else
#    Plots.default(size=(500,300)); #Plot canvas size
#  end
#  return nothing
#end

struct DateVal{D<:DateTime,V<:Union{Missing,Real}}
   dtime::Vector{D}
   values::Vector{V}
end

@recipe function plot(dt::DateVal,sz=(390,200))
   marker := :none
   markertype --> :none
   seriestype := :path
   fontfamily := "sans-serif"
   titlefont := 8
   size := sz
   return (dt.dtime,dt.values)
end

"""
Plotter(
  Dict(
    :interactive => false,
    :pdfoutput => false
  )
)

Plots a TS by default but performs interactive plotting if specified during instance creation.
- `:interactive` => boolean to indicate whether to use interactive plotting with `false` as default
- `:pdfoutput` => boolean to indicate whether ouput will be saved as pdf with `false` as default

Example:

csvfilter = CSVDateValReader(Dict(:filename=>fname,:dateformat=>"dd/mm/yyyy HH:MM"))
pltr = Plotter(Dict(:interactive => false))

mpipeline = @pipeline csvfilter |> pltr
myplot = fit_transform!(mpipeline)

Implements: `fit!`, `transform!`
"""
mutable struct Plotter <: Transformer
   name::String
   model::Dict{Symbol,Any}

   function Plotter(args=Dict())
      default_args = Dict(
         :name => "pltr",
         :interactive => false,
         :pdfoutput => false
      )
      #setupplot(margs[:pdfoutput])
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

"""
fit!(pltr::Plotter, features::T, labels::Vector=[])

Check validity of `features`: 2-column Date,Val data
"""
function fit!(pltr::Plotter, features::DataFrame, labels::Vector=[])::Nothing
   ncol(features) == 2 || throw(ArgumentError("dataframe must have 2 columns: Date, Val"))
   return nothing
end

function fit(pltr::Plotter, features::DataFrame, labels::Vector=[])::Plotter
   fit!(pltr,features,labels)
   return deepcopy(pltr)
end

"""
transform!(pltr::Plotter, features::T) 

Convert `missing` into `NaN` to allow plotting of discontinuities.
"""
function transform!(pltr::Plotter, features::DataFrame)::DataFrame
   features != DataFrame() || return DataFrame()
   ncol(features) == 2 || throw(ArgumentError("dataframe must have 2 columns: Date, Val"))
   sum(names(features) .== ("Date","Value"))  == 2 || throw(ArgumentError("wrong column names"))
   # covert missing to NaN
   df = deepcopy(features)
   df.Value = Array{Union{Missing, Float64,eltype(features.Value)},1}(missing,nrow(df))
   df.Value .= features.Value
   ndxmissing = findall(x->ismissing(x),df.Value)
   df.Value[ndxmissing] .= NaN
   #setupplot(pltr.model[:pdfoutput])
   #if pltr.model[:interactive] == true && pltr.model[:pdfoutput] == false
   #  # disable interactive plot due to Knockout.jl badly maintained (Interact.jl deps)
   #  #interactiveplot(df)
   #  pl=Plots.plot(df.Date,df.Value,xlabel="Date",ylabel="Value",legend=false,show=false);
   #  return pl
   #else
   #  pl=Plots.plot(df.Date,df.Value,xlabel="Date",ylabel="Value",legend=false,show=false);
   #  return pl
   #end
   sz = (500,300) # default
   if pltr.model[:pdfoutput] == true
      sz = (390,200)
   end
   dtval = DateVal(df.Date,df.Value)
   RecipesBase.plot(dtval,sz)
   return features
end

function transform(pltr::Plotter, features::DataFrame)::DataFrame
   return transform!(pltr,features)
end

#function interactiveplot(df::Union{Vector,Matrix,DataFrame})
#  mlength = length(df.Value)
#  @manipulate for min in slider(1:mlength,label="min",value=1),max in slider(1:mlength,label="max",value=mlength)
#     Plots.plot(df.Date[min:max],df.Value[min:max],xlabel="Date",ylabel="Value",legend=false,show=false);
#  end
#end


end
