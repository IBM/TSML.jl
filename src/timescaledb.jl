module TimescaleDBs

using HTTP, JSON2
using Dates
using DataFrames

using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!

export fit, fit!,transform, transform!
export TimescaleDB

# Transforms instances with nominal features into one-hot form
# and coerces the instance matrix to be of element type Float64.
mutable struct TimescaleDB <: Transformer
   name::String
   model::Dict{Symbol,Any}

   function TimescaleDB(args=Dict())
      default_args = Dict(
         :uri => "http://localhost:3000",
         :db => "dateval",
         :dbusername =>"postgres",
         :dbpassword => "mysecretpassword",
         :query => "select=date,value"
      )
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

function fit!(tdb::TimescaleDB, features::DataFrame=DataFrame(), labels::Vector=[])::Nothing
   (features == DataFrame() && labels == [])  || throw(ArgumentError("features and labels should be empty because data are from http request"))
   uri = tdb.model[:uri]; db = tdb.model[:db]; query = tdb.model[:query] 
   (uri != "" && query != "" && db != "") || error("missing uri/query/db")
   return nothing
end

function fit(tdb::TimescaleDB, features::DataFrame=DataFrame(), labels::Vector=[])::TimescaleDB
   fit!(tdb,features,labels)
   return deepcopy(tdb)
end

function transform!(tdb::TimescaleDB, features::DataFrame=DataFrame())::DataFrame
   features == DataFrame()  || throw(ArgumentError("features should be empty because data are from http request"))
   uri = tdb.model[:uri]; db = tdb.model[:db]; query = tdb.model[:query] 
   (uri != "" && query != "" && db != "") || error("missing uri/query/db")
   payload = uri*"/"*db*"?"*query
   req = HTTP.get(payload)
   req.status == 200 || error(req.status)
   body = String(req.body)
   df = body |> JSON2.read |> DataFrame
   ncol(df) == 2 || error("data should have 2 columns")
   nrow(df) > 0 || return DataFrame()
   rename!(df,names(df)[1] => :Date, names(df)[2] => :Value)
   df.Date = DateTime.(df.Date)
   eltype(df.Value) <: Number || throw(ArgumentError("values in second column are not numeric"))
   return df
end

function transform(tdb::TimescaleDB, features::DataFrame=DataFrame())::DataFrame
   return transform!(tdb,features)
end

end
