module DataProc
export mrun
export prun

using DataFrames
using RDatasets


iris=dataset("datasets","iris")

function mrun()
	iris |> x-> first(x,5)
end


function prun()
	iris |> x-> last(x,10)
end

end

