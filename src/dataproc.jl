module DataProc
export mrun
export prun
using TSML.Utils

using DataFrames

iris=getiris()

function mrun()
	iris |> x-> first(x,5)
end


function prun()
	iris |> x-> last(x,10)
end

end

