import AbstractTrees
import AbstractTrees.children
import AbstractTrees.printnode
using AbstractTrees: print_tree
using TSML

# From  https://github.com/JuliaOpt/Convex.jl/blob/master/src/utilities/tree_print.jl
# Plot it with: https://github.com/JuliaPlots/GraphRecipes.jl#abstracttrees-trees

# extennd abstract trees
children(d::Dict) = [p for p in d]
children(p::Pair) = AbstractTrees.children(p[2])

function printnode(io::IO, p::Pair)
  v = AbstractTrees.children(p[2])
  str = isempty(v)  ? string(p[1], ": ", p[2]) : string(p[1], ": ")
  print(io, str)
end

function showtree(p::Transformer)
  if isnothing(p.model)
    return
  elseif :transformers in keys(p.model) # pipeline
    println(:pipeline)
    print_tree(p.model[:transformers])
  else
    println(:transfomer)
    print_tree(p.model) # any transformer except pipeline
  end
end

function test_print_tree()
  there = Dict(:ok=>:yes)
  bb=Dict(:hello=>there)
  a=Dict(:b => bb, :d=>:e)
  d = Dict(:a => 2,:d => Dict(:b => 4,:c => "Hello"),:e => 5.0)
  print_tree(a)
  print_tree(d)
end

