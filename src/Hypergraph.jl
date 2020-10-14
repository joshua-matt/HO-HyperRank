"""
Hypergraph data structure.
"""

using MatrixNetworks
using SparseArrays

mutable struct MatrixHypergraph
   incidence::SparseMatrixCSC{Int32,Int32}
end

function MatrixHypergraph(edges::Vector{Vector{Int32}}, n::Int32, m::Int32)
   matrix = spzeros(Int32,n,m)

   for e = 1:length(edges)
      for v in edges[e]
         matrix[v,e] = 1
      end
   end

   return MatrixHypergraph(matrix)
end

function MatrixHypergraph(edges::Vector{Vector{Int32}})
   return MatrixHypergraph(edges, maximum([e[i] for e in edges for i = 1:length(e)]), Int32(length(edges)))
end

function read(file::String; separator::String=" ")
   edges::Vector{Vector{Int32}} = []
   open(file) do f
      for ln in readlines(f)[1:1000]
         push!(edges, parse.(Int32,split(ln,separator)))
      end
   end

   return MatrixHypergraph(edges)
end

function read_arb(folder::String)
   edges::Vector{Vector{Int32}} = []
   name = match(r"\\([^\\]+)$",folder).captures[1]
   open("$folder\\$name-nverts.txt") do nverts
      open("$folder\\$name-simplices.txt") do simplices
         vertices = parse.(Int32, readlines(simplices))
         i = 1
         for n in readlines(nverts)[1:10000]
            size = parse(Int32,n)
            push!(edges, vertices[i:i+size-1])
            i += size
         end
      end
   end

   return MatrixHypergraph(edges)
end

function dyadic_projection(M::MatrixHypergraph)
   Z = M.incidence * M.incidence'
   Z -= Diagonal(Z)
   return MatrixNetwork(Z)
end

"""
`dual`
======

Returns the dual of the given hypergraph.
"""
function dual(M::MatrixHypergraph)
   return MatrixHypergraph(M.incidence')
end
