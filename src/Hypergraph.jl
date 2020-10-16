"""
Hypergraph data structure.
"""

using MatrixNetworks
using SparseArrays
using LinearAlgebra

mutable struct MatrixHypergraph
   incidence::SparseMatrixCSC{Int64,Int64}
end

function MatrixHypergraph(edges::Vector{Vector{Int64}}, n::Int64, m::Int64)
   matrix = spzeros(Int64,n,m)

   for e = 1:length(edges)
      for v in edges[e]
         matrix[v,e] = 1
      end
   end

   return MatrixHypergraph(matrix)
end

function MatrixHypergraph(edges::Vector{Vector{Int64}})
   return MatrixHypergraph(edges, maximum([e[i] for e in edges for i = 1:length(e)]), Int64(length(edges)))
end

function read(file::String; separator::String=" ")
   edges::Vector{Vector{Int64}} = []
   open(file) do f
      for ln in readlines(f)
         push!(edges, parse.(Int64,split(ln,separator)))
      end
   end

   return MatrixHypergraph(edges)
end

function read_arb(folder::String)
   edges::Vector{Vector{Int64}} = []
   name = match(r"\\([^\\]+)$",folder).captures[1]
   open("$folder\\$name-nverts.txt") do nverts
      open("$folder\\$name-simplices.txt") do simplices
         vertices = parse.(Int64, readlines(simplices))
         i = 1
         for n in readlines(nverts)[1:10000]
            size = parse(Int64,n)
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
