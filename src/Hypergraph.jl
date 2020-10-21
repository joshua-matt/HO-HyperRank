"""
Hypergraph data structure.
"""

using MatrixNetworks # Used for holding the projected hypergraph
using SparseArrays # Used to represent incidence matrix
using LinearAlgebra # Used for transpose

"""
`MatrixHypergraph`
==================

A data structure that encodes a hypergraph as an incidence matrix.

Fields
------
   - `incidence::SparseMatrixCSC{Int64,Int64}`: A matrix where entry ij is 1 if node
                                                i participates in hyperedge j, 0 otherwise.
"""
mutable struct MatrixHypergraph
   incidence::SparseMatrixCSC{Int64,Int64}
end

"""
`MatrixHypergraph`
==================

Constructs a MatrixHypergraph from a list of edges and given dimensions.

Arguments
---------
   - `edges::Vector{Vector{Int64}}`: A list of hyperedges, where `edges[i]` contains the nodes in the ith hyperedge.
   - `n::Int64`: The number of nodes in the hypergraph.
   - `m::Int64`: The number of hyperedges in the hypergraph.

Returns
-------
A `MatrixHypergraph` where the incidence matrix is `n` by `m` and the jth column
represents `edges[j]`.
"""
function MatrixHypergraph(edges::Vector{Vector{Int64}}, n::Int64, m::Int64)
   matrix = spzeros(Int64,n,m)

   for e = 1:length(edges)
      for v in edges[e]
         matrix[v,e] = 1
      end
   end

   return MatrixHypergraph(matrix)
end

"""
`MatrixHypergraph`
==================

Constructs a MatrixHypergraph from a list of edges.

Arguments
---------
   - `edges::Vector{Vector{Int64}}`: A list of hyperedges, where `edges[i]` contains the nodes in the ith hyperedge.

Returns
-------
A `MatrixHypergraph` where the number of rows in the incidence matrix is the max observed node ID in `edges`,
and m is `length(edges)`. The jth column represents `edges[j]`.
"""
function MatrixHypergraph(edges::Vector{Vector{Int64}})
   return MatrixHypergraph(edges, maximum([e[i] for e in edges for i = 1:length(e)]), Int64(length(edges)))
end

"""
`read`
======

Reads in a hypergraph from a file, where each line is a list of integers
representing a hyperedge.

Arguments
---------
   - `file::String`: The location of the file to read in.
   - `separator::String(=" ")`: The string separating the nodes on each line.

Returns
-------
A `MatrixHypergraph` where the number of rows in the incidence matrix is the max node ID in the file
and the number of columns is the number of lines in the file.
"""
function read(file::String; separator::String=" ")
   edges::Vector{Vector{Int64}} = []
   open(file) do f # Open file
      for ln in readlines(f)
         push!(edges, parse.(Int64,split(ln,separator))) # Split and parse edge
      end
   end

   return MatrixHypergraph(edges)
end

"""
`read_arb`
==========

Read in a hypergraph from a data folder that contains each of the following files:
   - `DATA-nverts.txt`, where the jth line is the number of nodes in the jth hyperedge.
   - `DATA-simplices.txt`, a contiguous list of the nodes comprising the hyperedges, where the
      ordering of the hyperedges is the same as in the first file. (Taken from DATA-DESCRIPTION.txt in the Enron folder)
The name of the folder and the prefix "DATA" above should be the same.

Arguments
---------
   - `folder::String`: The location of the data folder.

Returns
-------
A `MatrixHypergraph` where the number of rows in the incidence matrix is the maximum
value in the simplices file, and the number of columns is the number of lines in the
nverts file. The jth column represents simplices[nverts[j]:nverts[j+1]-1].
"""
function read_arb(folder::String)
   edges::Vector{Vector{Int64}} = []
   name = match(r"\\([^\\]+)$",folder).captures[1]
   open("$folder\\$name-nverts.txt") do nverts
      open("$folder\\$name-simplices.txt") do simplices
         vertices = parse.(Int64, readlines(simplices))
         i = 1
         for n in readlines(nverts)
            size = parse(Int64,n)
            push!(edges, vertices[i:i+size-1])
            i += size
         end
      end
   end

   return MatrixHypergraph(edges)
end

"""
`dyadic_projection`
===================

Returns the dyadic projection of a hypergraph, where the vertices are the same,
and two vertices are linked if they share a hyperedge. Additionally, the weight
on a link is the number of hyperedges shared by the endvertices.

Arguments
---------
   - `M::MatrixHypergraph`: The hypergraph to project.

Returns
-------
A `MatrixNetwork` corresponding to the projected graph.
"""
function dyadic_projection(M::MatrixHypergraph)
   Z = M.incidence * M.incidence'
   Z -= Diagonal(Z)
   return MatrixNetwork(Z)
end

"""
`dual`
======

Returns the dual of the given hypergraph. This is the hypergraph such that
the vertices represent hyperedges from the original, and a hyperedge exists
between vertices if the corresponding hyperedges shared a node in the original.

Arguments
---------
   - `M::MatrixHypergraph`: The hypergraph to get the dual of.

Returns
-------
A `MatrixHypergraph` where the incidence matrix is the transpose of `M.incidence`.
"""
function dual(M::MatrixHypergraph)
   return MatrixHypergraph(M.incidence')
end
