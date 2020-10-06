"""
Hypergraph data structure.
"""

using LightGraphs, SimpleWeightedGraphs
using SparseArrays
using Combinatorics

mutable struct MatrixHypergraph
   incidence::SparseMatrixCSC
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
   return MatrixHypergraph(edges, maximum([e[i] for e in edges for i = 1:length(e)]), length(edges))
end

function MatrixHypergraph(file::String; separator::String=" ")
   edges::Vector{Vector{Int64}} = []
   open(file) do f
      for ln in readlines(f)[1:200]
         push!(edges, parse.(Int64,split(ln,separator)))
      end
   end

   return MatrixHypergraph(edges)
end

function dyadic_projection(M::MatrixHypergraph)
   Z = M.incidence * M.incidence'
   Z -= Diagonal(Z)
   return SimpleWeightedGraph(Z)
end

"""
`get_hyperwedges`
=================

Finds all connected triplets of hyperedges in a hypergraph.
"""
function get_hyperwedges(M::MatrixHypergraph)
   wedges::Vector{Vector{Int64}} = []
   G::SimpleWeightedGraph = dyadic_projection(dual(M))

   neighbor_list::Dict{Int64,Vector{Int64}} = Dict(i => neighbors(G,i) for i = 1:nv(G))

   by_deg::Vector{Int64} = sort(1:nv(G),by=x->degree(G,x),rev=true)

   for u in by_deg
      neigh = neighbor_list[u]
      len = length(neigh)

      for v = 1:len
         for w in setdiff(neighbor_list[neigh[v]], [u]) # 2-paths starting at u, going through v
            push!(wedges, [u,neigh[v],w])
         end
         for w = v+1:len # Wedges centered at u including v
            push!(wedges, [neigh[v],u,neigh[w]])
         end
      end
      rem_vertex!(G,u)
      by_deg[(x->x>u).(by_deg)] .-= 1
   end
   #println(total_neigh/total_v)
   return wedges
end

"""
`dual`
======

Returns the dual of the given hypergraph.
"""
function dual(M::MatrixHypergraph)
   return MatrixHypergraph(transpose(M.incidence))
end
