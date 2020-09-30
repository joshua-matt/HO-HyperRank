"""
Hypergraph data structure.
"""

using LightGraphs, SimpleWeightedGraphs

"""
`Hypergraph{T}`
=====================
Represents a vertex-labeled hypergraph as a list of edges.
Fields
------
   - `vals::Vector{T}`: The values corresponding to each node (or hyperedge, depending on use case)
   - `n::Int64`: The number of nodes
   - `m::Int64`: The number of hyperedges
   - `D::Vector{Int64}`: The degree sequence, where `D[i] = degree of node i`
   - `K::Vector{Int64}`: The edge dimension sequence, where `K[j] = size of edge j`
   - `edges::Vector{Vector{Int64}}`: The hyperedges and their members
A hypergraph is a generalization of a graph in which edges can contain any number
of nodes. So, for example, if we wanted to denote a three-way relationship between
nodes 1, 2, and 3, the hypergraph would contain the edge `{1, 2, 3}`.
Hypergraphs are useful because they allow us to more accurately indicate relationships
among groups of things. Consider a coauthorship network. In a dyadic graph,
we might represent a five-way collaboration as a 5-clique. While this does capture
the fact that all pairs of authors have now appeared in a paper together, it doesn't seem
right to categorize the event as 15 pairwise interactions. With a hypergraph, we
can more succinctly and intuitively represent the event as a single hyperedge of five nodes.
"""
mutable struct Hypergraph{T}
   vals::Vector{T}
   n::Int64
   m::Int64
   D::Vector{Int64}
   K::Vector{Int64}
   edges::Vector{Vector{Int64}}
end

"""
`Hypergraph_kernel`
===================
Verifies that:
   - The number of edges == `m`
   - The number of vals == `n`
   - All nodes are between 0 and `n`
If all conditions are met, returns degree and edge dimension sequences
"""
function Hypergraph_kernel(edges::Vector{Vector{Int64}}, vals::Vector{T},
                           n::Int64, m::Int64) where T
   @assert length(edges) == m # m is the number of edges
   @assert length(vals) == n # Each node has a val associated with it
   @assert all([0 < edges[i][j] < n + 1 for i = 1:m for j = 1:length(edges[i])]) # No node exceeds n

   D = zeros(Int64, n)
   K = zeros(Int64, m)
   for e = 1:m
      K[e] = size(edges[e],1)
      for v in edges[e]
         D[v] += 1
      end
   end
   edges = sort!.(edges, by=v -> D[v], rev=true) # Sort edges by descending node degree

   return D, K
end

"""
`Hypergraph` constructors
=============================
Functions
---------
   - `Hypergraph(edges, vals, n, m)`: Produces a hypergraph with the given edges, values, and size
   - `Hypergraph(edges, n, m)`: Produces a hypergraph with the given edge set and size, with all values as 1.0
   - `Hypergraph(edges)`: Produces a hypergraph with the given set of edges
Examples
--------
~~~~
Hypergraph([[1,2], [3,4], [1,4]], ["One", "Two", "Three", "Four"], 4, 3)
Hypergraph([[i,i+2] for i=1:3], 4, 3)
Hypergraph([[1,2,3], [2,4,6], [1,5]])
~~~~
"""
function Hypergraph(edges::Vector{Vector{Int64}}, vals::Vector{T},
                          n::Int64, m::Int64) where T
   D, K= Hypergraph_kernel(edges, vals, n, m)
   return Hypergraph(vals, n, m, D, K, edges)
end

function Hypergraph(edges::Vector{Vector{Int64}}, n::Int64, m::Int64)
   return Hypergraph(edges, ones(n), n, m)
end

function Hypergraph(edges::Vector{Vector{Int64}})
   return Hypergraph(edges, maximum([e[i] for e in edges for i = 1:length(e)]), length(edges))
end

Base.copy(H::Hypergraph) = Hypergraph(deepcopy(h.edges), deepcopy(h.vals), h.n, h.m)

"""
`dyadic_projection`
===================

Project a hypergraph to a weighted undirected graph.

Arguments
---------
   - `H::Hypergraph`: The hypergraph to project

Returns
-------
   - A `SimpleWeightedGraph` in which two nodes are connected iff they share
     a hyperedge and the edge weight is how many hyperedges they share
"""
function dyadic_projection(H::Hypergraph)
   weights = Dict()
   edges = sort.(H.edges)
   for e = 1:H.m
      for i = 1:H.K[e]
         for j in i+1:H.K[e]
            edge = H.edges[e]
            u, v = edge[i], edge[j]
            if !((u,v) in keys(weights))
               weights[(u,v)] = 0
            end
            weights[(u,v)] += 1
         end
      end
   end

   G = SimpleWeightedGraph(H.n)
   for e in keys(weights)
      add_edge!(G, e..., weights[e])
   end
   return G
end
