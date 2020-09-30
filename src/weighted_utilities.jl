SimpleWeightedGraphs.weight(G::SimpleWeightedGraph, v::Int64) = sum(G.weights.nzval[G.weights.colptr[v]:G.weights.colptr[v+1]-1])
