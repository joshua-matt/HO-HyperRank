include("Hypergraph.jl")
include("weighted_pagerank.jl")
using SparseArrays

function motif_pagerank(G::Matrix, M::Matrix; d=0.5, α=0.85, n::Integer=100, ϵ=1.0e-6) # TODO: I think the problem is that this pagerank only operates on unweighted graphs? Or with weighting procedure?
    #return pagerank(SimpleWeightedGraph(sparse((d*G + (1-d)*M))), α, n, ϵ)
    return pagerank(SimpleWeightedGraph(sparse(G)))
end

function all_motifs(id::Int64)
    ### PRECONDITION CHECKS ###
    if !(1 <= id <= 26) throw(ArgumentError("Expected 1 ⩽ id ⩽ 26, received id = $id.")) end


end

G = dyadic_projection(Hypergraph([[1,2,3], [2,3], [3,5,6], [4]], 7, 4))
M = dyadic_projection(Hypergraph([[1,3,4], [2,3], [3,5,6], [4]], 7, 4))
#println(motif_pagerank(Matrix(G.weights), Matrix(M.weights); d=1, n=100))
println(weight(G,3))
